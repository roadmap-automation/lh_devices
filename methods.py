import asyncio
import copy
import json
import logging
import traceback
from typing import List, Dict, Any
from dataclasses import dataclass, field

from device import DeviceBase, DeviceError
from gsioc import GSIOC, GSIOCMessage, GSIOCCommandType
from logutils import Loggable, MethodLogHandler, JsonFormatter

# ======== Method base classes ==========

@dataclass
class MethodError(DeviceError):
    ...

class MethodException(Exception):
    
    def __init__(self, *args, retry: bool = False):
        super().__init__(*args)
        self.retry = retry

@dataclass
class MethodResult:
    """Method result information"""

    name: str
    method_data: dict
    result: dict

class MethodBase(Loggable):
    """Base class for defining a method for LH serial devices. Contains information about:
        1. dead volume calculations
        2. required configurations
        3. which devices are involved
        4. locks, signals, controls
    """

    def __init__(self, devices: List[DeviceBase] = []) -> None:
        self.devices = devices
        self.error = MethodError()
        self.dead_volume_node: str | None = None

        self.metadata = []

        # set up a unique logger for this method instance
        Loggable.__init__(self)
        log_handler = MethodLogHandler(self.metadata)
        log_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(log_handler)
        self.log_handler = log_handler

    @property
    def name(self):
        return self.MethodDefinition.name

    def is_ready(self) -> bool:
        """Gets ready status of method. Requires all devices to be idle
            and the running flag to be False

        Returns:
            bool: True if method can be run
        """

        devices_reserved = any(dev.reserved for dev in self.devices)
        return (not devices_reserved)
    
    def reserve_all(self) -> None:
        """Reserves all devices used in method
        """

        for dev in self.devices:
            dev.reserved = True

    def release_all(self) -> None:
        """Releases all devices
        """

        for dev in self.devices:
            dev.reserved = False

    async def trigger_update(self) -> None:
        """Triggers update of all devices
        """
        for dev in self.devices:
            await dev.trigger_update()

    @dataclass
    class MethodDefinition:
        """Subclass containing the method definition and schema"""

        name: str

    async def run(self, **kwargs) -> None:
        """Runs the method with the appropriate keywords
        """

        pass

    async def start(self, **kwargs) -> MethodResult:
        """Starts a method run with error handling
        """

        async def on_cancel():
            self.logger.info(f'{self.name} canceled, releasing and updating all devices')
            self.error.retry = False
            self.release_all()
            await self.trigger_update()

        # clear metadata before starting
        self.metadata.clear()

        self.logger.info(f'{self.name} starting')

        # add method handler to device loggers
        for device in self.devices:
            device.logger.addHandler(self.log_handler)

        try:
            self.error.clear()
            await self.run(**kwargs)
        except asyncio.CancelledError:
            await on_cancel()

        except MethodException as e:
            # these are critical errors
            self.error.error = str(e)
            self.error.retry = e.retry
            self.logger.error(f'Critical error in {self.name}: {e}, retry is {e.retry}, waiting for error to be cleared')
            try:
                await self.trigger_update()
                await self.error.pause_until_clear()
                if self.error.retry:
                    # try again!
                    self.logger.info(f'{self.name} retrying')
                    retry_metadata = copy.copy(self.metadata)
                    await self.start(**kwargs)
                    self.metadata = retry_metadata + self.metadata
            except asyncio.CancelledError:
                await on_cancel()
        finally:
            self.logger.info(f'{self.name} finished')
            
            # remove method handler from device loggers
            for device in self.devices:
                device.logger.removeHandler(self.log_handler)
        
        logging.info(self.metadata)

        return MethodResult(name=self.name,
                            method_data=kwargs,
                            result=copy.copy(self.metadata))

    async def throw_error(self, error: str, critical: bool = False, retry: bool = False) -> None:
        """Populates the method error. If a critical error, stops method execution. If not critical,
            pauses until error is cleared.

        Args:
            error (str): error description
            critical(bool, optional): critical error flag. If True, ends method execution. If False, waits for a clear error signal before continuing. Defaults to False.
            retry (bool, optional): retry flag. If True, method is restarted from the beginning. Only applies to critical errors. Defaults to False.
        """

        if critical:
            raise MethodException(error, retry=retry)
        
        else:
            self.logger.error(f'Non-critical error in {self.__class__}: {error}, waiting for error to be cleared')
            self.error.error = error
            self.error.retry = retry
            await self.error.pause_until_clear()

class MethodBasewithGSIOC(MethodBase):

    def __init__(self, gsioc: GSIOC, devices: List[DeviceBase] = []) -> None:
        super().__init__(devices)

        self.gsioc = gsioc

        # enables triggering
        self.waiting: asyncio.Event = asyncio.Event()
        self.trigger: asyncio.Event = asyncio.Event()

        # container for gsioc tasks 
        self._gsioc_tasks: List[asyncio.Task] = []

    def connect_gsioc(self) -> None:
        """Start GSIOC listener and connect."""

        # TODO: This opens and closes the serial port a lot. Might be better to just start the GSIOC listener and then connect to it through monitor_gsioc
        self._gsioc_tasks = [asyncio.create_task(self.monitor_gsioc())]

    async def monitor_gsioc(self) -> None:
        """Monitor GSIOC communications. Note that only one device should be
            listening to a GSIOC device at a time.
        """

        logging.debug('Starting GSIOC monitor')
        async with self.gsioc.client_lock:
            logging.debug('Got GSIOC client lock...')
            try:
                while True:
                    data: GSIOCMessage = await self.gsioc.message_queue.get()
                    logging.debug(f'GSIOC got data {data}')
                    response = await self.handle_gsioc(data)
                    if data.messagetype == GSIOCCommandType.IMMEDIATE:
                        await self.gsioc.response_queue.put(response)
            except asyncio.CancelledError:
                logging.debug("Stopping GSIOC monitor...")
            except Exception:
                raise

    def disconnect_gsioc(self) -> None:
        """Stop listening to GSIOC
        """

        for task in self._gsioc_tasks:
            task.cancel()

    async def wait_for_trigger(self) -> None:
        """Uses waiting and trigger events to signal that assembly is waiting for a trigger signal
            and then release upon receiving the trigger signal"""
        
        self.waiting.set()
        await self.trigger.wait()
        self.waiting.clear()
        self.trigger.clear()

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:
        """Handles GSIOC messages. Put actions into gsioc_command_queue for async processing.

        Args:
            data (GSIOCMessage): GSIOC Message to be parsed / handled

        Returns:
            str: response (only for GSIOC immediate commands, else None)
        """
        
        response = None

        if data.data == 'Q':
            # busy query
            if self.waiting.is_set():
                response = 'waiting'
            elif all(dev.idle for dev in self.devices):
                response = 'idle'
            else:
                response = 'busy'

        # set trigger
        elif data.data == 'T':
            self.waiting.clear()
            self.trigger.set()
            response = 'ok'

        else:
            response = 'error: unknown command'

        return response
    
class MethodBaseDeadVolume(MethodBasewithGSIOC):

    def __init__(self, gsioc: GSIOC, devices: List[DeviceBase] = []) -> None:
        super().__init__(gsioc, devices)

        self.dead_volume: asyncio.Queue = asyncio.Queue(1)

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:

        # overwrites base class handling of dead volume
        if data.data == 'V':
            dead_volume = await self.dead_volume.get()
            #logging.info(f'Sending dead volume {dead_volume}')
            response = f'{dead_volume:0.0f}'
        else:
            response = await super().handle_gsioc(data)
        
        return response


if __name__=='__main__':

    logging_stuff = []
    mlh = MethodLogHandler(logging_stuff)

    logging.basicConfig(handlers=[
                        logging.StreamHandler(),
                        mlh
                    ],
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
    
    mlh.setFormatter(JsonFormatter(datefmt='%Y-%m-%d %H:%M:%S'))

    logging.info('I logged this')
    logging.info({'meta_data': {'I did something': 'result was this'}})

    print(logging_stuff)