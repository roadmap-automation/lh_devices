import asyncio
import json
import logging
import traceback
from typing import List
from dataclasses import dataclass, field

from device import DeviceBase, DeviceError
from gsioc import GSIOC, GSIOCMessage, GSIOCCommandType

@dataclass
class MethodError(DeviceError):
    ...

class MethodException(Exception):
    
    def __init__(self, *args, retry: bool = False):
        super().__init__(*args)
        self.retry = retry

class MethodBase:
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

    async def start(self, **kwargs) -> None:
        """Starts a method run with error handling
        """

        async def on_cancel():
            logging.info(f'{self.MethodDefinition.name} canceled, releasing and updating all devices')
            self.error.retry = False
            self.release_all()
            self.trigger_update()

        try:
            self.error.clear()
            await self.run(**kwargs)
        except asyncio.CancelledError:
            await on_cancel()

        except MethodException as e:
            # these are critical errors
            self.error.error = str(e)
            self.error.retry = e.retry
            logging.error(f'Critical error in {self.MethodDefinition.name}: {e}, retry is {e.retry}, waiting for error to be cleared')
            try:
                self.trigger_update()
                await self.error.pause_until_clear()
                if self.error.retry:
                    # try again!
                    logging.info(f'{self.MethodDefinition.name} retrying')
                    await self.start(**kwargs)
            except asyncio.CancelledError:
                await on_cancel()

        logging.info(f'{self.MethodDefinition.name} finished')

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
            logging.error(f'Non-critical error in {self.__class__}: {error}, waiting for error to be cleared')
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
