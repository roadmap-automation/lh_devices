import asyncio
import copy
import json
import logging
import traceback
from uuid import uuid4
from typing import List, Dict, Any, Callable, TypedDict, Coroutine
from dataclasses import dataclass, field, fields, Field

from .device import DeviceBase, DeviceError
from .gsioc import GSIOC, GSIOCMessage, GSIOCCommandType
from .logutils import Loggable, MethodLogHandler, MethodLogFormatter

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

    id: str | None = None
    created_time: str = ''
    source: str = ''
    method_name: str = ''
    method_data: dict = field(default_factory=dict)
    finished_time: str = ''
    log: list = field(default_factory=list)
    result: dict = field(default_factory=dict)

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
        log_handler.setFormatter(MethodLogFormatter())
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

        # clear metadata and result before starting
        self.metadata.clear()
        result = {}

        self.logger.info(f'{self.name} starting')

        # add method handler to device loggers
        for device in self.devices:
            device.logger.addHandler(self.log_handler)

        try:
            self.error.clear()
            result = await self.run(**kwargs)
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
                    newresult = await self.start(**kwargs)
                    self.metadata = retry_metadata + self.metadata
                    result = newresult.result
            except asyncio.CancelledError:
                await on_cancel()
        finally:
            self.logger.info(f'{self.name} finished')
            
            # remove method handler from device loggers
            for device in self.devices:
                device.logger.removeHandler(self.log_handler)
        
        return MethodResult(method_name=self.name,
                            method_data=kwargs,
                            log=copy.copy(self.metadata),
                            created_time=self.metadata[0]['time'],
                            finished_time=self.metadata[-1]['time'],
                            result=result)

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

        self.logger.debug('Starting GSIOC monitor')
        async with self.gsioc.client_lock:
            self.logger.debug('Got GSIOC client lock...')
            try:
                while True:
                    data: GSIOCMessage = await self.gsioc.message_queue.get()
                    self.logger.debug(f'GSIOC got data {data}')
                    response = await self.handle_gsioc(data)
                    if data.messagetype == GSIOCCommandType.IMMEDIATE:
                        await self.gsioc.response_queue.put(response)
            except asyncio.CancelledError:
                self.logger.debug("Stopping GSIOC monitor...")
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

    def activate_trigger(self):
        """Activates the trigger
        """

        self.waiting.clear()
        self.trigger.set()

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
            self.activate_trigger()
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
            #self.logger.info(f'Sending dead volume {dead_volume}')
            response = f'{dead_volume:0.0f}'
        else:
            response = await super().handle_gsioc(data)
        
        return response

class ActiveMethod(TypedDict):
    method: MethodBase
    method_data: dict

class MethodRunner:

    def __init__(self):
        
        # Dictionary of known methods
        self.methods: Dict[str, MethodBase] = {}

        # Active method with its initialization data
        self.active_methods: Dict[str, ActiveMethod] = {}

        # Running tasks (used to prevent garbage collection)
        self._running_tasks: Dict[asyncio.Task, Dict[str, str]] = {}

        # Event that is triggered when all methods are completed
        self.event_finished: asyncio.Event = asyncio.Event()

    @property
    def method_schema(self) -> Dict[str, tuple[Field,...]]:
        """Return the schema for the MethodDefinition class of all methods
        """

        return {method_name: fields(m.MethodDefinition) for method_name, m in self.methods.items()}

    def remove_running_task(self, result: asyncio.Future) -> None:
        """Callback when method is complete. Should generally be done last

        Args:
            result (Any): calling method
        """

        self._running_tasks.pop(result)

        # if this was the last method to finish, set event_finished
        if len(self._running_tasks) == 0:
            self.event_finished.set()

    def run_method(self, method: Coroutine, id: str | None = None, name: str = '') -> None:
        """Runs a coroutine method. Designed for complex operations with assembly hardware"""

        # clear finished event because something is now running
        self.event_finished.clear()

        # create unique ID if one is not provided
        if id is None:
            id = str(uuid4())

        # create a task and add to set to avoid garbage collection
        task = asyncio.create_task(method)
        logging.debug(f'Running task {task} from method {method} with id {id}')
        self._running_tasks.update({task: dict(id=id,
                                              method_name=name)})

        # register callbacks upon task completion
        task.add_done_callback(self.remove_running_task)

    def cancel_methods_by_id(self, id: str) -> None:
        """Cancel a running method by searching for its id"""

        for task, iinfo in self._running_tasks.items():
            if id == iinfo['id']:
                logging.debug(f'Cancelling task {iinfo["name"]}')
                task.cancel()

    def cancel_methods_by_name(self, method_name: str):
        for task, iinfo in self._running_tasks.items():
            if method_name == iinfo['method_name']:
                logging.debug(f'Cancelling task {iinfo["method_name"]}')
                task.cancel()

    def clear_method_error(self, method_name: str, retry: bool | None = None):
        """Looks for an active method with method_name and clears its error"""

        active_method = self.active_methods.get(method_name, None)
        if active_method is not None:
            active_method['method'].error.clear(retry)

    def is_ready(self, method_name: str) -> bool:
        """Checks if all devices are unreserved for method

        Args:
            method_name (str): name of method to check

        Returns:
            bool: True if all devices are unreserved
        """

        return self.methods[method_name].is_ready()