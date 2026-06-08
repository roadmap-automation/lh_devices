import asyncio
import copy
import json
import logging
import traceback

from aiohttp import web
from aiohttp.web_app import Application as Application
from dataclasses import dataclass, field, fields, Field, MISSING
from typing import List, Dict, Any, Callable, TypedDict, Coroutine, Optional
from uuid import uuid4

from .device import DeviceBase, DeviceError
from .logutils import Loggable, MethodLogHandler, MethodLogFormatter
from .notify import notifier
from .webview import WebNodeBase
from .waste import WasteInterfaceBase

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

    def __post_init__(self):
        if self.result is None:
            self.result = {}
        if self.method_data is None:
            self.method_data = {}
        if self.log is None:
            self.log = []

class MethodBase(Loggable):
    """Base class for defining a method for LH serial devices. Contains information about:
        1. dead volume calculations
        2. required configurations
        3. which devices are involved
        4. locks, signals, controls
        5. waste tracking
    """

    def __init__(self, devices: List[DeviceBase] = [], waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        self.devices = devices
        self.waste_tracker = waste_tracker
        self.error = MethodError()
        self.dead_volume_node: str | None = None

        self.metadata = []
        self.active_reservations = set()

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
            self.active_reservations.add(dev)
            dev.reserved = True

    def release(self, dev: DeviceBase):
        """Releases a single device
        """

        if dev in self.active_reservations:
            self.logger.info(f'{self.name}: releasing device {dev.name}')
            dev.reserved = False
            self.active_reservations.remove(dev)
        else:
            self.logger.warning(f'{self.name}: device {dev.name} not found in active reservations')

    def release_all(self) -> None:
        """Releases all remaining devices
        """

        # conversion to list prevents size of self.active_reservations from changing with releases
        for dev in list(self.active_reservations):
            self.release(dev)

    async def trigger_update(self) -> None:
        """Triggers update of all devices
        """
        for dev in self.devices:
            await dev.trigger_update()

    @dataclass
    class MethodDefinition:
        """Subclass containing the method definition and schema"""

        name: str

    async def run(self, **kwargs) -> dict:
        """Runs the method with the appropriate keywords
        """

        pass

    async def on_cancel(self):
        self.logger.info(f'{self.name} canceled, releasing and updating all devices')
        self.error.retry = False
        self.release_all()
        await self.trigger_update()

    async def start(self, **kwargs) -> MethodResult:
        """Starts a method run with error handling
        """

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
            await self.on_cancel()

        except MethodException as e:
            # these are critical errors
            self.error.error = str(e)
            self.error.retry = e.retry
            self.logger.error(f'Critical error in {self.name}: {e}, retry is {e.retry}, waiting for error to be cleared')
            try:
                await self.trigger_update()
                notifier.notify(subject=f'Error in {self.name}', msg=e)
                await self.error.pause_until_clear()
                if self.error.retry:
                    # try again!
                    self.logger.info(f'{self.name} retrying')
                    retry_metadata = copy.copy(self.metadata)
                    newresult = await self.start(**kwargs)
                    self.metadata = retry_metadata + self.metadata
                    result = newresult.result
            except asyncio.CancelledError:
                await self.on_cancel()
        except Exception as e:
            # these are unexpected errors that should be logged and notified about, but not retried            
            import traceback
            error_trace = traceback.format_exc()
            self.logger.error(f'Unhandled Python Exception in {self.name}:\n{error_trace}')
            self.error.error = str(e)
            result = {'error': str(e), 'traceback': error_trace}
        finally:
            self.logger.info(f'{self.name} finished')
            await self.trigger_update()
            
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
            notifier.notify(subject=f'Error in {self.name}', msg=error)
            await self.trigger_update()
            await self.error.pause_until_clear()

class MethodBasewithTrigger(MethodBase):

    def __init__(self, devices = [], waste_tracker = WasteInterfaceBase()):
        super().__init__(devices, waste_tracker)

        # enables triggering
        self.waiting: asyncio.Event = asyncio.Event()
        self.trigger: asyncio.Event = asyncio.Event()

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

class MethodBasewithBrokerTrigger(MethodBasewithTrigger):
    """Base class for IS methods that synchronise with gilson_lh via broker messages.

    Replaces MethodBasewithGSIOC + MethodBaseDeadVolume. The dead_volume queue
    and wait_for_trigger() semantics are preserved; the GSIOC serial layer is
    gone. The DeviceBrokerWorker monitors dead_volume and activates triggers
    when gsioc.trigger.<task_id> arrives on the broker.
    """

    def __init__(self, devices: List[DeviceBase] = [], waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__(devices, waste_tracker=waste_tracker)

        self.dead_volume: asyncio.Queue = asyncio.Queue(1)


class MethodBasewithCompositionRelay(MethodBase):
    """Base class for methods that relay a composition to peer devices in the same MethodGroup.

    Call ``self.emit_composition(composition)`` from within ``run()`` to schedule
    a ``composition.transfer.<task_id>`` broker event.  Peer devices that include
    ``await_composition_transfer: true`` in their task parameters will receive the
    composition before their own method starts.

    The broker plugin publishes the event after ``run()`` returns but before
    ``task.completed``, so peers are guaranteed to receive it while they are still
    waiting on their own hardware tasks.
    """

    def __init__(self, devices: List[DeviceBase] = [], waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__(devices, waste_tracker)
        self._pending_composition_transfer: Optional[dict] = None

    def emit_composition(self, composition) -> None:
        """Signal the broker to emit composition.transfer to MethodGroup peers on completion.

        ``composition`` must have a ``model_dump()`` method (any Pydantic model).
        """
        self._pending_composition_transfer = composition.model_dump()


class MethodBasewithCompositionReceive(MethodBase):
    """Base class for methods that receive a composition from a MethodGroup peer.

    Call ``await self.receive_composition()`` from within ``run()`` at the
    point where the composition is needed.  The call blocks until the broker
    delivers a ``composition.transfer`` event for this task's MethodGroup, then
    returns the resolved composition dict (or None on timeout).

    The broker plugin detects this base class automatically via ``isinstance``
    and wires up the RabbitMQ subscription before the method starts — no flag
    is needed in the task parameters.
    """

    def __init__(self, devices: List[DeviceBase] = [], waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__(devices, waste_tracker)
        self._incoming_composition: asyncio.Queue = asyncio.Queue(maxsize=1)

    async def receive_composition(self) -> Optional[dict]:
        """Block until a composition arrives from a MethodGroup peer.

        Returns the composition dict, or None if the broker transfer timed out.
        Must be called exactly once per method run.
        """
        return await self._incoming_composition.get()


class MethodBaseDeadVolume(MethodBasewithBrokerTrigger):
    """Compatibility shim for code that still uses the old GSIOC-coupled base class.

    The gsioc argument is accepted but ignored — GSIOC coordination is now
    handled by the broker plugin, not the method.  qcmd/distribution.py is the
    only known caller; it should receive the same refactor as lhmethods.py.
    """

    def __init__(self, gsioc=None, devices: List[DeviceBase] = [], waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__(devices=devices, waste_tracker=waste_tracker)

    def connect_gsioc(self) -> None:
        pass

    def disconnect_gsioc(self) -> None:
        pass


class ActiveMethod(TypedDict):
    method: MethodBase
    method_data: dict

class MethodRunner:

    def __init__(self):

        # Dictionary of known methods
        self.methods: Dict[str, MethodBase] = {}

        # Task type reported in the broker schema for each registered method.
        # Values are plain strings ('none', 'measure', 'prepare', 'transfer').
        self.method_types: Dict[str, str] = {}

        # Active method with its initialization data
        self.active_methods: Dict[str, ActiveMethod] = {}

        # Running tasks (used to prevent garbage collection)
        self._running_tasks: Dict[asyncio.Task, Dict[str, str]] = {}

        # Event that is triggered when all methods are completed
        self.event_finished: asyncio.Event = asyncio.Event()

    def register(self, name: str, method: MethodBase, task_type: str = 'none') -> None:
        """Register a method and its broker task type."""
        self.methods[name] = method
        self.method_types[name] = task_type

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
                logging.debug(f'Cancelling task {iinfo["method_name"]}')
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
    
class MethodPlugin(WebNodeBase):

    def __init__(self, id: str = '', name: str = ''):

        self.id = id
        self.name = name
        self.method_runner = MethodRunner()
        self.method_callbacks: List[Coroutine] = []

    @property
    def methods(self) -> Dict[str, MethodBase]:
        return self.method_runner.methods

    def register(self, name: str, method: MethodBase, task_type: str = 'none') -> None:
        """Register a method and its broker task type."""
        self.method_runner.register(name, method, task_type)

    @property
    def active_methods(self) -> Dict[str, ActiveMethod]:
        return self.method_runner.active_methods

    async def process_method(self, method_name: str, method_data: dict, id: str | None = None) -> MethodResult:
        """Chain of run tasks to accomplish. Subclass to change the logic"""
        self.active_methods.update({method_name: ActiveMethod(method=self.methods[method_name],
                                                        method_data=method_data)})
        try:
            result = await self.methods[method_name].start(**method_data)
            result.id = id
            result.source = self.name
        except asyncio.CancelledError:
            logging.debug(f'Task {method_name} with id {id} cancelled')
        finally:
            self.active_methods.pop(method_name)
            await self.trigger_update()

        await asyncio.gather(*[callback(result) for callback in self.method_callbacks])
        
        return result

    def run_method(self, method_name: str, method_data: dict, id: str | None = None) -> None:

        #if not self.methods[method_name].is_ready():
        #    self.logger.error(f'{self.name}: not all devices in {method_name} are available')
        #else:
        self.method_runner.run_method(self.process_method(method_name, method_data, id), id, method_name)

    def _method_schemas_for_display(self) -> Dict:
        """Return {method_name: [{name, type, default}, ...]} for all registered methods,
        excluding the 'name' field and any field with a complex type (Composition, WellLocation)."""
        result = {}
        for method_name, method in self.methods.items():
            field_list = []
            for f in fields(method.MethodDefinition):
                if f.name == 'name':
                    continue
                type_str = f.type if isinstance(f.type, str) else str(f.type)
                if any(t in type_str for t in ('Composition', 'WellLocation')):
                    continue
                if f.default is not MISSING:
                    default = f.default
                elif f.default_factory is not MISSING:
                    continue
                else:
                    default = None
                field_list.append({'name': f.name, 'type': type_str, 'default': default})
            result[method_name] = field_list
        return result

    async def get_info(self) -> Dict:
        d = await super().get_info()
        d.update({'active_methods': {method_name: dict(method_data=active_method['method_data'],
                                                       has_error=(active_method['method'].error.error is not None),
                                                       has_broker_trigger=isinstance(active_method['method'], MethodBasewithBrokerTrigger))
                                      for method_name, active_method in self.active_methods.items()},
                  'method_schemas': self._method_schemas_for_display()
                 }
                )
        return d
    
    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)
        if command == 'clear_error':
            target_method: MethodBase = self.active_methods.get(data['method'], None)['method']
            if target_method is not None:
                target_method.error.clear(retry=data['retry'])
                await self.trigger_update()
        elif command == 'send_trigger':
            target_method: MethodBasewithTrigger = self.active_methods.get(data['method'], None)['method']
            if target_method is not None:
                target_method.activate_trigger()
        elif command == 'cancel_method':
            target_method = self.active_methods.get(data['method'], None)['method']
            if target_method is not None:
                self.method_runner.cancel_methods_by_name(data['method'])
        elif command == 'run_method':
            method_name = data.get('method_name')
            method_data = data.get('method_data', {})
            if method_name in self.methods:
                method_data['name'] = method_name
                self.run_method(method_name, method_data)

    async def _handle_task(self, request: web.Request) -> web.Response:
        """Handles a submitted task"""

        return web.Response(text='not implemented', status=500)

    async def _get_status(self, request: web.Request) -> web.Response:
        """Status request"""

        return web.Response(text='not implemented', status=500)

    async def _get_task(self, request: web.Request) -> web.Response:
        """Handles requests for information about a task. Dummy method round-trips the response through a TaskData serialization process."""

        return web.Response(text='not implemented', status=500)

    def _get_routes(self) -> web.RouteTableDef:

        routes = web.RouteTableDef()

        @routes.post('/SubmitTask')
        async def handle_task(request: web.Request) -> web.Response:
            return await self._handle_task(request)
       
        @routes.get('/GetStatus')
        async def get_status(request: web.Request) -> web.Response:
            return await self._get_status(request)

        @routes.get('/GetTaskData')
        async def get_task(request: web.Request) -> web.Response:
            return await self._get_task(request)            

        return routes

    def create_web_app(self, template='roadmap.html') -> Application:
        app = super().create_web_app(template=template)
        
        app.add_routes(self._get_routes())

        return app
    