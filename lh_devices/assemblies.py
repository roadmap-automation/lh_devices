import json
import asyncio
import logging
from uuid import uuid4
from aiohttp import web
from typing import List, Dict, Coroutine, Literal, TypedDict
from dataclasses import asdict

from .device import DeviceBase, DeviceError, ValvePositionerBase
from .gilson.gsioc import GSIOC, GSIOCMessage, GSIOCCommandType
from .connections import Port, Node, connect_nodes
from .methods import MethodBase, MethodBasewithTrigger, MethodBasewithGSIOC, MethodRunner, ActiveMethod, MethodResult
from .components import ComponentBase
from .webview import WebNodeBase

class Network:
    """Representation of a node network
    """

    def __init__(self, devices: List[DeviceBase | ComponentBase]) -> None:
        self.devices = devices
        self._port_to_node_map: dict[Port, Node] = {}
        self.nodes: List[Node] = []
        self.update()

    def update(self) -> None:
        """Updates the node network
        """
        nodes = [node for device in self.devices for node in device.get_nodes()]
        self._port_to_node_map = {n.base_port: n for n in nodes}
        self.nodes = nodes

    def add_device(self, device: DeviceBase | ComponentBase) -> None:
        """Adds a device to node network and updates it"""
        self.devices.append(device)
        self.update()

    def get_dead_volume(self, source_port: Port, destination_port: Port) -> float:
        """Gets dead volume in the fluid path from a source to destination port. Both
            ports must be in the network. Source port can have only one connection.

        Args:
            source_port (Port): port representing the source
            destination_port (Port): port representing the destination

        Returns:
            float: total dead volume (in uL) along fluid path
        """

        dead_volume = 0.0
        current_node = self._port_to_node_map[source_port]
        previous_port = source_port
        current_port = source_port
        # iterate through the network
        while True:
            # find node associated with previous port
            new_port, dv = current_node.trace_connection(previous_port)

            if len(new_port) == 0:
                logging.warning('Warning: chain broken, destination_port not reached, dead volume incomplete')
                break

            if destination_port in new_port:
                dv = [dv[new_port.index(destination_port)]]
            elif len(new_port) > 1:
                logging.warning('Warning: ambiguous connection chain, dead volume incomplete')
                break
            
            # update dead volume from traced connection
            dead_volume += sum(dv)

            # set previous port to the current port
            previous_port = current_port

            # update current port and current node
            current_port = new_port[0]
            current_node = self._port_to_node_map[current_port]

            # break if we've reached destination port
            if current_port == destination_port:
                break

        return dead_volume

def merge_networks(network1: Network, node1: Node, network2: Network, node2: Node, dead_volume: float = 0.0) -> Network:
    """Merge two networks by connecting network1.node1 and network2.node2

    Args:
        network1 (Network): first network to connect
        node1 (Node): node in first network to connect
        network2 (Network): second network to connect
        node2 (Node): node in second network to connect
        dead_volume (float, optional): dead volume of connection from node1 to node2; default 0.0

    Returns:
        Network: merged network
    """

    connect_nodes(node1, node2, dead_volume=dead_volume)
    unique_devices = set(network1.devices) | set(network2.devices)
    return Network(list(unique_devices))

class Mode:
    """Define assembly configuration. Contains a dictionary of valves of the current
        assembly to move. Also defines final node for dead volume tracing"""
    
    def __init__(self, valves: Dict[ValvePositionerBase, int] = {}, final_node: Node | None = None) -> None:

        self.valves = valves
        self.final_node = final_node

    async def activate(self) -> None:
        """Moves valves to positions defined in configuration dictionary
        """

        await asyncio.gather(*(valve.run_until_idle(valve.move_valve(position)) for valve, position in self.valves.items()))

    def __repr__(self) -> str:

        return '; '.join(f'{valve.name} -> {position}' for valve, position in self.valves.items())

class AssemblyBase(WebNodeBase):
    """Assembly of liquid handling devices
    """

    def __init__(self, devices: List[DeviceBase], name='') -> None:
        
        self.name = name
        self.id = str(uuid4())
        WebNodeBase.__init__(self, self.id, self.name)
        self.devices = devices
        self.network = Network(self.devices)
        self.modes: Dict[str, Mode] = {}
        self.current_mode = None

    async def initialize(self) -> None:
        """Initialize network of devices
        """
        await asyncio.gather(*(device.initialize() for device in self.devices))

    async def change_mode(self, mode: str) -> None:
        """Changes to given assembly mode (valve configuration) defined in self.modes

        Args:
            mode (str): name of mode
        """

        if mode in self.modes:
            self.logger.info(f'{self.name}: Changing mode to {mode}')
            await self.move_valves(self.modes[mode].valves)
            self.current_mode = mode
            for valve in self.modes[mode].valves:
                await valve.trigger_update()
        else:
            self.logger.error(f'Mode {mode} not in modes dictionary {self.modes}')

    async def move_valves(self, valve_config: Dict[ValvePositionerBase, int]) -> None:
        """Batch change valve conditions. Enables predefined valve modes.

        Args:
            valve_config (Dict[ValvePositionerBase, int]): dict with valve positioner as key and valve
                                                        position as value
        """

        await asyncio.gather(*(dev.run_until_idle(dev.move_valve(pos)) for dev, pos in valve_config.items()))

    def get_dead_volume(self, source_node: Node, mode: str | None = None) -> float:
        """Gets dead volume of configuration mode given a source connection

        Args:
            source_node (Node): Node of source connection.
            mode (str | None, optional): mode to interrogate. Defaults to None.

        Returns:
            float: dead volume in uL
        """
        
        # use current mode if mode is not given        
        mode = self.current_mode if mode is None else mode

        if mode in self.modes:

            final_node = self.modes[mode].final_node

            if (source_node is not None) & (final_node is not None):

                valve_config: Dict[ValvePositionerBase, int] = self.modes[mode].valves

                # this isn't optimal because it temporarily changes state
                current_positions = []
                for dev, pos in valve_config.items():
                    if dev in self.devices:
                        current_positions.append(dev.valve.position)
                        dev.valve.move(pos)

                self.network.update()

                dead_volume = self.network.get_dead_volume(source_node.base_port, final_node.base_port)

                for dev, pos in zip(valve_config.keys(), current_positions):
                    if dev in self.devices:
                        dev.valve.move(pos)

                self.network.update()

                return dead_volume

            else:
                self.logger.warning(f'{self.name}: source_node or final_node not defined. Returning 0')
                return 0.0
       
        else:
            self.logger.error(f'{self.name}: mode {mode} does not exist')

    @property
    def idle(self) -> bool:
        """Assembly is idle if all devices in the assembly are idle

        Returns:
            bool: True if all devices are idle
        """
        return all(dev.idle for dev in self.devices) # & (not len(self.running_tasks))
    
    @property
    def reserved(self) -> bool:
        """Assembly is reserved if any of the devices are reserved

        Returns:
            bool: True if any devices are reserved
        """
        return any(dev.reserved for dev in self.devices)

    @property
    def error(self) -> DeviceError | None:
        """Error exists if any device has an error

        Returns:
            DeviceError: first error from any of the devices
        """

        return next((dev.error for dev in self.devices if dev.error.error is not None), DeviceError())

    def create_web_app(self, template='roadmap.html') -> web.Application:
        """Creates a web application for this specific assembly by creating a webpage per device

        Returns:
            web.Application: web application for this device
        """

        app = super().create_web_app(template)

        for device in self.devices:
            app.add_subapp(f'/{device.id}/', device.create_web_app(template))

        return app

    
    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)
        if command == 'change_mode':
            await self.change_mode(data)

    async def get_info(self) -> dict:
        """Gets object state as dictionary

        Returns:
            dict: object state
        """

        d = await super().get_info()

        d.update({  'type': 'assembly',
                    'devices': {device.id: device.name for device in self.devices},
                    'modes': [mode for mode in self.modes],
                    'current_mode': self.current_mode,
                    'assemblies': {},
                    'controls': {},
                    'state': {'idle': self.idle,
                              'reserved': self.reserved,
                              'error': asdict(self.error)}})
        
        return d

class AssemblyMode(Mode):
    """Assembly-level mode, used for changing the modes of sub-assemblies. Combines mode valve configurations with valve configurations.
        Modes override conflicts with existing valves (no attempt at conflict resolution)
    """

    def __init__(self, modes: Dict[AssemblyBase, Mode] = {}, valves: Dict[ValvePositionerBase, int] = {}, final_node: Node | None = None) -> None:
        super().__init__(valves, final_node)

        for mode in modes.values():
            self.valves.update(mode.valves)

class ModeGroup(Mode):
    """Similar to AssemblyMode but simply collects modes in a list (does not associate them with the parent assembly)
    """

    def __init__(self, modes: List[Mode | AssemblyMode] = [], final_node: Node | None = None) -> None:
        
        valves = {}
        for mode in modes:
            valves.update(mode.valves)

        super().__init__(valves, final_node)


class AssemblyBasewithGSIOC(AssemblyBase):
    """Assembly with support for GSIOC commands
    """

    def __init__(self, devices: List[DeviceBase], name='') -> None:
        super().__init__(devices, name=name)

        # enables triggering
        self.waiting: asyncio.Event = asyncio.Event()
        self.trigger: asyncio.Event = asyncio.Event()

    async def initialize_gsioc(self, gsioc: GSIOC) -> None:
        """Initialize GSIOC communications. Only use for top-level assemblies."""

        await asyncio.gather(gsioc.listen(), self.monitor_gsioc(gsioc))

    async def monitor_gsioc(self, gsioc: GSIOC) -> None:
        """Monitor GSIOC communications. Note that only one device should be
            listening to a GSIOC device at a time.
        """

        async with gsioc.client_lock:
            while True:
                data: GSIOCMessage = await gsioc.message_queue.get()

                response = await self.handle_gsioc(data)
                if data.messagetype == GSIOCCommandType.IMMEDIATE:
                    await gsioc.response_queue.put(response)

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
            elif self.idle:
                response = 'idle'
            else:
                response = 'busy'

        # get dead volume of current mode in uL
        elif data.data == 'V':
            response = f'{self.get_dead_volume():0.0f}'

        # set trigger
        elif data.data == 'T':
            self.waiting.clear()
            self.trigger.set()
            response = 'ok'

        # requested mode change (deprecated)
        elif data.data.startswith('mode: '):
            mode = data.data.split('mode: ', 1)[1]
            self.run_method(self.change_mode(mode))
                
        # received JSON data for running a method
        elif data.data.startswith('{'):

            # parse JSON
            try:
                dd: dict = json.loads(data.data)
            except json.decoder.JSONDecodeError as e:
                self.logger.error(f'{self.name}: JSON decoding error on string {data.data}: {e.msg}')
                dd: dict = {}

            if 'method' in dd.keys():
                method_name, method_kwargs = dd['method'], dd['kwargs']
                self.logger.debug(f'{self.name}: Method {method_name} requested')

                # check that method exists
                if hasattr(self, method_name):
                    self.logger.info(f'{self.name}: Starting method {method_name} with kwargs {method_kwargs}')
                    method = getattr(self, method_name)
                    self.run_method(method(**method_kwargs))

                else:
                    self.logger.warning(f'{self.name}: unknown method name {method_name}')
            
            else:
                response = 'error: unknown JSON data'
        
        else:
            response = 'error: unknown command'

        return response

class NestedAssemblyBase(AssemblyBase):
    """Nested assembly class that allows specification of sub-assemblies
    """

    def __init__(self, devices: List[DeviceBase], assemblies: List[AssemblyBase], name='') -> None:
        unique_devices = set([dev for assembly in assemblies for dev in assembly.devices] + devices)
        super().__init__(list(unique_devices), name)

        self.assemblies = assemblies

    def create_web_app(self, template='assembly.html') -> web.Application:
        app = super().create_web_app(template)

        for assembly in self.assemblies:
            app.add_subapp(f'/{assembly.id}/', assembly.create_web_app())

        return app

    async def get_info(self) -> Dict:
        """Updates base class information with 

        Returns:
            Dict: _description_
        """
        d = await super().get_info()
        d.update({'assemblies': {assembly.id: assembly.name for assembly in self.assemblies}})
        return d

class InjectionChannelBase(AssemblyBase):

    def __init__(self, devices: List[DeviceBase],
                       injection_node: Node | None = None,
                       name: str = '') -> None:
        
        # Devices
        self.injection_node = injection_node
        super().__init__(devices, name=name)
        self.method_runner = MethodRunner()
        self.method_callbacks: List[Coroutine] = []

        # Define node connections for dead volume estimations
        self.network = Network(self.devices)

        # Measurement modes

    def get_dead_volume(self, mode: str | None = None, injection_node: Node | None = None) -> float:
        if injection_node is None:
            injection_node = self.injection_node
        return super().get_dead_volume(injection_node, mode)

    @property
    def methods(self) -> Dict[str, MethodBase]:
        return self.method_runner.methods

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

        await asyncio.gather(*[callback(result) for callback in self.method_callbacks])
        await self.trigger_update()
        
        return result

    def run_method(self, method_name: str, method_data: dict, id: str | None = None) -> None:

        #if not self.methods[method_name].is_ready():
        #    self.logger.error(f'{self.name}: not all devices in {method_name} are available')
        #else:
        self.method_runner.run_method(self.process_method(method_name, method_data, id), id, method_name)

    @property
    def error(self) -> DeviceError | None:
        """Error exists if any device or method has an error

        Returns:
            DeviceError: first error from any of the devices or methods
        """

        active_methods = [method['method'] for method in self.active_methods.values()]

        return next((dev.error for dev in self.devices + active_methods if dev.error.error is not None), DeviceError())

    async def get_info(self) -> Dict:
        """Updates base class information with 

        Returns:
            Dict: _description_
        """
        d = await super().get_info()
        d.update({'active_methods': {method_name: dict(method_data=active_method['method_data'],
                                                       has_error=(active_method['method'].error.error is not None),
                                                       has_gsioc=isinstance(active_method['method'], MethodBasewithGSIOC))
                                      for method_name, active_method in self.active_methods.items()}
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
