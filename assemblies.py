import json
import asyncio
import logging
from typing import List, Tuple, Dict, Coroutine

from gsioc import GSIOC, GSIOCMessage, GSIOCCommandType
from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump
from connections import Port, Node, connect_nodes
from components import ComponentBase

class Network:
    """Representation of a node network
    """

    def __init__(self, devices: List[HamiltonBase | ComponentBase]) -> None:
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
            #logging.debug((current_node, current_port))
            new_port, dv = current_node.trace_connection(previous_port)

            if len(new_port) == 0:
                logging.warning('Warning: chain broken, destination_port not reached, dead volume incomplete')
                break

            if len(new_port) > 1:
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

class AssemblyBase:
    """Assembly of Hamilton LH devices
    """

    def __init__(self, devices: List[HamiltonBase], name='') -> None:
        
        self.name = name
        self.devices = devices
        self.network = Network(self.devices)
        self.modes = {}
        self.current_mode = None
        self.running_tasks = set()

        # Event that is triggered when all methods are completed
        self.event_finished: asyncio.Event = asyncio.Event()

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
            logging.info(f'{self}: Changing mode to {mode}')
            await self.move_valves(self.modes[mode])
            self.current_mode = mode
        else:
            logging.error(f'Mode {mode} not in modes dictionary of {self}')

    async def move_valves(self, valve_config: Dict[HamiltonValvePositioner, int]) -> None:
        """Batch change valve conditions. Enables predefined valve modes.

        Args:
            valve_config (Dict[HamiltonBase, int]): dict with valve positioner as key and valve
                                                        position as value
        """

        await asyncio.gather(*(dev.run_until_idle(dev.move_valve(pos)) for dev, pos in valve_config.items() if dev in self.devices))

    def get_dead_volume(self) -> float:
        """Gets dead volume of current configuration mode

        Returns:
            float: dead volume in uL
        """

        nodes: List[Node] = self.modes[self.current_mode]['dead_volume_nodes']
        return self.network.get_dead_volume(nodes[0].base_port, nodes[1].base_port)

    def method_complete_callback(self, result: asyncio.Future) -> None:
        """Callback when method is complete

        Args:
            result (Any): calling method
        """

        self.running_tasks.discard(result)

        # if this was the last method to finish, set event_finished
        if len(self.running_tasks) == 0:
            self.event_finished.set()

    def run_method(self, method: Coroutine) -> None:
        """Runs a coroutine method. Designed for complex operations with assembly hardware"""

        # clear finished event because something is now running
        self.event_finished.clear()

        # create a task and add to set to avoid garbage collection
        task = asyncio.create_task(method)
        logging.debug(f'Running task {task} from method {method}')
        self.running_tasks.add(task)

        # register callback upon task completion
        task.add_done_callback(self.method_complete_callback)

    @property
    def idle(self) -> bool:
        return all(dev.idle for dev in self.devices) & (not len(self.running_tasks))
    
class AssemblyBasewithGSIOC(AssemblyBase):
    """Assembly with support for GSIOC commands
    """

    def __init__(self, devices: List[HamiltonBase], name='') -> None:
        super().__init__(devices, name=name)
        self.waiting: asyncio.Event = asyncio.Event()

    async def initialize_gsioc(self, gsioc: GSIOC) -> None:
        """Initialize GSIOC communications. Only use for top-level assemblies."""

        await asyncio.gather(gsioc.listen(), self.monitor_gsioc(gsioc))

    async def monitor_gsioc(self, gsioc: GSIOC) -> None:
        """Monitor GSIOC communications. Note that only one device should be
            listening to a GSIOC device at a time.
        """

        while True:
            data: GSIOCMessage = await gsioc.message_queue.get()

            response = await self.handle_gsioc(data)
            if data.messagetype == GSIOCCommandType.IMMEDIATE:
                await gsioc.response_queue.put(response)

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
        if data.data == 'V':
            response = f'{self.get_dead_volume():0.0f}'
        
        # requested mode change (deprecated)
        elif data.data.startswith('mode: '):
            mode = data.data.split('mode: ', 1)[1]
            self.run_method(self.change_mode(mode))
                
        # received JSON data for running a method
        elif data.data.startswith('{'):

            # parse JSON
            dd: dict = json.loads(data.data)
            if 'method' in dd.keys():
                method_name, method_kwargs = dd['method'], dd['kwargs']
                logging.debug(f'{self.name}: Method {method_name} requested')

                # check that method exists
                if hasattr(self, method_name):
                    logging.info(f'{self.name}: Starting method {method_name} with kwargs {method_kwargs}')
                    method = getattr(self, method_name)
                    self.run_method(method(**method_kwargs))

                else:
                    logging.warning(f'{self.name}: unknown method name {method_name}')
            
            else:
                response = 'error: unknown JSON data'
        
        else:
            response = 'error: unknown command'

        return response

class AssemblyTest(AssemblyBase):

    def __init__(self, loop_valve: HamiltonValvePositioner, syringe_pump: HamiltonSyringePump, name='') -> None:
        super().__init__(devices=[loop_valve, syringe_pump], name=name)

        connect_nodes(syringe_pump.valve.nodes[1], loop_valve.valve.nodes[0], 101)
        connect_nodes(syringe_pump.valve.nodes[2], loop_valve.valve.nodes[1], 102)

        self.modes = {'LoopInject': 
                    {loop_valve: 2,
                     syringe_pump: 2,
                     'dead_volume_nodes': [syringe_pump.valve.nodes[0], loop_valve.valve.nodes[0]]},
                 'LHInject':
                    {loop_valve: 1,
                     syringe_pump: 1,
                     'dead_volume_nodes': [syringe_pump.valve.nodes[2], loop_valve.valve.nodes[1]]}
                 }


class RoadmapChannel(AssemblyBase):
    """Assembly of MVP and PSD devices creating one ROADMAP channel
    """

    def __init__(self, loop_valve: HamiltonValvePositioner, cell_valve: HamiltonValvePositioner, syringe_pump: HamiltonSyringePump, name='') -> None:
        connect_nodes(loop_valve.valve.nodes[0], cell_valve.valve.nodes[1], dead_volume=100)
        # make any additional connections here; network initialization will then know about all the connections
        super().__init__([loop_valve, cell_valve, syringe_pump], name=name)

# TODO: Think about serialization / deserialization or loading from a config file. Should be
#           straightforward to reconstruct the network, if not the comm information

# TODO: Make Modes a dataclass with "movement" and "dead_volume_nodes" properties. Needs to use
#        device names or internal names instead of direct device references if dataclass.

# TODO: Make a method class (?) / decorator (?) 

# TODO: Decide whether to require the entire method string from the LH (passed through eventually)
#        or make a more integrated system