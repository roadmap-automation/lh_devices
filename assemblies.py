import asyncio
from typing import List, Tuple

from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump, batch_run
from connections import Port, Node, connect_nodes

class Network:
    """Representation of a node network
    """

    def __init__(self, devices: List[HamiltonBase]) -> None:
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
            ports must be in the network

        Args:
            source_port (Port): port representing the source
            destination_port (Port): port representing the destination

        Returns:
            float: total dead volume (in uL) along fluid path
        """

        dead_volume = 0.0
        current_port = source_port

        # iterate through the network
        while current_port != destination_port:
            current_node = self._port_to_node_map[current_port]
            new_port, dv = current_node.trace_connection(current_port)

            if not new_port:
                print('Warning: chain broken, destination_port not reached')
                break

            dead_volume += sum(dv)
            current_port = new_port

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

    def __init__(self, devices: List[HamiltonBase]) -> None:
        
        self.devices = devices
        self.network = Network(self.devices)
        self.batch_queue = asyncio.Queue()
    
    async def initialize(self) -> None:
        """Initialize network of devices
        """

        await asyncio.gather(device.initialize() for device in self.devices)

    @property
    def idle(self) -> bool:
        return all(dev.idle for dev in self.devices)

class AssemblyTest(AssemblyBase):

    def __init__(self, loop_valve: HamiltonValvePositioner, syringe_pump: HamiltonSyringePump) -> None:
        super().__init__(devices=[loop_valve, syringe_pump])

    async def initialize(self) -> None:
        
        await batch_run([dev.run_in_batch(dev.initialize_device(), self.batch_queue) for dev in self.devices], self.batch_queue)
        await asyncio.gather(*[dev.poll_until_idle() for dev in self.devices])


class RoadmapChannel(AssemblyBase):
    """Assembly of MVP and PSD devices creating one ROADMAP channel
    """

    def __init__(self, loop_valve: HamiltonValvePositioner, cell_valve: HamiltonValvePositioner, syringe_pump: HamiltonSyringePump) -> None:
        connect_nodes(loop_valve.valve.nodes[0], cell_valve.valve.nodes[1], dead_volume=100)
        # make any additional connections here; network initialization will then know about all the connections
        super().__init__([loop_valve, cell_valve, syringe_pump])

# TODO: Think about serialization / deserialization or loading from a config file. Should be
#           straightforward to reconstruct the network, if not the comm information
