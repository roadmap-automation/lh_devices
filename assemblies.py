import asyncio
from typing import List, Tuple

from HamiltonDevice import HamiltonBase
from connections import Port, Node, Connection

class Network:
    """Representation of a node network
    """

    def __init__(self, devices: List[HamiltonBase]) -> None:
        self.devices = devices
        self._port_to_node_map: dict[Port, Node] = {}
        self.nodes: List[Node] = []
        self.update()

    async def initialize(self) -> None:
        """Initialize network of devices
        """

        await asyncio.gather(device.initialize() for device in self.devices)

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


