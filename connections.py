import math
import asyncio
from typing import List, Tuple, Dict

class Port:
    """
    Representation of a valve port
    """

class Connection:
    """Representation of a connection between two Ports

        Contains dead volume information (in uL)
    """

    def __init__(self, dead_volume: float = 0) -> None:
        
        # dead volume in uL
        self.dead_volume = dead_volume

    def set_dead_volume_from_tubing(self, tubing_id: float, tubing_length: float) -> None:
        """Set dead volume in uL by tubing parameters

        Args:
            tubing_id (float): tubing ID in cm
            tubing_length (float): tubing length in cm
        """
        self.dead_volume = math.pi * (tubing_id / 2.0) ** 2 * tubing_length * 1000

class Node:
    """Representation of a network node that wraps a Port object
    """

    def __init__(self, base_port: Port) -> None:
        
        # node is centered on this port
        self.base_port = base_port

        self.connections: dict[Port, Connection] = {}

    def connect(self, new_port: Port, dead_volume: float = 0.0, connection = None) -> Connection:
        """connects node to other ports

        Args:
            new_port (Port): existing port to connect to
            dead_volume (float, optional): dead volume of connection. Defaults to 0.0.
            connection (Connection, optional): if specified, uses this connection. Defaults to None.

        Returns:
            Connection: connection object describing connection (or optional provided connection)
        """

        new_connection = connection if connection else Connection(dead_volume)
        self.connections[new_port] = new_connection

        return new_connection
    
    def trace_connection(self, source_port: Port) -> Tuple[List[Port], List[float]] | Tuple[None, None]:
        """Traces connectivity in a network of ports.

        Args:
            source_port (Port): "upstream" port from which connection is being traced

        Returns:
            Port: "downstream" port from current node
            dead_volume: dead volume of connection
        """

        connected_ports = list(self.connections.keys())
        if source_port in connected_ports:
            connected_ports.pop(connected_ports.index(source_port))
            dead_volumes = [self.connections[p].dead_volume for p in connected_ports]
        else:
            raise ValueError('Specified source port is not in connections')
        
        return connected_ports, dead_volumes
    
    def disconnect(self, port: Port) -> None:
        """Disconnects a port. Does not disconnect the reverse connection.

        Args:
            port (Port): port to disconnect
        """

        self.connections.pop(port)

def connect_nodes(node1: Node, node2: Node, dead_volume: float = 0.0) -> None:
    """Connects two nodes. Order is arbitrary.

    Args:
        node1 (Node): first node to connect
        node2 (Node): second node to connect
        dead_volume (float, optional): dead volume of connection
    """

    connection = node1.connect(node2.base_port, dead_volume=dead_volume)
    node2.connect(node1.base_port, connection=connection)

def disconnect_nodes(node1: Node, node2: Node) -> None:
    """Disconnect two nodes. Order is arbitrary.

    Args:
        node1 (Node): first node to disconnect
        node2 (Node): second node to disconnect
    """

    node1.disconnect(node2.base_port)
    node2.disconnect(node1.base_port)

