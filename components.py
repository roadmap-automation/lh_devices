from typing import List
from connections import Port, Node, connect_nodes

class ComponentBase:
    """Base class for fluid components
    """

    def __init__(self) -> None:

        self.ports: List[Port] = []
        self.generate_nodes()

    def generate_nodes(self) -> None:

        self.nodes = [Node(port) for port in self.ports]

    def get_nodes(self) -> List[Node]:

        return self.nodes
    
class FlowCell(ComponentBase):
    """Representation of a unidirectional flow cell
    """

    def __init__(self, volume=0.0) -> None:
        
        self.inlet_port = Port()
        self.outlet_port = Port()
        self.ports = [self.inlet_port, self.outlet_port]
        
        self.generate_nodes()
        connect_nodes(*self.nodes, dead_volume=volume)