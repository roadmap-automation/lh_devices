from typing import List

from .connections import Port, Node, connect_nodes, disconnect_nodes

class ComponentBase:
    """Base class for fluid components
    """

    def __init__(self, name: str = '') -> None:

        self.name = name
        self.ports: List[Port] = []
        self._generate_nodes()

    def _generate_nodes(self) -> None:

        self.nodes = [Node(port) for port in self.ports]

    def get_nodes(self) -> List[Node]:

        return self.nodes
    
    def __repr__(self):

        if self.name:
            return self.name
        else:
            return object.__repr__(self)
    
class FlowCell(ComponentBase):
    """Representation of a flow cell or sample loop. Inlet and outlet ports are essentially
        arbitrary.
    """

    def __init__(self, volume: float = 0.0, name: str = '') -> None:
        
        self.name = name
        self.inlet_port = Port(name=f'{self.name}.inlet_port')
        self.outlet_port = Port(name=f'{self.name}.outlet_port')
        self.ports = [self.inlet_port, self.outlet_port]
        
        self._generate_nodes()
        self.inlet_node, self.outlet_node = self.get_nodes()
        self._volume = 0.0
        self.set_volume(volume)

    def set_volume(self, volume: float):
        """Sets volume of flow cell

        Args:
            volume (float): volume of flow cell in uL
        """

        self._volume = volume
        disconnect_nodes(*self.nodes)
        connect_nodes(*self.nodes, dead_volume=volume)

    def get_volume(self):
        """Gets volume of flow cell

        Returns:
            float: volume of flow cell in uL
        """

        return self._volume

class InjectionPort(ComponentBase):
    """Representation of an injection port
    """
    
    def __init__(self, name: str = '') -> None:
        super().__init__(name)

        self.injection_port = Port(name=f'{self.name}.injection_port')
        self.ports = [self.injection_port]
        self._generate_nodes()
