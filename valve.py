from typing import List

from connections import Port, Node
from components import ComponentBase

class ValveBase(ComponentBase):
    """
    Valve base representation.
    
    Conventions:
    Ports are numbered from 0 to n_ports - 1 starting with the port on the underside
        of the valve and proceeding clockwise around the valve.
    """

    def __init__(self,
                 n_ports: int,
                 n_positions: int,
                 position: int = 1,
                 ports: List[Port] = [],
                 name: str = '') -> None:
        self.name = name
        
        self.n_ports = n_ports
        self.n_positions = n_positions

        if not len(ports):
            self.ports = [Port(name=f'{self.name}.port_{i}') for i in range(n_ports)]
        else:
            if len(ports) != n_ports:
                raise ValueError(f'{len(ports)} ports specified but {n_ports} required')
            self.ports = ports

        self._generate_nodes()

        self.position = None
        self.move(position)

    def get_connection(self, port: Port) -> Port | None:

        if port not in self.ports:
            raise ValueError(f'Port is not associated with this valve')

        port_idx = self._portmap[self.ports.index(port)]

        return self.ports[port_idx] if port_idx is not None else None

    def clear_connections(self) -> None:
        """Clears internal connections
        """

        for node in self.nodes:
            for prt in node.connections.keys():
                if prt in self.ports:
                    node.disconnect(prt)

    def update_connections(self) -> None:
        """Updates internal connections
        """

        for k, v in self._portmap.items():
            self.nodes[k].connect(self.nodes[v].base_port)

    def update_map(self) -> None:
        
        self._portmap: dict[int, int] = {}

    def move(self, position) -> None:
        if position not in range(1, self.n_positions + 1):
            raise ValueError(f'Requested position {position} is not an integer between 1 and {self.n_positions}')
        
        self.position = position
        self.update_map()
        self.clear_connections()
        self.update_connections()

class DistributionValve(ValveBase):
    """Distribution valve representation. "Position" corresponds to the port number of the outlet port.

        "n_ports" is the number of outlet ports only, not including the input port. 0 is the input port,
            and ports are numbered consecutively clockwise from the bottom.
        "position" corresponds to the port number of the valve outlet
    """

    def __init__(self, n_ports: int, position: int = 1, ports: List[Port] = []) -> None:
        super().__init__(n_ports + 1, n_ports, position, ports)

    def update_map(self):
        """Updates the port map.
        """
        self._portmap = {}
        self._portmap[0] = self.position
        self._portmap[self.position] = 0

class LoopFlowValve(ValveBase):
    """Loop flow valve representation, with n_ports and 2 positions. Port 0 is the bottom port, 
        and ports are numbered consecutively clockwise.

    For loop flow valves (n_positions = 2):
        position = 1 corresponds to a connection between ports 0 and 1, 2 and 3, ...
            (n_ports - 2) and (n_ports - 1)
        position = 2 corresponds to a connection between ports 1 and 2, ...
            (nports - 1) and 0.

    """

    def __init__(self, n_ports: int, position: int = 1, ports: List[Port] = []) -> None:
        super().__init__(n_ports, 2, position, ports)
    
    def update_map(self) -> None:
        
        kv = list()
        offset = -(self.position * 2 - 3) # +1 if position 1; -1 if position 2
        for i in range(0, self.n_ports, 2):
            kv.append((i, (i + offset) % self.n_ports))
            kv.append(((i + offset) % self.n_ports, i))
        
        self._portmap = dict(kv)

class TValve(ValveBase):
    """T Valve (each position connects two adjacent inlets / outlets)

    """

    def __init__(self, n_ports: int, position: int = 1, ports: List[Port] = []) -> None:
        super().__init__(n_ports, n_ports, position, ports)

    def update_map(self):
        """Updates the port map.
        """
        self._portmap = {}
        self._portmap[self.position - 1] = self.position % self.n_ports
        self._portmap[self.position % self.n_ports] = self.position - 1

class SyringeYValve(TValve):
    """Y valve to sit atop syringe pump. Port 0 is down, and ports are number clockwise.
        Implementation is that of a 3-port T valve.
    """

    def __init__(self, position: int = 1, ports: List[Port] = []) -> None:
        super().__init__(3, position, ports)
