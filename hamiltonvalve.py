from typing import List

from hamilton import HamiltonBase, HamiltonSerial

class Port:
    """
    Representation of a valve port
    """

class ValveBase:
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
                 ports: List[Port] = []) -> None:
        self.n_ports = n_ports
        self.n_positions = n_positions

        if not len(ports):
            self.ports = [Port() for _ in range(n_ports)]
        else:
            if len(ports) != n_ports:
                raise ValueError(f'{len(ports)} ports specified but {n_ports} required')
            self.ports = ports

        self.position = None
        self.move(position)

    def get_connection(self, port: Port) -> Port | None:

        if port not in self.ports:
            raise ValueError(f'Port is not associated with this valve')

        port_idx = self._portmap[self.ports.index(port)]

        return self.ports[port_idx] if port_idx is not None else None

    def update_map(self):
        
        self._portmap = {i: None for i in range(self.n_ports)}

    def move(self, position) -> None:
        if position not in range(1, self.n_positions + 1):
            raise ValueError(f'Requested position {position} is not an integer between 1 and {self.n_positions}')
        
        self.position = position
        self.update_map()

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
        self._portmap = {i: (0 if i == self.position else None) for i in range(1, self.n_ports)}
        self._portmap[0] = self.position

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
        super().update_map()
        self._portmap[self.position - 1] = self.position % self.n_ports
        self._portmap[self.position % self.n_ports] = self.position - 1

class HamiltonValvePositioner(HamiltonBase):
    """Hamilton MVP4 device
    """

    def __init__(self, serial_instance: HamiltonSerial, address: str, valve: ValveBase) -> None:
        super().__init__(serial_instance, address)

        self.valve = valve
        self.initialized = False

    async def initialize(self) -> None:

        # TODO: might be Y!!! Depends on whether left or right is facing front or back.
        response, error = await self.run_until_idle('Z')
        if not error:
            self.initialized = True
        else:
            print(f'Initialization error {error}')

    async def move_valve(self, position: int) -> None:
        """Moves to a particular valve position. See specific valve documentation.

        Args:
            position (int): position to move the valve to
        """

        initial_value = self.valve.position
        
        # this checks for errors
        self.valve.move(position)

        response, error = await self.run_until_idle(f'I{position}')
        if error:
            print(f'Move error {error}')
            self.valve.position = initial_value

