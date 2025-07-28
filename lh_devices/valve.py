import logging
import svg
import math
from dataclasses import dataclass
from typing import List

from .connections import Port
from .components import ComponentBase
from .positioner import PositionerState, PositionerBase

@dataclass
class ValveState(PositionerState):
    number_ports: int

class ValveBase(ComponentBase, PositionerBase):
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

        # valve code for hamilton MVP and PSD devices
        self.hamilton_valve_code = None
        
        self.n_ports = n_ports
        self.n_positions = n_positions

        if not len(ports):
            self.ports = [Port(name=f'{self.name}.port_{i}') for i in range(n_ports)]
        else:
            if len(ports) != n_ports:
                logging.error(f'{len(ports)} ports specified but {n_ports} required')
                return
            self.ports = ports

        self._generate_nodes()

        self.position = None
        self.move(position)

    def get_connection(self, port: Port) -> Port | None:

        if port not in self.ports:
            logging.error(f'Port {port} is not associated with valve {self}')
            return

        port_idx = self._portmap[self.ports.index(port)]

        return self.ports[port_idx] if port_idx is not None else None

    def clear_connections(self) -> None:
        """Clears internal connections
        """

        for node in self.nodes:
            for prt in list(node.connections.keys()):
                if prt in self.ports:
                    node.disconnect(prt)

    def update_connections(self) -> None:
        """Updates internal connections
        """

        for k, v in self._portmap.items():
            self.nodes[k].connect(self.nodes[v].base_port)

    def update_map(self) -> None:
        
        self._portmap: dict[int, int] = {}

    def move(self, position: int) -> None:
        """Move valve to position

        Args:
            position (int): desired position
        """
        if self.validate_move(position):
            self.position = position
            self.update_map()
            self.clear_connections()
            self.update_connections()
    
    @property
    def state(self) -> ValveState:
        """Gets the current state
        """

        return ValveState(name=self.name,
                          position=self.position,
                          number_positions=self.n_positions,
                          number_ports=self.n_ports,
                          svg=self._render_valve())


    def get_info(self) -> ValveState:
        """Valve status information

        Returns:
            ValveState: valve status information dataclass
        """

        return self.state

    def _render_valve(self, through_center=True, highlight_zero=True, half_rotate=False):

        pad = 2
        node_radius = 4
        stroke_width = 1
        thick_stroke = 1.5
        big_radius = 20
        outer_radius = big_radius + 2 * node_radius
        centerx, centery = outer_radius + pad, outer_radius + pad
        angle_offset = 0 if not half_rotate else math.pi / self.n_ports
        vertices = [(round(-big_radius * math.sin(i * 2 * math.pi / self.n_ports + angle_offset)) + centerx,
                    round(big_radius * math.cos(i * 2 * math.pi / self.n_ports + angle_offset)) + centery)
                    for i in range(0, self.n_ports)]

        elements = [svg.Circle(cx=centerx, cy=centery, r=outer_radius, stroke='black', stroke_width=thick_stroke, fill='black', fill_opacity=0, stroke_dasharray=4)]
        zero_color = 'darkred' if highlight_zero else 'black'
        v = vertices[0]
        elements.append(svg.Circle(cx=v[0], cy=v[1], r=node_radius, stroke='black', stroke_width=stroke_width, fill=zero_color, fill_opacity=0.5))

        for v in vertices[1:]:
            elements.append(svg.Circle(cx=v[0], cy=v[1], r=node_radius, stroke='black', stroke_width=stroke_width, fill='black', fill_opacity=0.5))

        for start, end in self._portmap.items():
            vs = vertices[start]
            vn = vertices[end]

            pathdata = [svg.MoveTo(x=vs[0], y=vs[1])]
            if through_center:
                pathdata.append(svg.LineTo(x=centerx, y=centery))
            pathdata.append(svg.LineTo(x=vn[0], y=vn[1]))

            elements.append(svg.Path(stroke="black", stroke_width=thick_stroke, fill_opacity=0, d=pathdata))

        canvas = svg.SVG(
            viewBox=svg.ViewBoxSpec(0, 0, (outer_radius + pad) * 2, (outer_radius + pad) * 2),
            elements=elements
        )

        return canvas.as_str()

class SyringeValveBase(ValveBase):

    def __init__(self, n_ports: int, n_positions: int, position: int = 1, ports: List[Port] = [], name: str = '') -> None:
        super().__init__(n_ports, n_positions, position, ports, name)

        self.aspirate_position: int | None = None
        self.dispense_position: int | None = None

class DistributionValve(ValveBase):
    """Distribution valve representation. "Position" corresponds to the port number of the outlet port.

        "n_ports" is the number of outlet ports only, not including the input port. 0 is the input port,
            and ports are numbered consecutively clockwise from the bottom.
        "position" corresponds to the port number of the valve outlet
    """

    def __init__(self, n_ports: int, position: int = 0, ports: List[Port] = [], name=None) -> None:
        super().__init__(n_ports + 1, n_ports, position, ports, name)

        hamilton_valve_codes = {4: 4, 6: 6, 8: 3}
        if n_ports in hamilton_valve_codes.keys():
            self.hamilton_valve_code = hamilton_valve_codes[n_ports]

    def update_map(self):
        """Updates the port map.
        """
        self._portmap = {}
        if self.position != 0:
            self._portmap[0] = self.position
            self._portmap[self.position] = 0

class LoopFlowValve(ValveBase):
    """Loop flow valve representation, with n_ports and 2 positions. Port 0 is the bottom port. 
        and ports are numbered consecutively clockwise. If there is no bottom port, 0 is the 
        first in the clockwise direction.

    For loop flow valves (n_positions = 2):
        position = 1 corresponds to a connection between ports 0 and 1, 2 and 3, ...
            (n_ports - 2) and (n_ports - 1)
        position = 2 corresponds to a connection between ports 1 and 2, ...
            (nports - 1) and 0.

    """

    def __init__(self, n_ports: int, position: int = 1, ports: List[Port] = [], name=None) -> None:
        super().__init__(n_ports, 2, position, ports, name)

        hamilton_valve_codes = {4: 4, 6: 6, 8: 3}
        if n_ports in hamilton_valve_codes.keys():
            self.hamilton_valve_code = hamilton_valve_codes[n_ports]
    
    def update_map(self) -> None:

        if self.position != 0:
            kv = list()
            offset = -(self.position * 2 - 3) # +1 if position 1; -1 if position 2
            for i in range(0, self.n_ports, 2):
                kv.append((i, (i + offset) % self.n_ports))
                kv.append(((i + offset) % self.n_ports, i))
            
            self._portmap = dict(kv)
        else:
            self._portmap = {}

    def _render_valve(self):
        half_rotate = True if self.n_ports == 6 else False
        return super()._render_valve(through_center=False, highlight_zero=False, half_rotate=half_rotate)

class LValve(ValveBase):
    """L Valve (each position connects two adjacent inlets / outlets). Port 0 is down (syringe),
        and ports are numbered clockwise from the syringe port.
    """

    def __init__(self, n_ports: int, position: int = 1, ports: List[Port] = [], name: str = '') -> None:
        super().__init__(n_ports, n_ports, position, ports, name)

        hamilton_valve_codes = {3: 0, 4: 4}
        if n_ports in hamilton_valve_codes.keys():
            self.hamilton_valve_code = hamilton_valve_codes[n_ports]

    def update_map(self):
        """Updates the port map.
        """
        self._portmap = {}
        if self.position != 0:
            self._portmap[self.position - 1] = self.position % self.n_ports
            self._portmap[self.position % self.n_ports] = self.position - 1

    def _render_valve(self):
        return super()._render_valve(True, False)

class YValve(LValve):

    def __init__(self, position: int = 0, ports: List[Port] = [], name=None) -> None:
        super().__init__(3, position, ports, name)
        self.hamilton_valve_code = 0

class SyringeLValve(SyringeValveBase):
    """L Valve (each position connects two adjacent inlets / outlets) that sits atop syringe
        pump. Port 0 is down (syringe), and ports are numbered clockwise from the syringe port.
        Default aspiration and dispense positions are the first and last ports (reservoir inlet
        on the left, outlet on the right)

    """

    def __init__(self, n_ports: int, position: int = 0, ports: List[Port] = [], name=None) -> None:
        super().__init__(n_ports, n_ports, position, ports, name)

        hamilton_valve_codes = {3: 0, 4: 4}
        if n_ports in hamilton_valve_codes.keys():
            self.hamilton_valve_code = hamilton_valve_codes[n_ports]

        self.aspirate_position = 1
        self.dispense_position = n_ports

    def update_map(self):
        """Updates the port map.
        """
        self._portmap = {}
        if self.position != 0:
            self._portmap[self.position - 1] = self.position % self.n_ports
            self._portmap[self.position % self.n_ports] = self.position - 1

class SyringeYValve(SyringeLValve):
    """Syringe Pump Y valve, which is an implementation of a 3-port L valve.
    """

    def __init__(self, position: int = 0, ports: List[Port] = [], name=None) -> None:
        super().__init__(3, position, ports, name)

