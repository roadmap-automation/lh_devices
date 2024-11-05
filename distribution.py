from typing import List, Dict

from aiohttp.web_app import Application as Application

from assemblies import Mode
from device import DeviceBase, ValvePositionerBase
from components import InjectionPort
from assemblies import AssemblyBase

class DistributionBase(AssemblyBase):

    def __init__(self, n_positions, devices: List[DeviceBase], injection_port: InjectionPort, name='') -> None:
        super().__init__(devices, name)

        self.injection_port = injection_port
        self.n_positions = n_positions
        self.modes: Dict[str, Mode] = {}

class DistributionSingleValve(DistributionBase):

    def __init__(self, distribution_valve: ValvePositionerBase, injection_port: InjectionPort, name='') -> None:
        super().__init__(distribution_valve.valve.n_positions, [distribution_valve], injection_port, name)

        self.modes = {str(i): Mode({distribution_valve: i}, final_node=distribution_valve.get_nodes()[i]) for i in range(self.n_positions + 1)}