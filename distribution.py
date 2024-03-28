import json
import logging
from typing import Coroutine, List, Dict
from dataclasses import dataclass

from aiohttp.web_app import Application as Application
from aiohttp import web

from assemblies import Mode
from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump
from gsioc import GSIOC, GSIOCMessage
from components import InjectionPort, FlowCell
from assemblies import AssemblyBase, AssemblyBasewithGSIOC, Network, NestedAssemblyBase
from connections import connect_nodes, Node
from methods import MethodBase

class DistributionBase(AssemblyBase):

    def __init__(self, n_positions, devices: List[HamiltonBase], injection_port: InjectionPort, name='') -> None:
        super().__init__(devices, name)

        self.injection_port = injection_port
        self.n_positions = n_positions
        self.modes: Dict[int, Mode] = {}

class DistributionSingleValve(DistributionBase):

    def __init__(self, distribution_valve: HamiltonValvePositioner, injection_port: InjectionPort, name='') -> None:
        super().__init__(distribution_valve.valve.n_positions, [distribution_valve], injection_port, name)

        self.modes = {i: Mode({distribution_valve: i}) for i in range(self.n_positions)}