import asyncio
from typing import Coroutine, List, Dict
from dataclasses import dataclass

from aiohttp.web_app import Application as Application

from ..assemblies import Network, AssemblyMode, ModeGroup
from ..distribution import DistributionBase, DistributionSingleValveTwoSource
from ..gilson.gsioc import GSIOC
from ..multichannel import MultiChannelAssembly
from ..rinse.rinsesystem import RinseSystem
from ..waste import WasteInterfaceBase

from .channel import RoadmapChannelBubbleSensor
from .channelmethods import InjectLoop, InjectLoopBubbleSensor, PrimeLoop, RoadmapChannelInit, RoadmapChannelSleep
from .lhmethods import LoadLoop, LoadLoopBubbleSensor, DirectInject, DirectInjectBubbleSensor, DirectInjectPrime
from .rinsemethods import RinseLoadLoop, RinseLoadLoopBubbleSensor, RinseDirectInjectPrime, RinseDirectInject, RinseDirectInjectBubbleSensor

class RoadmapChannelAssembly(MultiChannelAssembly):

    def __init__(self,
                 channels: List[RoadmapChannelBubbleSensor],
                 distribution_system: DistributionBase,
                 gsioc: GSIOC,
                 database_path: str | None = None,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase(),
                 name='') -> None:
        
        super().__init__(channels=channels,
                         assemblies=[distribution_system],
                         database_path=database_path,
                         name=name)

        """TODO:
            1. Generalize methods to have specific upstream and downstream connection points (if necessary)
            2. add change_direction capability to ROADMAP channels
        """

        # Build network
        self.injection_port = distribution_system.injection_port
        self.network = Network(self.devices + [self.injection_port])

        self.modes = {'Standby': AssemblyMode(modes={distribution_system: distribution_system.modes['8']})}

        distribution_system.network = self.network
        # update channels with upstream information
        for i, ch in enumerate(channels):
            # update network for accurate dead volume calculation
            ch.network = self.network
            ch.injection_node = self.injection_port.nodes[0]

            # add system-specific methods to the channel
            ch.methods.update({'LoadLoop': LoadLoop(ch, distribution_system.modes[str(1 + 2 * i)], gsioc, waste_tracker=waste_tracker),
                               'LoadLoopBubbleSensor': LoadLoopBubbleSensor(ch, distribution_system.modes[str(1 + 2 * i)], gsioc, waste_tracker=waste_tracker),
                               'InjectLoop': InjectLoop(ch, waste_tracker=waste_tracker),
                               'InjectLoopBubbleSensor': InjectLoopBubbleSensor(ch, waste_tracker=waste_tracker),
                               'DirectInjectPrime': DirectInjectPrime(ch, distribution_system.modes[str(2 + 2 * i)], gsioc, waste_tracker=waste_tracker),
                               'DirectInject': DirectInject(ch, distribution_system.modes[str(2 + 2 * i)], gsioc, waste_tracker=waste_tracker),
                               'DirectInjectBubbleSensor': DirectInjectBubbleSensor(ch, distribution_system.modes[str(2 + 2 * i)], gsioc, ch.inlet_bubble_sensor, ch.outlet_bubble_sensor, waste_tracker=waste_tracker),
                               'RoadmapChannelInit': RoadmapChannelInit(ch),
                               'RoadmapChannelSleep': RoadmapChannelSleep(ch),
                               'PrimeLoop': PrimeLoop(ch, waste_tracker=waste_tracker)
                               })

        self.distribution_system = distribution_system

    async def initialize(self) -> None:
        """Initialize the loop as a unit and the distribution valve separately"""
        await asyncio.gather(*[ch.initialize() for ch in self.channels], self.distribution_system.initialize())
        await self.trigger_update()

class RoadmapChannelAssemblyRinse(MultiChannelAssembly):

    def __init__(self,
                 channels: List[RoadmapChannelBubbleSensor],
                 distribution_system: DistributionSingleValveTwoSource,
                 rinse_system: RinseSystem,
                 gsioc: GSIOC,
                 database_path: str | None = None,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase(),
                 name='') -> None:
        
        super().__init__(channels=channels,
                         assemblies=[distribution_system],
                         database_path=database_path,
                         name=name)

        # Build network
        self.injection_port = distribution_system.injection_port
        self.network = Network(self.devices + [self.injection_port, rinse_system.rinse_loop, rinse_system.loop_injection_port, rinse_system.direct_injection_port] + rinse_system.devices)

        self.modes.update({'Standby': AssemblyMode(modes={rinse_system: rinse_system.modes['Standby'],
                                                     distribution_system: distribution_system.modes['8']})})

        distribution_system.network = self.network
        rinse_system.network = self.network
        # update channels with upstream information
        for i, ch in enumerate(channels):
            # update network for accurate dead volume calculation
            ch.network = self.network
            ch.injection_node = self.injection_port.nodes[0]

            # add system-specific methods to the channel
            lh_loop_mode = ModeGroup([distribution_system.modes[str(1 + 2 * i)], distribution_system.modes['LH']])
            rinse_loop_mode = ModeGroup([distribution_system.modes[str(1 + 2 * i)], distribution_system.modes['Rinse']])
            lh_direct_mode = ModeGroup([distribution_system.modes[str(2 + 2 * i)], distribution_system.modes['LH']])
            rinse_direct_mode = ModeGroup([distribution_system.modes[str(2 + 2 * i)], distribution_system.modes['Rinse']])
            ch.methods.update({'LoadLoop': LoadLoop(ch, lh_loop_mode, gsioc, waste_tracker=waste_tracker),
                               'LoadLoopBubbleSensor': LoadLoopBubbleSensor(ch, lh_loop_mode, gsioc, waste_tracker=waste_tracker),
                               'InjectLoop': InjectLoop(ch, waste_tracker=waste_tracker),
                               'InjectLoopBubbleSensor': InjectLoopBubbleSensor(ch, waste_tracker=waste_tracker),
                               'DirectInjectPrime': DirectInjectPrime(ch, lh_direct_mode, gsioc, waste_tracker=waste_tracker),
                               'DirectInject': DirectInject(ch, lh_direct_mode, gsioc, waste_tracker=waste_tracker),
                               'DirectInjectBubbleSensor': DirectInjectBubbleSensor(ch, lh_direct_mode, gsioc, ch.inlet_bubble_sensor, ch.outlet_bubble_sensor, waste_tracker=waste_tracker),
                               'RoadmapChannelInit': RoadmapChannelInit(ch),
                               'RoadmapChannelSleep': RoadmapChannelSleep(ch),
                               'PrimeLoop': PrimeLoop(ch, waste_tracker=waste_tracker),
                               'RinseLoadLoop': RinseLoadLoop(ch, rinse_loop_mode, rinse_system, waste_tracker=waste_tracker),
                               'RinseLoadLoopBubbleSensor': RinseLoadLoopBubbleSensor(ch, rinse_loop_mode, rinse_system, waste_tracker=waste_tracker),
                               'RinseDirectInjectPrime': RinseDirectInjectPrime(ch, rinse_direct_mode, rinse_system, waste_tracker=waste_tracker),
                               'RinseDirectInject': RinseDirectInject(ch, rinse_direct_mode, rinse_system, waste_tracker=waste_tracker),
                               'RinseDirectInjectBubbleSensor': RinseDirectInjectBubbleSensor(ch, rinse_direct_mode, rinse_system, ch.inlet_bubble_sensor, ch.outlet_bubble_sensor, waste_tracker=waste_tracker),
                               })
            
        self.distribution_system = distribution_system
        self.rinse_system = rinse_system

    async def initialize(self) -> None:
        """Initialize the loop as a unit and the distribution valve separately"""
        await asyncio.gather(*[ch.initialize() for ch in self.channels], self.rinse_system.initialize(), self.distribution_system.initialize())
        await self.trigger_update()
