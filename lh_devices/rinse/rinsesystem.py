import asyncio
import json
from typing import Coroutine, List, Dict
from dataclasses import dataclass
from pathlib import Path

from aiohttp.web_app import Application as Application

from ..assemblies import AssemblyBase
from ..device import ValvePositionerBase, SyringePumpBase
from ..distribution import DistributionBase
from ..hamilton.HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump
from ..components import FlowCell
from ..assemblies import InjectionChannelBase, Network, Mode, AssemblyMode
from ..connections import Node
from ..methods import MethodBase, MethodBasewithTrigger
from ..bubblesensor import BubbleSensorBase
from ..waste import WasteInterfaceBase
from ..webview import sio

from lh_manager.liquid_handler.bedlayout import LHBedLayout, find_composition, Rack, Well
from lh_manager.waste_manager.wastedata import Composition, WATER

class RinseSystemBase(InjectionChannelBase):

    def __init__(self,
                 syringe_pump: SyringePumpBase,
                 source_valve: ValvePositionerBase,
                 selector_valve: ValvePositionerBase,
                 sample_loop: FlowCell,
                 injection_node: Node | None = None,
                 layout_path: Path | None = None,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase(),
                 name='Rinse System'):
        
        self.syringe_pump = syringe_pump
        self.source_valve = source_valve
        self.selector_valve = selector_valve
        self.sample_loop = sample_loop

        # define attribute for the bed layout. Will be loaded upon initialization
        self.layout: LHBedLayout | None = None
        self.layout_path = layout_path

        super().__init__([syringe_pump, source_valve, selector_valve], injection_node, name)

        self.modes = {'Standby': Mode({source_valve: 0,
                                       selector_valve: 0,
                                        syringe_pump: 0}),
                      'LoadLoop': Mode({source_valve: 1,
                                        syringe_pump: 3}),
                      'AspirateAirGap': Mode({source_valve: 2,
                                              syringe_pump: 3}),
                      'PumpAspirate': Mode({syringe_pump: 1}),
                      'PumpPrimeLoop': Mode({source_valve: 1,
                                             selector_valve: 8,
                                             syringe_pump: 3}),
                      'PumpInject': Mode({source_valve: 2,
                                          syringe_pump: 3}),
                      }

    async def initialize(self):
        """Loads layout from JSON, creating a new one if it does not exist, and then continues with initialization
        """

        if self.layout_path is not None:
            if self.layout_path.exists():
                with open(self.layout_path, 'r') as fh:
                    self.layout = LHBedLayout.model_validate_json(fh.read())

            else:
                self.logger.info(f'Layout path {self.layout_path} does not exist, creating empty rinse system layout...')
                rinse_rack = Rack(columns=2, rows=3, max_volume=2000, style='staggered')
                water_rack = Rack(columns=1, rows=1, max_volume=2000, style='grid')
                self.layout = LHBedLayout(racks={'Water': water_rack,
                                                 'Rinse': rinse_rack})
                self.layout.add_well_to_rack('Water', Well(composition=WATER, volume=0, well_number=1))
                self.save_layout()

        return await super().initialize()

    def save_layout(self):
        """Writes layout to JSON"""

        if self.layout_path is not None:
            with open(self.layout_path, 'w') as fh:
                fh.write(self.layout.model_dump_json(indent=2))

    async def trigger_layout_update(self):
        """Emits a layout_update message through socketio"""
        await sio.emit(f'layout_{self.id}')
        self.save_layout()

    def get_well(self, composition: Composition) -> int:
        """Searches through existing wells and finds the first one with the appropriate composition

        Args:
            composition (Composition): composition to find

        Returns:
            int: well index
        """

        wells = find_composition(composition, self.layout.get_all_wells())
        return wells[0]
    
    async def aspirate_air_gap(self, air_gap_volume: float = 0.1, speed: float=2.0):
        """Aspirates an air gap into the sample loop

        Args:
            air_gap_volume (float, optional): Air gap volume (ml) to aspirate. Defaults to 0.1.
            speed (float, optional): Speed (ml/min). Defaults to 2.0.
        """
        
        await self.change_mode('AspirateAirGap')
        await self.syringe_pump.aspirate(air_gap_volume * 1000, speed * 1000 / 60)

    async def aspirate_solvent(self, index: int, volume: float, speed: float=2.0):
        """Aspirates solvent into the sample loop

        Args:
            index (int): index of solvent to aspirate
            volume (float): Volume to aspirate (ml).
            speed (float, optional): Speed (ml/min). Defaults to 2.0.
        """

        await self.change_mode('LoadLoop')
        await self.selector_valve.move_valve(index)
        await self.syringe_pump.aspirate(volume * 1000, speed * 1000 / 60)

    async def primeloop(self,
                        n_prime: int = 1, # number of repeats
                        volume: float | None = None # prime volume. Uses sample loop volume if None.
                         ) -> None:
        """subroutine for priming the loop method. Primes the loop, but does not activate locks. Uses
            max aspiration flow rate for dispensing as well"""

        await self.change_mode('PumpPrimeLoop')

        volume = self.sample_loop.get_volume() if volume is None else volume

        for _ in range(n_prime):
            await self.syringe_pump.smart_dispense(volume, self.syringe_pump.max_aspirate_flow_rate)

class PrimeRinseLoop(MethodBase):
    """Primes the loop of the rinse system
    """

    def __init__(self, rinsesystem: RinseSystemBase, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__([rinsesystem.syringe_pump, rinsesystem.source_valve, rinsesystem.selector_valve], waste_tracker=waste_tracker)
        self.rinsesystem = rinsesystem

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):

        name: str = "PrimeRinseLoop"
        number_of_primes: str | int = 1

    async def run(self, **kwargs):

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)
        number_of_primes = int(method.number_of_primes)

        await self.rinsesystem.primeloop(number_of_primes)
        await self.waste_tracker.submit_water(number_of_primes * self.rinsesystem.sample_loop.get_volume() / 1000)

        self.logger.info(f'{self.rinsesystem.name}.{method.name}: Switching to Standby mode')
        await self.rinsesystem.change_mode('Standby')

        self.release_all()

class InitiateRinse(MethodBase):

    def __init__(self, rinsesystem: RinseSystemBase, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()):
        super().__init__(rinsesystem.devices, waste_tracker)

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):

        name: str = "InitiateRinse"

    async def run(self, **kwargs):

        pass


class RinseSystem(RinseSystemBase):

    def __init__(self,
                 syringe_pump: HamiltonSyringePump,
                 source_valve: HamiltonValvePositioner,
                 selector_valve: HamiltonValvePositioner,
                 sample_loop: FlowCell,
                 injection_node: Node | None = None,
                 layout_path: Path | None = None,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase(),
                 name='Rinse System'):
        super().__init__(syringe_pump, source_valve, selector_valve, sample_loop, injection_node, layout_path, waste_tracker, name)

        self.methods.update({'InitiateRinse': InitiateRinse(self, waste_tracker),
                             'PrimeRinseLoop': PrimeRinseLoop(self, waste_tracker),
                            })