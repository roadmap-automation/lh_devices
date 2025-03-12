import asyncio
import json
from typing import Coroutine, List, Dict
from dataclasses import dataclass, asdict
from pathlib import Path

from aiohttp.web_app import Application as Application
from aiohttp import web

from autocontrol.status import Status
from autocontrol.task_struct import TaskData

from ..assemblies import InjectionChannelBase, Mode, Network
from ..autocontrolplugin import AutocontrolPlugin
from ..components import FlowCell, InjectionPort
from ..device import SyringePumpBase, ValvePositionerBase
from ..hamilton.HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump
from ..layout import LayoutPlugin
from ..methods import MethodBase, MethodResult, MethodError
from ..waste import WasteInterfaceBase, WasteItem
from ..webview import sio

from lh_manager.liquid_handler.bedlayout import LHBedLayout, find_composition, Rack, Well
from lh_manager.waste_manager.wastedata import Composition, WATER

class RinseSystemBase(InjectionChannelBase, LayoutPlugin):

    def __init__(self,
                 syringe_pump: SyringePumpBase,
                 source_valve: ValvePositionerBase,
                 selector_valve: ValvePositionerBase,
                 sample_loop: FlowCell,
                 loop_injection_port: InjectionPort | None = None,
                 direct_injection_port: InjectionPort | None = None,
                 layout_path: Path | None = None,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase(),
                 name: str = 'Rinse System',
                 id: str = None):
        
        self.syringe_pump = syringe_pump
        self.source_valve = source_valve
        self.selector_valve = selector_valve
        self.sample_loop = sample_loop
        self.loop_injection_port = loop_injection_port
        self.direct_injection_port = direct_injection_port
        self.waste_tracker = waste_tracker

        # define attribute for the bed layout. Will be loaded upon initialization
        InjectionChannelBase.__init__(self, [syringe_pump, source_valve, selector_valve], loop_injection_port.nodes[0], name, id)

        self.network = Network(self.devices + [loop_injection_port, direct_injection_port])

        self.modes = {'Standby': Mode({source_valve: 0,
                                       selector_valve: 0,
                                        syringe_pump: 0}),
                      'Aspirate': Mode({source_valve: 4,
                                        syringe_pump: 3},
                                        final_node=self.source_valve.valve.nodes[4]),
                      'AspirateAirGap': Mode({source_valve: 1,
                                              syringe_pump: 3}),
                      'PumpAspirate': Mode({syringe_pump: 1}),
                      'PumpPrimeLoop': Mode({source_valve: 4,
                                             selector_valve: 8,
                                             syringe_pump: 3}),
                      'PumpDirectInject': Mode({source_valve: 3,
                                          syringe_pump: 3}),
                      'PumpLoopInject': Mode({source_valve: 3,
                                        syringe_pump: 3}),

                      }
        
        LayoutPlugin.__init__(self, self.id, self.name)
        self.layout_path = layout_path

        # attempt to load the layout from log file
        self.load_layout()

        if self.layout is None:
            self.logger.info(f'Layout path {self.layout_path} does not exist, creating empty rinse system layout...')
            rinse_rack = Rack(columns=3, rows=2, max_volume=2000, style='staggered', wells=[], height=300, width=600, x_translate=300, y_translate=0, shape='circle')
            water_rack = Rack(columns=1, rows=1, max_volume=2000, style='grid', wells=[], height=300, width=300, x_translate=0, y_translate=0, shape='rect')
            self.layout = LHBedLayout(racks={'Water': water_rack,
                                                'Rinse': rinse_rack})
            self.layout.add_well_to_rack('Water', Well(composition=WATER, volume=0, rack_id='', well_number=1))
            self.save_layout()            

    async def release(self):
        """Releases reservations on all devices
        """
        for dev in self.devices:
            dev.reserved = False
            await dev.trigger_update()

    def get_well(self, composition: Composition) -> Well:
        """Searches through existing wells and finds the first one with the appropriate composition

        Args:
            composition (Composition): composition to find

        Returns:
            int: well index
        """

        wells = find_composition(composition, self.layout.get_all_wells())
        return wells[0]
    
    def _aspirate_dead_volume(self) -> float:
        """Returns the volume that is left in the lines after aspirating a plug from a solvent bottle

        Returns:
            float: dead volume in ul

        """

        return 0
    
    async def aspirate_air_gap(self, air_gap_volume: float, speed: float | None = None):
        """Aspirates an air gap into the sample loop

        Args:
            air_gap_volume (float): Air gap volume (ul) to aspirate
            speed (float, optional): Speed (ul / s). Defaults to max aspirate flow rate
        """
        
        await self.change_mode('AspirateAirGap')
        if speed is None:
            speed = 3.0 * 1000 / 60 # 3 mL/min; not very fast but only takes a few seconds
        await self.syringe_pump.run_syringe_until_idle(self.syringe_pump.aspirate(air_gap_volume, speed))

    async def aspirate_solvent(self, index: int, volume: float, speed: float | None = None, use_dead_volume: bool = True) -> float:
        """Aspirates solvent into the sample loop

        Args:
            index (int): index of solvent to aspirate
            volume (float): Volume to aspirate (ul).
            speed (float, optional): Speed (ul / s). Defaults to max aspirate flow rate
            use_dead_volume (bool, optional): If true, account for dead volume between source and selector valves

        Returns:
            float: actual volume aspirated (ul)
        """

        await self.change_mode('Aspirate')
        await self.selector_valve.move_valve(index)
        await self.selector_valve.trigger_update()

        # calculate dead volume. Note that if this fails, it means the mode is not correct for aspiration anyway.
        dead_volume = 0.0
        if use_dead_volume:
            dead_volume = self._aspirate_dead_volume()
            self.logger.info(f'{self.name}: aspiration dead volume is {dead_volume}')

        if speed is None:
            speed = self.syringe_pump.max_aspirate_flow_rate

        await self.syringe_pump.run_syringe_until_idle(self.syringe_pump.aspirate(volume + dead_volume, speed))

        return volume + dead_volume

    async def aspirate_plug(self, target_well: Well, volume: float, air_gap_volume: float = 0.1, speed: float | None = None):
        """Aspirates a plug separated by an air gap. Automatically calculates dead volume from network.

        Args:
            target_well (Well): Well from which to aspirate
            volume (float): Volume to aspirate (ul).
            air_gap_volume (float, optional): Air gap volume (ul) to aspirate
            speed (float, optional): Speed (ul / s). Defaults to max aspirate flow rate
        """

        if speed is None:
            speed = self.syringe_pump.max_aspirate_flow_rate

        await self.aspirate_air_gap(air_gap_volume)
        actual_volume = await self.aspirate_solvent(target_well.well_number, volume, speed)
        target_well.volume -= actual_volume
        self.save_layout()
        await self.aspirate_air_gap(air_gap_volume)

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

    def create_web_app(self, template='roadmap.html'):
        app = super().create_web_app(template)

        app.add_routes(LayoutPlugin._get_routes(self))

        return app

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

class PrimeRinseSource(MethodBase):
    """Primes one of the source lines of the rinse system
    """

    def __init__(self, rinsesystem: RinseSystemBase, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__([rinsesystem.syringe_pump, rinsesystem.source_valve, rinsesystem.selector_valve], waste_tracker=waste_tracker)
        self.rinsesystem = rinsesystem

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):

        name: str = "PrimeRinseSource"
        index: str | int = 1
        volume: str | float = 1 # mL/min
        number_of_primes: str | int = 1

    async def run(self, **kwargs):

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)
        index = int(method.index)
        volume = float(method.volume) * 1000
        number_of_primes = int(method.number_of_primes)

        target_well, _ = self.rinsesystem.layout.get_well_and_rack('Rinse', index)
        target_composition = target_well.composition

        for _ in range(number_of_primes):
            await self.rinsesystem.aspirate_solvent(index, volume, use_dead_volume=False)
            await self.rinsesystem.primeloop(n_prime=1, volume=volume)
        
        await self.waste_tracker.submit(WasteItem(composition=target_composition, volume=volume * float(number_of_primes)))

        self.logger.info(f'{self.rinsesystem.name}.{method.name}: Switching to Standby mode')
        await self.rinsesystem.change_mode('Standby')

        self.release_all()

class InitiateRinse(MethodBase):

    def __init__(self, rinsesystem: RinseSystemBase, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()):
        super().__init__(rinsesystem.devices, waste_tracker)
        self.rinsesystem = rinsesystem

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):

        name: str = "InitiateRinse"

    async def run(self, **kwargs):
        """Waits 10 seconds for a companion method to start. Throws error if it hasn't started yet. Then polls every second until rinse system is released.
        """

        self.reserve_all()
        await self.trigger_update()

        try:
            # poll every second to see if devices have been released yet
            while self.rinsesystem.reserved:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.release_all()

class RinseSystem(AutocontrolPlugin, RinseSystemBase):

    def __init__(self,
                 syringe_pump: HamiltonSyringePump,
                 source_valve: HamiltonValvePositioner,
                 selector_valve: HamiltonValvePositioner,
                 rinse_loop: FlowCell,
                 loop_injection_port: InjectionPort | None = None,
                 direct_injection_port: InjectionPort | None = None,
                 layout_path: Path | None = None,
                 database_path: Path | None = None,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase(),
                 name: str = 'Rinse System',
                 id: str = 'rinse_system'):
        RinseSystemBase.__init__(self, syringe_pump, source_valve, selector_valve, rinse_loop, loop_injection_port, direct_injection_port, layout_path, waste_tracker, name, id)
        AutocontrolPlugin.__init__(self, database_path, self.id, self.name)
        self.rinse_loop = rinse_loop

        self.methods.update({'InitiateRinse': InitiateRinse(self, waste_tracker),
                             'PrimeRinseLoop': PrimeRinseLoop(self, waste_tracker),
                             'PrimeRinseSource': PrimeRinseSource(self, waste_tracker)
                            })
        
        if database_path is not None:
            self.method_callbacks.append(self.async_save_to_database)
            
    def _aspirate_dead_volume(self):
        return self.get_dead_volume('Aspirate', self.selector_valve.valve.nodes[0])

    async def _get_status(self, request):
        return web.Response(text=json.dumps(dict(status=Status.BUSY if any(dev.reserved for dev in self.devices) else Status.IDLE,
                                        channel_status=[])),
                            status=200)

    async def get_info(self) -> Dict:
        d = await AutocontrolPlugin.get_info(self)
        d.update(await RinseSystemBase.get_info(self))

        #d.setdefault('controls', {})
        d['controls'] = d['controls'] | {'release': {'type': 'button',
                                                     'text': 'Release'},
                                         'prime_loop': {'type': 'number',
                                                'text': 'Prime loop repeats: '},
                                         'prime_source': {'type': 'number',
                                                          'text': 'Prime source bottle (2 mL): '},
                                         }
        
        return d
    
    async def event_handler(self, command: str, data: Dict) -> None:

        if command == 'prime_loop':
            return self.run_method('PrimeRinseLoop', dict(name='PrimeRinseLoop', number_of_primes=int(data['n_prime'])))
        if command == 'prime_source':
            return self.run_method('PrimeRinseSource', dict(name='PrimeRinseSource', index=int(data['n_prime']), volume=2.0))
            #return await self.primeloop(int(data['n_prime']))
        if command == 'release':
            await self.release()
        else:
            await RinseSystemBase.event_handler(self, command, data)
            await AutocontrolPlugin.event_handler(self, command, data)