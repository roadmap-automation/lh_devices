import asyncio
import json
from typing import Coroutine, List, Dict
from dataclasses import dataclass, asdict
from pathlib import Path

from aiohttp.web_app import Application as Application
from aiohttp import web

from autocontrol.status import Status
from autocontrol.task_struct import TaskData

from ..device import SyringePumpBase, ValvePositionerBase
from ..hamilton.HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump
from ..history import HistoryDB
from ..components import FlowCell, InjectionPort
from ..assemblies import InjectionChannelBase, Mode, Network
from ..connections import Node
from ..methods import MethodBase, MethodResult
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
                 loop_injection_port: InjectionPort | None = None,
                 direct_injection_port: InjectionPort | None = None,
                 layout_path: Path | None = None,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase(),
                 name='Rinse System'):
        
        self.syringe_pump = syringe_pump
        self.source_valve = source_valve
        self.selector_valve = selector_valve
        self.sample_loop = sample_loop
        self.loop_injection_port = loop_injection_port
        self.direct_injection_port = direct_injection_port
        self.waste_tracker = waste_tracker

        # define attribute for the bed layout. Will be loaded upon initialization
        self.layout: LHBedLayout | None = None
        self.layout_path = layout_path

        super().__init__([syringe_pump, source_valve, selector_valve], loop_injection_port.nodes[0], name)

        self.network = Network(self.devices + [loop_injection_port, direct_injection_port])

        self.modes = {'Standby': Mode({source_valve: 0,
                                       selector_valve: 0,
                                        syringe_pump: 0}),
                      'Aspirate': Mode({source_valve: 1,
                                        syringe_pump: 3}),
                      'AspirateAirGap': Mode({source_valve: 2,
                                              syringe_pump: 3}),
                      'PumpAspirate': Mode({syringe_pump: 1}),
                      'PumpPrimeLoop': Mode({source_valve: 1,
                                             selector_valve: 8,
                                             syringe_pump: 3}),
                      'PumpDirectInject': Mode({source_valve: 3,
                                          syringe_pump: 3}),
                      'PumpLoopInject': Mode({source_valve: 3,
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
                rinse_rack = Rack(columns=2, rows=3, max_volume=2000, style='staggered', wells=[])
                water_rack = Rack(columns=1, rows=1, max_volume=2000, style='grid', wells=[])
                self.layout = LHBedLayout(racks={'Water': water_rack,
                                                 'Rinse': rinse_rack})
                self.layout.add_well_to_rack('Water', Well(composition=WATER, volume=0, rack_id='', well_number=1))
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
    
    async def aspirate_air_gap(self, air_gap_volume: float, speed: float | None = None):
        """Aspirates an air gap into the sample loop

        Args:
            air_gap_volume (float): Air gap volume (ul) to aspirate
            speed (float, optional): Speed (ul / s). Defaults to max aspirate flow rate
        """
        
        await self.change_mode('AspirateAirGap')
        if speed is None:
            speed = self.syringe_pump.max_aspirate_flow_rate
        await self.syringe_pump.run_syringe_until_idle(self.syringe_pump.aspirate(air_gap_volume, speed))

    async def aspirate_solvent(self, index: int, volume: float, speed: float | None = None):
        """Aspirates solvent into the sample loop

        Args:
            index (int): index of solvent to aspirate
            volume (float): Volume to aspirate (ul).
            speed (float, optional): Speed (ul / s). Defaults to max aspirate flow rate
        """

        await self.change_mode('Aspirate')
        await self.selector_valve.move_valve(index)
        if speed is None:
            speed = self.syringe_pump.max_aspirate_flow_rate

        await self.syringe_pump.run_syringe_until_idle(self.syringe_pump.aspirate(volume, speed))

    async def aspirate_plug(self, target_well: Well, volume: float, air_gap_volume: float = 0.1, speed: float | None = None):
        """Aspirates a plug separated by an air gap

        Args:
            target_well (Well): Well from which to aspirate
            volume (float): Volume to aspirate (ul).
            air_gap_volume (float, optional): Air gap volume (ul) to aspirate
            speed (float, optional): Speed (ul / s). Defaults to max aspirate flow rate
        """

        if speed is None:
            speed = self.syringe_pump.max_aspirate_flow_rate

        await self.aspirate_air_gap(air_gap_volume)
        await self.aspirate_solvent(target_well.well_number, volume, speed)
        target_well.volume -= volume
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
                 rinse_loop: FlowCell,
                 loop_injection_port: InjectionPort | None = None,
                 direct_injection_port: InjectionPort | None = None,
                 layout_path: Path | None = None,
                 database_path: Path | None = None,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase(),
                 name='Rinse System'):
        super().__init__(syringe_pump, source_valve, selector_valve, rinse_loop, loop_injection_port, direct_injection_port, layout_path, waste_tracker, name)
        self.rinse_loop = rinse_loop

        self.methods.update({'InitiateRinse': InitiateRinse(self, waste_tracker),
                             'PrimeRinseLoop': PrimeRinseLoop(self, waste_tracker),
                            })
        
        if database_path is not None:
            self.method_callbacks.append(self.save_to_database)
            
            self.database_path = database_path

    async def save_to_database(self, result: MethodResult):
        """Saves a method result to the database, only if a task id is associated with it

        Args:
            result (MethodResult): result to save
        """

        if result.id is not None:
            with HistoryDB(self.database_path) as db:
                db.smart_insert(result)

    def read_from_database(self, id: str) -> MethodResult | None:
        """Reads a method result from the database

        Args:
            id (str): id of record

        Returns:
            MethodResult | None: MethodResult object if id exists, otherwise None
        """

        with HistoryDB(self.database_path) as db:
            return db.search_id(id)

    def create_web_app(self, template='roadmap.html') -> Application:
        app = super().create_web_app(template=template)
        routes = web.RouteTableDef()

        @routes.post('/SubmitTask')
        async def handle_task(request: web.Request) -> web.Response:
            # TODO: turn task into a dataclass; parsing will change
            # testing: curl -X POST http://localhost:5004/SubmitTask -d "{\"channel\": 0, \"method_name\": \"InitiateRinse\", \"method_data\": {\"name\": \"InitiateRinse\"}}"
            data = await request.json()
            task = TaskData(**data)
            self.logger.info(f'{self.name} received task {task}')
            method = task.method_data['method_list'][0]
            method_name: str = method['method_name']
            method_data: dict = method['method_data']
            if self.method_runner.is_ready(method_name):
                self.run_method(method_name, method_data, id=str(task.id))
                  
                return web.Response(text='accepted', status=200)
                
            return web.Response(text='busy', status=503)
       
        @routes.get('/GetStatus')
        async def get_status(request: web.Request) -> web.Response:
            
            statuses = [Status.BUSY if dev.reserved else Status.IDLE for dev in self.devices]

            return web.Response(text=json.dumps(dict(status=Status.IDLE,
                                          channel_status=statuses)),
                                status=200)

        @routes.get('/GetTaskData')
        async def get_task(request: web.Request) -> web.Response:
            data = await request.json()
            task = TaskData(**data)
            
            record = self.read_from_database(task.id)
            if record is None:
                return web.Response(text=f'error: id {task.id} does not exist', status=400)

            return web.Response(text=json.dumps(asdict(record)), status=200)
        
        app.add_routes(routes)

        return app
    
    async def get_info(self) -> Dict:
        d = await super().get_info()

        d['controls'] = d['controls'] | {'prime_loop': {'type': 'number',
                                                'text': 'Prime loop repeats: '}}
        
        return d
    
    async def event_handler(self, command: str, data: Dict) -> None:

        if command == 'prime_loop':
            return self.run_method('PrimeRinseLoop', dict(name='PrimeRinseLoop', number_of_primes=int(data['n_prime'])))
            #return await self.primeloop(int(data['n_prime']))
        else:
            return await super().event_handler(command, data)