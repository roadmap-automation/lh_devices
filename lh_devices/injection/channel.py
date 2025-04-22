from typing import Dict

from aiohttp.web_app import Application as Application
from lh_manager.liquid_handler.bedlayout import LHBedLayout, Composition, Rack, Well

from ..assemblies import InjectionChannelBase, Network, Mode
from ..bubblesensor import BubbleSensorBase
from ..components import FlowCell
from ..connections import Node
from ..device import ValvePositionerBase, SyringePumpBase
from ..hamilton.HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump
from ..layout import LayoutPlugin

class RoadmapChannelBase(InjectionChannelBase, LayoutPlugin):

    def __init__(self, loop_valve: HamiltonValvePositioner,
                       syringe_pump: HamiltonSyringePump,
                       flow_cell: FlowCell,
                       sample_loop: FlowCell,
                       injection_node: Node | None = None,
                       name: str = '') -> None:
        
        # Devices
        self.loop_valve = loop_valve
        self.syringe_pump = syringe_pump
        self.flow_cell = flow_cell
        self.sample_loop = sample_loop
        super().__init__([loop_valve, syringe_pump], injection_node=injection_node, name=name)

        LayoutPlugin.__init__(self, self.id, self.name)
        self.layout = LHBedLayout(racks={'Carrier': Rack(columns=1, rows=1, max_volume=2000, min_volume=300, style='grid', wells=[], height=300, width=300, x_translate=0, y_translate=0, shape='circle', editable=True),
                                        self.sample_loop.name: Rack(columns=1,
                                            rows=1,
                                            max_volume=self.sample_loop.get_volume() / 1000,
                                            min_volume=0.0,
                                            wells=[Well(composition=Composition(), volume=0.0, rack_id=sample_loop.name, well_number=1, id=None)],
                                            style='grid',
                                            height=300,
                                            width=300,
                                            x_translate=300,
                                            y_translate=300,
                                            shape='rect',
                                            editable=False)})

        # Define node connections for dead volume estimations
        self.network = Network(self.devices + [self.flow_cell, self.sample_loop])

        # Measurement modes
        self.modes = {'Standby': Mode({loop_valve: 0,
                                       syringe_pump: 0}),
                     'LoadLoop': Mode({loop_valve: 1,
                                       syringe_pump: 3},
                                       final_node=syringe_pump.valve.nodes[2]),
                    'PumpAspirate': Mode({loop_valve: 0,
                                          syringe_pump: 1}),
                    'PumpPrimeLoop': Mode({loop_valve: 1,
                                           syringe_pump: 4}),
                    'PumpInject': Mode({loop_valve: 2,
                                        syringe_pump: 4}),
                    'LHPrime': Mode({loop_valve: 2,
                                     syringe_pump: 0},
                                     final_node=loop_valve.valve.nodes[3]),
                    'LHInject': Mode({loop_valve: 1,
                                      syringe_pump: 0},
                                      final_node=loop_valve.valve.nodes[3])
                    }

    @property
    def well(self):
        return self.layout.racks[self.sample_loop.name].wells[0]

    async def initialize(self) -> None:
        """Overwrites base initialization to ensure valves and pumps are in appropriate mode for homing syringe"""

        # initialize loop valve
        await self.loop_valve.initialize()

        # move to a position where loop goes to waste
        await self.loop_valve.move_valve(self.modes['PumpPrimeLoop'].valves[self.loop_valve])

        # initialize syringe pump. If plunger not homed, this will push solution into the loop
        await self.syringe_pump.initialize()

        # If syringe pump was already initialized, plunger may not be homed. Force it to home.
        #await self.change_mode('PumpPrimeLoop')
        #await self.syringe_pump.home()

        # change to standby mode
        await self.change_mode('Standby')

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

        self.well.composition = Composition()
        self.well.volume = 0.0            

    async def get_info(self) -> Dict:
        d = await super().get_info()

        d['controls'] = d.get('controls', {}) | {'release': {'type': 'button',
                                                    'text': 'Release'},
                                                 'prime_loop': {'type': 'number',
                                                    'text': 'Prime loop repeats: '},
}
        
        return d
    
    async def event_handler(self, command: str, data: Dict) -> None:

        if command == 'prime_loop':
            return self.run_method('PrimeLoop', dict(name='PrimeLoop', number_of_primes=int(data['n_prime'])))
            #return await self.primeloop(int(data['n_prime']))
        elif command == 'release':
            await self.release()
        else:
            return await super().event_handler(command, data)

class RoadmapChannelBubbleSensor(RoadmapChannelBase):
    """Roadmap channel with populated methods
    """

    def __init__(self, loop_valve: ValvePositionerBase, syringe_pump: SyringePumpBase, flow_cell: FlowCell, sample_loop: FlowCell, inlet_bubble_sensor: BubbleSensorBase, outlet_bubble_sensor: BubbleSensorBase, injection_node: Node | None = None, name: str = '') -> None:
        super().__init__(loop_valve, syringe_pump, flow_cell, sample_loop, injection_node, name)

        self.inlet_bubble_sensor = inlet_bubble_sensor
        self.outlet_bubble_sensor = outlet_bubble_sensor