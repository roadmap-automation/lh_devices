import asyncio
from typing import Coroutine, List, Dict
from dataclasses import dataclass

from aiohttp.web_app import Application as Application

from ..device import ValvePositionerBase, SyringePumpBase
from ..distribution import DistributionBase
from ..hamilton.HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump
from ..gilson.gsioc import GSIOC
from ..components import FlowCell
from ..assemblies import InjectionChannelBase, Network,Mode, AssemblyMode
from ..connections import Node
from ..methods import MethodBase, MethodBaseDeadVolume
from ..multichannel import MultiChannelAssembly
from ..bubblesensor import BubbleSensorBase
from ..waste import WasteInterfaceBase

class RoadmapChannelBase(InjectionChannelBase):

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
        
        self.methods.update({'PrimeLoop': PrimeLoop(self)})

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

    async def get_info(self) -> Dict:
        d = await super().get_info()

        d['controls'] = d['controls'] | {'prime_loop': {'type': 'number',
                                                'text': 'Prime loop repeats: '}}
        
        return d
    
    async def event_handler(self, command: str, data: Dict) -> None:

        if command == 'prime_loop':
            return self.run_method('PrimeLoop', dict(name='PrimeLoop', number_of_primes=int(data['n_prime'])))
            #return await self.primeloop(int(data['n_prime']))
        else:
            return await super().event_handler(command, data)

class PrimeLoop(MethodBase):
    """Primes the loop of one ROADMAP channel
    """

    def __init__(self, channel: RoadmapChannelBase, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__([channel.syringe_pump, channel.loop_valve], waste_tracker=waste_tracker)
        self.channel = channel

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):

        name: str = "PrimeLoop"
        number_of_primes: str | int = 1

    async def run(self, **kwargs):

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)
        number_of_primes = int(method.number_of_primes)

        await self.channel.primeloop(number_of_primes)
        await self.waste_tracker.submit_water(number_of_primes * self.channel.syringe_pump.syringe_volume / 1000)

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        self.release_all()

class LoadLoop(MethodBaseDeadVolume):
    """Loads the loop of one ROADMAP channel
    """

    def __init__(self, channel: RoadmapChannelBase, distribution_mode: AssemblyMode, gsioc: GSIOC, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__(gsioc, [channel.syringe_pump, channel.loop_valve, *distribution_mode.valves.keys()], waste_tracker=waste_tracker)
        self.channel = channel
        self.dead_volume_mode: str = 'LoadLoop'
        self.distribution_mode = distribution_mode

    @dataclass
    class MethodDefinition(MethodBaseDeadVolume.MethodDefinition):
        
        name: str = "LoadLoop"
        pump_volume: str | float = 0, # uL
        excess_volume: str | float = 0, #uL
        air_gap: str | float = 0, #uL, not used

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        pump_volume = float(method.pump_volume)
        excess_volume = float(method.excess_volume)

        # Connect to GSIOC communications
        self.connect_gsioc()

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        await self.distribution_mode.activate()
        dead_volume = self.channel.get_dead_volume(self.dead_volume_mode)

        # blocks if there's already something in the dead volume queue
        await self.dead_volume.put(dead_volume)
        self.logger.info(f'{self.channel.name}.{method.name}: dead volume set to {dead_volume}')

        # Wait for trigger to switch to LoadLoop mode
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for first trigger')
        await self.wait_for_trigger()
        if self.dead_volume.qsize():
            self.dead_volume.get_nowait()
            self.logger.warning(f'{self.channel.name}.{method.name}: Trigger received but dead volume was not read')
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LoadLoop mode')

        # Move all valves
        await asyncio.gather(self.distribution_mode.activate(), self.channel.change_mode('LoadLoop'))

        # Wait for trigger to switch to PumpAspirate mode
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for second trigger')
        await self.wait_for_trigger()

        # At this point, liquid handler is done, release communications
        self.disconnect_gsioc()
        for valve in self.distribution_mode.valves.keys():
            valve.reserved = False
            await valve.trigger_update()
        #self.release_liquid_handler.set()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to PumpPrimeLoop mode')
        await self.channel.change_mode('PumpPrimeLoop')

        # smart dispense the volume required to move plug quickly through loop
        self.logger.info(f'{self.channel.name}.{method.name}: Moving plug through loop, total injection volume {self.channel.sample_loop.get_volume() - (pump_volume + excess_volume)} uL')
        await self.channel.syringe_pump.smart_dispense(self.channel.sample_loop.get_volume() - (pump_volume + excess_volume), self.channel.syringe_pump.max_dispense_flow_rate)
        await self.waste_tracker.submit_water((self.channel.sample_loop.get_volume() - (pump_volume + excess_volume)) / 1000)

        # switch to standby mode
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        self.release_all()

class LoadLoopBubbleSensor(MethodBaseDeadVolume):
    """Loads the loop of one ROADMAP channel using a bubble sensor at the waste to detect the air gap.
        Bubble sensor must be powered by digital output 2 (index 1) and read from digital input 2
    """

    def __init__(self, channel: RoadmapChannelBase, distribution_mode: AssemblyMode, gsioc: GSIOC, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__(gsioc, [channel.syringe_pump, channel.loop_valve, *distribution_mode.valves.keys()], waste_tracker=waste_tracker)
        self.channel = channel
        self.dead_volume_mode: str = 'LoadLoop'
        self.distribution_mode = distribution_mode

    @dataclass
    class MethodDefinition(MethodBaseDeadVolume.MethodDefinition):
        
        name: str = "LoadLoopBubbleSensor"
        pump_volume: str | float = 0, # uL
        excess_volume: str | float = 0 # uL, not used
        air_gap: str | float = 0, #uL

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        pump_volume = float(method.pump_volume)
        air_gap = float(method.air_gap)

        # set minimum pump volume before checking for bubbles
        min_pump_volume = 0.5 * pump_volume if pump_volume > 200 else 0

        # Connect to GSIOC communications
        self.connect_gsioc()

        # Power the bubble sensor
        await self.channel.syringe_pump.set_digital_output(1, True)

        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for initial trigger')
        await self.distribution_mode.activate()
        await self.wait_for_trigger()

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.channel.get_dead_volume(self.dead_volume_mode)

        # blocks if there's already something in the dead volume queue
        await self.dead_volume.put(dead_volume)
        self.logger.info(f'{self.channel.name}.{method.name}: dead volume set to {dead_volume}')

        # Wait for trigger to switch to LoadLoop mode
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for first trigger')
        await self.wait_for_trigger()
        if self.dead_volume.qsize():
            self.dead_volume.get_nowait()
            self.logger.warning(f'{self.channel.name}.{method.name}: Trigger received but dead volume was not read')

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LoadLoop mode')

        # Move all valves
        await self.channel.change_mode('LoadLoop')

        # Wait for trigger to switch to PumpAspirate mode
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for second trigger')
        await self.wait_for_trigger()

        # At this point, liquid handler is done, release communications
        self.disconnect_gsioc()
        for valve in self.distribution_mode.valves.keys():
            valve.reserved = False
            await valve.trigger_update()
        
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to PumpPrimeLoop mode')
        await self.channel.change_mode('PumpPrimeLoop')

        # smart dispense the volume required to move plug quickly through loop, interrupting if sensor 2 goes low (air detected)
        self.logger.info(f'{self.channel.name}.{method.name}: Moving plug through loop until air gap detected, total injection volume {self.channel.sample_loop.get_volume() - (pump_volume)} uL with minimum volume {min_pump_volume} uL')
        if min_pump_volume > 0:
            actual_volume0 = await self.channel.syringe_pump.smart_dispense(min_pump_volume, self.channel.syringe_pump.max_dispense_flow_rate)
            await self.waste_tracker.submit_water(actual_volume0 / 1000)
        else:
            actual_volume0 = 0.0
        actual_volume = await self.channel.syringe_pump.smart_dispense(self.channel.sample_loop.get_volume() - pump_volume - min_pump_volume, self.channel.syringe_pump.max_dispense_flow_rate, 6)
        self.logger.info(f'{self.channel.name}.{method.name}: Actually injected {actual_volume + actual_volume0} uL')
        await self.waste_tracker.submit_water(actual_volume / 1000)

        async def traverse_air_gap(nominal_air_gap: float, flow_rate: float = self.channel.syringe_pump.max_dispense_flow_rate, volume_step: float = 10) -> float:
            
            total_air_gap_volume = 0
            total_air_gap_volume += await self.channel.syringe_pump.smart_dispense(nominal_air_gap, flow_rate)
            while not (await self.channel.syringe_pump.get_digital_input(2)):
                total_air_gap_volume += await self.channel.syringe_pump.smart_dispense(volume_step, flow_rate)
            
            return total_air_gap_volume

        # traverse the air gap until fluid is detected at sensor 2 again
        self.logger.info(f'{self.channel.name}.{method.name}: Traversing air gap...')
        total_air_gap_volume = await traverse_air_gap(air_gap, self.channel.syringe_pump.max_dispense_flow_rate)
        self.logger.info(f'{self.channel.name}.{method.name}: Total air gap volume: {total_air_gap_volume} uL')
        await self.waste_tracker.submit_water(total_air_gap_volume / 1000)

        # switch to standby mode
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        self.release_all()


class InjectLoop(MethodBase):
    """Injects the contents of the loop of one ROADMAP channel
    """

    def __init__(self, channel: RoadmapChannelBase, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__([channel.syringe_pump, channel.loop_valve], waste_tracker=waste_tracker)
        self.channel = channel

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "InjectLoop"
        pump_volume: str | float = 0, # uL
        pump_flow_rate: str | float = 1, # mL/min

    async def run(self, **kwargs):
        """InjectLoop method"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        pump_volume = float(method.pump_volume)
        pump_flow_rate = float(method.pump_flow_rate) * 1000 / 60 # convert to uL / s

        # change to inject mode
        await self.channel.change_mode('PumpInject')
        self.logger.info(f'{self.channel.name}.{method.name}: Injecting {pump_volume} uL at flow rate {pump_flow_rate} uL / s')
        await self.channel.syringe_pump.smart_dispense(pump_volume, pump_flow_rate)
        await self.waste_tracker.submit_water(pump_volume / 1000)

        # Prime loop
        await self.channel.primeloop()
        await self.waste_tracker.submit_water(self.channel.syringe_pump.syringe_volume / 1000)
        await self.channel.syringe_pump.run_until_idle(self.channel.syringe_pump.home())

        self.release_all()

class InjectLoopBubbleSensor(MethodBase):
    """Injects the contents of the loop of one ROADMAP channel, using a bubble sensor at the end of the loop to detect
        the air gap. Bubble sensor must be powered from digital output 1 (index 0) and read from digital input 1.
    """

    def __init__(self, channel: RoadmapChannelBase, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__([channel.syringe_pump, channel.loop_valve], waste_tracker=waste_tracker)
        self.channel = channel

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "InjectLoopBubbleSensor"
        pump_volume: str | float = 0, # uL
        pump_flow_rate: str | float = 1, # mL/min

    async def run(self, **kwargs):
        """InjectLoop method"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        pump_volume = float(method.pump_volume)

        # set minimum pump volume before checking for bubbles
        min_pump_volume = 0.5 * pump_volume if pump_volume > 200 else 0

        pump_flow_rate = float(method.pump_flow_rate) * 1000 / 60 # convert to uL / s

        # Power the bubble sensor
        await self.channel.syringe_pump.set_digital_output(0, True)

        # change to inject mode
        await self.channel.change_mode('PumpInject')
        self.logger.info(f'{self.channel.name}.{method.name}: Injecting {pump_volume} uL at flow rate {pump_flow_rate} uL / s')

        # inject, interrupting if sensor 1 goes low (air detected at end of sample loop)
        if min_pump_volume > 0:
            actual_volume0 = await self.channel.syringe_pump.smart_dispense(min_pump_volume, pump_flow_rate)
            await self.waste_tracker.submit_water(actual_volume0 / 1000)
        else:
            actual_volume0 = 0.0
        actual_volume = await self.channel.syringe_pump.smart_dispense(pump_volume - min_pump_volume, pump_flow_rate, 5)
        await self.waste_tracker.submit_water(actual_volume / 1000)
        self.logger.info(f'{self.channel.name}.{method.name}: Actually injected {actual_volume + actual_volume0} uL')

        # Switch to prime loop mode and flush
        await self.channel.primeloop()
        await self.waste_tracker.submit_water(self.channel.syringe_pump.syringe_volume / 1000)
        await self.channel.syringe_pump.run_until_idle(self.channel.syringe_pump.home())

        self.release_all()

class DirectInjectPrime(MethodBaseDeadVolume):
    """Prime direct inject line
    """

    def __init__(self, channel: RoadmapChannelBase, distribution_mode: AssemblyMode, gsioc: GSIOC, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__(gsioc, [channel.loop_valve, *distribution_mode.valves.keys()], waste_tracker=waste_tracker)
        self.channel = channel
        self.dead_volume_mode: str = 'LHPrime'
        self.distribution_mode = distribution_mode

    @dataclass
    class MethodDefinition(MethodBaseDeadVolume.MethodDefinition):
        
        name: str = "DirectInjectPrime"
        pump_volume: str | float = 0, # uL
        pump_flow_rate: str | float = 1, # mL/min        

    async def run(self, **kwargs):
        """Same as DirectInject but does not switch to injection mode"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        # Connect to GSIOC communications
        self.connect_gsioc()

        # Wait for initial trigger
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for initial trigger')
        await self.distribution_mode.activate()
        await self.wait_for_trigger()

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.channel.get_dead_volume(self.dead_volume_mode)

        # blocks if there's already something in the dead volume queue
        await self.dead_volume.put(dead_volume)
        self.logger.info(f'{self.channel.name}.{method.name}: dead volume set to {dead_volume}')

        # Wait for trigger to switch to LHPrime mode (fast injection of air gap + dead volume + extra volume)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for first trigger')
        await self.wait_for_trigger()
        if self.dead_volume.qsize():
            self.dead_volume.get_nowait()
            self.logger.warning(f'{self.channel.name}.{method.name}: Trigger received but dead volume was not read')

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await asyncio.gather(self.channel.change_mode('LHPrime'), self.distribution_mode.activate())

        # Wait for trigger to switch to standby
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for second trigger')
        await self.wait_for_trigger()

        # switch to standby mode    
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        # At this point, liquid handler is done, release communications
        self.disconnect_gsioc()
        self.release_all()

class DirectInject(MethodBaseDeadVolume):
    """Directly inject from LH to a ROADMAP channel flow cell
    """

    def __init__(self, channel: RoadmapChannelBase, distribution_mode: AssemblyMode, gsioc: GSIOC, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__(gsioc, [channel.loop_valve, *distribution_mode.valves.keys()], waste_tracker=waste_tracker)
        self.channel = channel
        self.dead_volume_mode: str = 'LHPrime'
        self.distribution_mode = distribution_mode

    @dataclass
    class MethodDefinition(MethodBaseDeadVolume.MethodDefinition):
        
        name: str = "DirectInject"
        pump_volume: str | float = 0, # uL
        pump_flow_rate: str | float = 1, # mL/min        

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        # Connect to GSIOC communications
        self.connect_gsioc()

        # Wait for initial trigger
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for initial trigger')
        await self.distribution_mode.activate()
        await self.wait_for_trigger()

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.channel.get_dead_volume(self.dead_volume_mode)

        # blocks if there's already something in the dead volume queue
        await self.dead_volume.put(dead_volume)
        self.logger.info(f'{self.channel.name}.{method.name}: dead volume set to {dead_volume}')

        # Wait for trigger to switch to LHPrime mode (fast injection of air gap + dead volume + extra volume)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for first trigger')
        await self.wait_for_trigger()
        if self.dead_volume.qsize():
            self.dead_volume.get_nowait()
            self.logger.warning(f'{self.channel.name}.{method.name}: Trigger received but dead volume was not read')

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await asyncio.gather(self.channel.change_mode('LHPrime'), self.distribution_mode.activate())

        # Wait for trigger to switch to {method.name} mode (LH performs injection)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for second trigger')
        await self.wait_for_trigger()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHInject mode')
        await self.channel.change_mode('LHInject')

        # Wait for trigger to switch to LHPrime mode (fast injection of extra volume + final air gap)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for third trigger')
        await self.wait_for_trigger()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await self.channel.change_mode('LHPrime')

        # Wait for trigger to switch to Standby mode (this may not be necessary)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for fourth trigger')
        await self.wait_for_trigger()

        # switch to standby mode    
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        # At this point, liquid handler is done, release communications
        self.disconnect_gsioc()
        self.release_all()

class DirectInjectBubbleSensor(MethodBaseDeadVolume):
    """Directly inject from LH to measurement system through distribution valve and injection system, using bubble sensors to direct flow.
    """

    def __init__(self, channel: RoadmapChannelBase, distribution_mode: AssemblyMode, gsioc: GSIOC,
                 inlet_bubble_sensor: BubbleSensorBase, outlet_bubble_sensor: BubbleSensorBase,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__(gsioc, [channel.loop_valve, *distribution_mode.valves.keys()], waste_tracker=waste_tracker)
        self.channel = channel
        self.inlet_bubble_sensor = inlet_bubble_sensor
        self.outlet_bubble_sensor = outlet_bubble_sensor
        self.distribution_mode = distribution_mode
        self.dead_volume_mode: str = 'LHPrime'

    @dataclass
    class MethodDefinition(MethodBaseDeadVolume.MethodDefinition):
        
        name: str = "DirectInject"
        pump_volume: str | float = 0, # uL
        pump_flow_rate: str | float = 1, # mL/min

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)
        pump_volume = float(method.pump_volume)
        pump_flow_rate = float(method.pump_flow_rate) * 1000 / 60 # convert to uL / s

        # set minimum pump volume before checking for bubbles
        min_pump_volume = 0.5 * pump_volume if pump_volume > 200 else 0

        # Connect to GSIOC communications
        self.connect_gsioc()

        # power up bubble sensors
        await self.inlet_bubble_sensor.initialize()
        await self.outlet_bubble_sensor.initialize()

        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for initial trigger')
        await self.distribution_mode.activate()
        await self.wait_for_trigger()

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.channel.get_dead_volume(self.dead_volume_mode)

        # blocks if there's already something in the dead volume queue
        await self.dead_volume.put(dead_volume)
        self.logger.info(f'{self.channel.name}.{method.name}: dead volume set to {dead_volume}')

        # Wait for trigger to switch to LHPrime mode (fast injection of air gap + dead volume + extra volume)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for first trigger')
        await self.wait_for_trigger()
        if self.dead_volume.qsize():
            self.dead_volume.get_nowait()
            self.logger.warning(f'{self.channel.name}.{method.name}: Trigger received but dead volume was not read')

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await asyncio.gather(self.channel.change_mode('LHPrime'), self.distribution_mode.activate())

        # Wait for another trigger, which indicates that the LH is going to start asking after the bubble status
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for trigger to traverse air gap')
        await self.wait_for_trigger()
        self.logger.info(f'{self.channel.name}.{method.name}: Traversing air gap...')
        
        # make sure there's always something there to read
        liquid_in_line = False
        await self.dead_volume.put(int(liquid_in_line))

        # traverse air gap
        while not liquid_in_line:
            liquid_in_line = await self.outlet_bubble_sensor.read()
            self.logger.info(f'{self.channel.name}.{method.name}:     Outlet bubble sensor value: {int(liquid_in_line)}')
            # if end condition reached, remove old queue value and put in current one
            if liquid_in_line:
                if self.dead_volume.qsize():
                    self.dead_volume.get_nowait()
            await self.dead_volume.put(int(liquid_in_line))

        # Wait for trigger to switch to LHInject mode (LH performs injection)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for second trigger')
        await self.wait_for_trigger()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHInject mode')
        await self.channel.change_mode('LHInject')

        # monitor the process for air in the line
        self.logger.info(f'{self.channel.name}.{method.name}: Starting air monitor on inlet bubble sensor with delay {min_pump_volume/pump_flow_rate: 0.2f} s...')
        monitor_task = asyncio.create_task(self.detect_air_gap(delay=min_pump_volume/pump_flow_rate, callback=self.channel.change_mode('LHPrime')))
        
        # submit dead volume to waste (this is the only place it is tracked; total air gap size is not tracked)
        await self.waste_tracker.submit_water(dead_volume)

        # Wait for trigger to switch to LHPrime mode (fast injection of extra volume + final air gap)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for third trigger')
        await self.wait_for_trigger()

        # cancel monitor task if it hasn't already been triggered by air in line
        if not monitor_task.done():
            monitor_task.cancel()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await self.channel.change_mode('LHPrime')

        # Wait for trigger to switch to Standby mode (this may not be necessary)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for fourth trigger')
        await self.wait_for_trigger()

        # switch to standby mode    
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        # At this point, liquid handler is done, release communications
        self.disconnect_gsioc()
        self.release_all()

    async def detect_air_gap(self, callback: Coroutine, poll_interval: float = 0.1, delay: float = 0.0):
        """Helper method to detect air gap
        """

        liquid_in_line = True
        self.logger.info(f'{self.channel.name}.detect_air_gap: Waiting {delay} s')
        await asyncio.sleep(delay)
        try:
            while liquid_in_line:
                _, liquid_in_line = await asyncio.gather(asyncio.sleep(poll_interval), self.inlet_bubble_sensor.read())

            self.logger.info(f'{self.channel.name}.detect_air_gap: Air detected, activating callback')            
            await callback
        except asyncio.CancelledError:
            callback.close()

class RoadmapChannelInit(MethodBase):
    """Initialize a ROADMAP channel
    """

    def __init__(self, channel: RoadmapChannelBase) -> None:
        super().__init__([])
        self.channel = channel

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "RoadmapChannelInit"

    async def run(self, **kwargs) -> None:
        
        method = self.MethodDefinition(**kwargs)
        self.logger.info(f'{self.channel.name} received Init method')

class RoadmapChannelSleep(MethodBase):
    """Roadmap Channel Sleep. Used for simulating other operations
    """

    def __init__(self, channel: RoadmapChannelBase) -> None:
        super().__init__(devices=channel.devices)
        self.channel = channel

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "RoadmapChannelSleep"
        sleep_time: float = 0.0

    async def run(self, **kwargs) -> None:
        
        method = self.MethodDefinition(**kwargs)
        self.logger.info(f'{self.channel.name} sleeping {method.sleep_time} min')
        self.reserve_all()
        await self.channel.trigger_update()
        await asyncio.sleep(method.sleep_time * 60)
        #await self.channel.syringe_pump.move_valve(0)
        #await self.throw_error('test error', critical=True)
        self.release_all()
        await self.channel.trigger_update()
        self.logger.info(f'{self.channel.name} sleep complete')

class RoadmapChannel(RoadmapChannelBase):
    """Roadmap channel with populated methods
    """

    def __init__(self, loop_valve: ValvePositionerBase, syringe_pump: SyringePumpBase, flow_cell: FlowCell, sample_loop: FlowCell, injection_node: Node | None = None, name: str = '') -> None:
        super().__init__(loop_valve, syringe_pump, flow_cell, sample_loop, injection_node, name)

        # add standalone methods
        self.methods.update({'InjectLoop': InjectLoop(self),
                        'RoadmapChannelInit': RoadmapChannelInit(self)})

class RoadmapChannelBubbleSensor(RoadmapChannel):
    """Roadmap channel with populated methods
    """

    def __init__(self, loop_valve: ValvePositionerBase, syringe_pump: SyringePumpBase, flow_cell: FlowCell, sample_loop: FlowCell, inlet_bubble_sensor: BubbleSensorBase, outlet_bubble_sensor: BubbleSensorBase, injection_node: Node | None = None, name: str = '') -> None:
        super().__init__(loop_valve, syringe_pump, flow_cell, sample_loop, injection_node, name)

        self.inlet_bubble_sensor = inlet_bubble_sensor
        self.outlet_bubble_sensor = outlet_bubble_sensor

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
