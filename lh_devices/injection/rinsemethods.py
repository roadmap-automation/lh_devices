"""Methods requiring coordination with the rinse system for a ROADMAP channel"""

import asyncio
import json

from dataclasses import dataclass, field
from typing import Coroutine

from ..assemblies import AssemblyMode
from ..bubblesensor import BubbleSensorBase
from ..distribution import DistributionBase
from ..methods import MethodBase
from ..rinse.rinsesystem import RinseSystem
from ..waste import WasteInterfaceBase, WasteItem, Composition, WATER

from .channel import RoadmapChannelBase

# TODO: check for mL/uL conflicts, waste accounting, check stopping of methods, etc.

class RinseLoadLoop(MethodBase):
    """Loads the loop of one ROADMAP channel
    """

    def __init__(self, channel: RoadmapChannelBase, distribution_system: DistributionBase, distribution_mode: AssemblyMode, rinse_system: RinseSystem, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__([channel.syringe_pump, channel.loop_valve, *distribution_system.devices, *rinse_system.devices], waste_tracker=waste_tracker)
        self.channel = channel
        self.rinse_system = rinse_system
        self.distribution_mode = distribution_mode
        self.distribution_system = distribution_system

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "RinseLoadLoop"
        composition: Composition = field(default_factory=lambda: WATER)
        aspirate_flow_rate: str | float = 1 # mL/min
        flow_rate: str | float = 1 # mL/min
        pump_volume: str | float = 1 # ml
        excess_volume: str | float = 0.1 #mL
        air_gap: str | float = 0.1 #ml
        loop_rinse_volume: str | float = 0.5 # ml

    async def traverse_loop(self, method: MethodDefinition):

        pump_volume = float(method.pump_volume)
        excess_volume = float(method.excess_volume)

        # smart dispense the volume required to move plug quickly through loop
        self.logger.info(f'{self.channel.name}.{method.name}: Moving plug through loop, total injection volume {self.channel.sample_loop.get_volume() - (pump_volume + excess_volume)} uL')
        await self.channel.syringe_pump.smart_dispense(self.channel.sample_loop.get_volume() - (pump_volume + excess_volume), self.channel.syringe_pump.max_dispense_flow_rate)
        await self.waste_tracker.submit_water((self.channel.sample_loop.get_volume() - (pump_volume + excess_volume)) / 1000)

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        composition = Composition.model_validate(method.composition)
        target_well = self.rinse_system.get_well(composition)

        air_gap = float(method.air_gap)
        pump_volume = float(method.pump_volume)
        excess_volume = float(method.excess_volume)
        aspirate_flow_rate = float(method.aspirate_flow_rate)
        flow_rate = float(method.flow_rate)
        loop_rinse_volume = float(method.loop_rinse_volume)

        # set source and channel selector and calculate dead volume
        await self.distribution_system.change_mode('Rinse')
        await self.distribution_mode.activate()
        dead_volume = self.channel.get_dead_volume('LoadLoop', self.rinse_system.loop_injection_node)
        self.logger.info(f'{self.channel.name}.{method.name}: dead volume is {dead_volume}')

        # Wait for trigger to switch to LoadLoop mode
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LoadLoop mode')

        # Move all valves
        await self.channel.change_mode('LoadLoop')

        # aspirate material of interest in rinse system. If water (well_index 0), we are using the contents of the loop and the order is different
        if target_well.rack_id == 'Water':
            await self.rinse_system.aspirate_air_gap(air_gap)
            actual_volume = await self.rinse_system.syringe_pump.smart_dispense(air_gap + pump_volume + excess_volume, flow_rate)
            await self.rinse_system.aspirate_air_gap(air_gap)
            actual_volume += await self.rinse_system.syringe_pump.smart_dispense(dead_volume + air_gap + loop_rinse_volume)
            await self.waste_tracker.submit_water(pump_volume + dead_volume + excess_volume + loop_rinse_volume)

        else:
            # aspirate plug
            await self.rinse_system.aspirate_plug(target_well, (pump_volume + excess_volume) / 1000, air_gap, aspirate_flow_rate)
            await self.waste_tracker.submit(WasteItem(composition=target_well.composition,
                                                      volume=(pump_volume + excess_volume) / 1000))

            # push aspirated material through to loop
            total_volume = 2 * air_gap + pump_volume + excess_volume + dead_volume + loop_rinse_volume
            actual_volume = await self.rinse_system.syringe_pump.smart_dispense(total_volume, flow_rate)
            await self.waste_tracker.submit_water(dead_volume)

        # rinse and distribution systems are done, release relevant devices
        self.rinse_system.change_mode('Standby')
        self.rinse_system.release()
        for valve in self.distribution_mode.valves.keys():
            valve.reserved = False
            await valve.trigger_update()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to PumpPrimeLoop mode')
        await self.channel.change_mode('PumpPrimeLoop')

        # smart dispense the volume required to move plug quickly through loop
        self.traverse_loop(method)

        # switch to standby mode
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        self.release_all()
        await self.trigger_update()

class RinseLoadLoopBubbleSensor(RinseLoadLoop):
    """Loads the loop of one ROADMAP channel
    """

    @dataclass
    class MethodDefinition(RinseLoadLoop.MethodDefinition):
        
        name: str = "RinseLoadLoopBubbleSensor"
        composition: Composition = field(default_factory=lambda: WATER)
        aspirate_flow_rate: str | float = 1 # mL/min
        flow_rate: str | float = 1 # mL/min
        pump_volume: str | float = 1 # ml
        excess_volume: str | float = 0.1 #mL
        air_gap: str | float = 0.1 #ml
        loop_rinse_volume: str | float = 0.5 # ml

    async def traverse_loop(self, method: MethodDefinition):

        air_gap = float(method.air_gap)
        pump_volume = float(method.pump_volume)
        excess_volume = float(method.excess_volume)

        # set minimum pump volume before checking for bubbles
        min_pump_volume = 0.5 * pump_volume if pump_volume > 200 else 0

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

class RinseDirectInjectPrime(MethodBase):
    """Prime direct inject line
    """

    def __init__(self, channel: RoadmapChannelBase, distribution_system: DistributionBase, distribution_mode: AssemblyMode, rinse_system: RinseSystem, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__([channel.syringe_pump, channel.loop_valve, *distribution_system.devices, *rinse_system.devices], waste_tracker=waste_tracker)
        self.channel = channel
        self.rinse_system = rinse_system
        self.distribution_mode = distribution_mode
        self.distribution_system = distribution_system

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "DirectInjectPrime"
        pump_volume: str | float = 1, # uL
        pump_flow_rate: str | float = 1, # mL/min        

    async def run(self, **kwargs):
        """Same as DirectInject but does not switch to injection mode"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)
        pump_volume = float(method.pump_volume)
        pump_flow_rate = float(method.pump_flow_rate)

        # set distribution system
        await self.distribution_system.change_mode('Rinse')
        await self.distribution_mode.activate()

        # get dead volume
        dead_volume = self.channel.get_dead_volume('LHPrime', self.rinse_system.direct_injection_node)

        # blocks if there's already something in the dead volume queue
        self.logger.info(f'{self.channel.name}.{method.name}: dead volume is {dead_volume}')

        # Change to prime mode
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await self.channel.change_mode('LHPrime')

        # prime with water
        actual_volume = await self.rinse_system.syringe_pump.smart_dispense(pump_volume + dead_volume, pump_flow_rate)
        await self.waste_tracker.submit_water(actual_volume / 1000)

        # switch to standby mode    
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.rinse_system.change_mode('Standby')
        await self.channel.change_mode('Standby')

        self.release_all()

class RinseDirectInject(MethodBase):
    """Directly inject from LH to a ROADMAP channel flow cell
    """

    def __init__(self, channel: RoadmapChannelBase, distribution_system: DistributionBase, distribution_mode: AssemblyMode, rinse_system: RinseSystem, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__([channel.syringe_pump, channel.loop_valve, *distribution_system.devices, *rinse_system.devices], waste_tracker=waste_tracker)
        self.channel = channel
        self.rinse_system = rinse_system
        self.distribution_mode = distribution_mode
        self.distribution_system = distribution_system

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "RinseDirectInject"
        composition: Composition = field(default_factory=lambda: WATER)
        aspirate_flow_rate: str | float = 1 # mL/min
        flow_rate: str | float = 1 # mL/min
        inject_flow_rate: str | float = 1 #mL/min
        pump_volume: str | float = 1 # ml
        excess_volume: str | float = 0.1 #mL
        rinse_volume: str | float = 0.5 # mL
        air_gap: str | float = 0.1 #ml 

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        composition = Composition.model_validate(method.composition)
        target_well = self.rinse_system.get_well(composition)

        air_gap = float(method.air_gap)
        pump_volume = float(method.pump_volume)
        excess_volume = float(method.excess_volume)
        rinse_volume = float(method.rinse_volume)
        aspirate_flow_rate = float(method.aspirate_flow_rate)
        inject_flow_rate = float(method.inject_flow_rate)
        flow_rate = float(method.flow_rate)

        # set source and channel selector and calculate dead volume
        await self.distribution_system.change_mode('Rinse')
        await self.distribution_mode.activate()
        dead_volume = self.channel.get_dead_volume('LHPrime', self.rinse_system.direct_injection_node)
        self.logger.info(f'{self.channel.name}.{method.name}: dead volume is {dead_volume}')

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await self.channel.change_mode('LHPrime')

        # aspirate material of interest in rinse system. If water (well_index 0), we are using the contents of the rinse loop and the order is different
        if target_well.rack_id == 'Water':
            # aspirate air gap
            await self.rinse_system.aspirate_air_gap(air_gap)
            # move air gap through the dead volume
            actual_volume = await self.rinse_system.syringe_pump.smart_dispense(dead_volume + air_gap + 0.5 * excess_volume, flow_rate)
            # switch to inject mode
            await self.channel.change_mode('LHInject')
            # inject plug
            actual_volume += await self.rinse_system.syringe_pump.smart_dispense(pump_volume, inject_flow_rate)
            # switch back to prime mode
            await self.channel.change_mode('LHPrime')
            # push excess volume to waste
            actual_volume += await self.rinse_system.syringe_pump.smart_dispense(excess_volume * 0.5 + rinse_volume, flow_rate)
            await self.waste_tracker.submit_water(pump_volume + dead_volume + excess_volume + rinse_volume)

        else:
            # aspirate plug
            await self.rinse_system.aspirate_plug(target_well, (pump_volume + excess_volume) / 1000, air_gap, aspirate_flow_rate)
            await self.waste_tracker.submit(WasteItem(composition=target_well.composition,
                                                      volume=(pump_volume + excess_volume) / 1000))

            # move plug through the dead volume
            await self.rinse_system.syringe_pump.smart_dispense(dead_volume + air_gap + 0.5 * excess_volume, flow_rate)
            # switch to inject mode
            await self.channel.change_mode('LHInject')
            # inject plug
            await self.rinse_system.syringe_pump.smart_dispense(pump_volume, inject_flow_rate)
            # switch back to prime mode
            await self.channel.change_mode('LHPrime')
            # push excess volume to waste
            await self.rinse_system.syringe_pump.smart_dispense(excess_volume * 0.5 + rinse_volume + air_gap, flow_rate)
            await self.waste_tracker.submit_water(dead_volume + rinse_volume)

        # switch to standby mode
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        self.release_all()
        await self.trigger_update()

class RinseDirectInjectBubbleSensor(MethodBase):
    """Directly inject from LH to measurement system through distribution valve and injection system, using bubble sensors to direct flow.
    """

    def __init__(self,
                 channel: RoadmapChannelBase,
                 distribution_system: DistributionBase,
                 distribution_mode: AssemblyMode,
                 rinse_system: RinseSystem,
                 inlet_bubble_sensor: BubbleSensorBase,
                 outlet_bubble_sensor: BubbleSensorBase,
                 waste_tracker: WasteInterfaceBase = WasteInterfaceBase()) -> None:
        super().__init__([channel.syringe_pump, channel.loop_valve, *distribution_system.devices, *rinse_system.devices], waste_tracker=waste_tracker)
        self.channel = channel
        self.rinse_system = rinse_system
        self.distribution_mode = distribution_mode
        self.distribution_system = distribution_system
        self.inlet_bubble_sensor = inlet_bubble_sensor
        self.outlet_bubble_sensor = outlet_bubble_sensor

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "RinseDirectInjectBubbleSensor"
        composition: Composition = field(default_factory=lambda: WATER)
        aspirate_flow_rate: str | float = 1 # mL/min
        flow_rate: str | float = 1 # mL/min
        inject_flow_rate: str | float = 1 #mL/min
        pump_volume: str | float = 1 # ml
        excess_volume: str | float = 0.1 #mL
        rinse_volume: str | float = 0.5 # mL
        air_gap: str | float = 0.1 #ml 

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        composition = Composition.model_validate(method.composition)
        target_well = self.rinse_system.get_well(composition)

        air_gap = float(method.air_gap)
        pump_volume = float(method.pump_volume)
        excess_volume = float(method.excess_volume)
        rinse_volume = float(method.rinse_volume)
        aspirate_flow_rate = float(method.aspirate_flow_rate)
        inject_flow_rate = float(method.inject_flow_rate)
        flow_rate = float(method.flow_rate)

        # set minimum pump volume before checking for bubbles
        min_pump_volume = 0.5 * pump_volume if pump_volume > 0.2 else 0

        # set source and channel selector and calculate dead volume
        await self.distribution_system.change_mode('Rinse')
        await self.distribution_mode.activate()
        dead_volume = self.channel.get_dead_volume('LHPrime', self.rinse_system.direct_injection_node)
        self.logger.info(f'{self.channel.name}.{method.name}: dead volume is {dead_volume}')

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await self.channel.change_mode('LHPrime')

        # aspirate material of interest in rinse system. If water (well_index 0), we are using the contents of the rinse loop and the order is different
        if target_well.rack_id == 'Water':
            # aspirate air gap
            await self.rinse_system.aspirate_air_gap(air_gap)
            
            # move air gap through the dead volume. If air gap detected, stop the syringe pump
            actual_volume = await self.dispense_with_monitor(self.outlet_bubble_sensor, (dead_volume + air_gap + 0.5 * excess_volume) * 1000, flow_rate * 1000 / 60, min_pump_volume=min_pump_volume * 1000)
            actual_air_gap_volume = await self.traverse_air_gap(air_gap, flow_rate)

            # switch to inject mode
            await self.channel.change_mode('LHInject')

            # inject plug. No need to monitor for back air gap because this is water and there isn't one
            actual_volume += await self.rinse_system.syringe_pump.smart_dispense(pump_volume, inject_flow_rate)

            # switch back to prime mode
            await self.channel.change_mode('LHPrime')

            # push excess volume to waste
            actual_volume += await self.rinse_system.syringe_pump.smart_dispense(excess_volume * 0.5 + rinse_volume, flow_rate)
            await self.waste_tracker.submit_water(pump_volume + dead_volume + excess_volume + rinse_volume)

        else:
            # aspirate plug
            await self.rinse_system.aspirate_plug(target_well, (pump_volume + excess_volume) / 1000, air_gap, aspirate_flow_rate)
            await self.waste_tracker.submit(WasteItem(composition=target_well.composition,
                                                      volume=(pump_volume + excess_volume) / 1000))

            # move air gap through the dead volume. If air gap detected, stop the syringe pump
            await self.dispense_with_monitor(self.outlet_bubble_sensor, (dead_volume + air_gap + 0.5 * excess_volume) * 1000, flow_rate * 1000 / 60, min_pump_volume=min_pump_volume * 1000)
            actual_air_gap_volume = await self.traverse_air_gap(air_gap, flow_rate)
            # switch to inject mode
            await self.channel.change_mode('LHInject')
            # inject plug. If back air gap detected, stop the pump
            await self.dispense_with_monitor(self.inlet_bubble_sensor, (pump_volume) * 1000, inject_flow_rate * 1000 / 60, min_pump_volume=min_pump_volume * 1000)
            # switch back to prime mode
            await self.channel.change_mode('LHPrime')
            # push excess volume to waste
            await self.rinse_system.syringe_pump.smart_dispense(excess_volume * 0.5 + rinse_volume + air_gap, flow_rate)
            await self.waste_tracker.submit_water(dead_volume + rinse_volume)
        
        # switch to standby mode
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        self.release_all()
        await self.trigger_update()

    async def dispense_with_monitor(self, bubble_sensor: BubbleSensorBase, volume: float, flow_rate: float, min_pump_volume: float = 0) -> float:

        # move air gap through the dead volume. If air gap detected, stop the syringe pump
        dispense_task = asyncio.create_task(self.rinse_system.syringe_pump.smart_dispense(volume, flow_rate))
        monitor_task = asyncio.create_task(self.detect_air_gap(bubble_sensor=bubble_sensor, delay=min_pump_volume/flow_rate, callback=async_cancel(dispense_task)))

        # blocks until dispense is either done or cancelled
        actual_volume = await dispense_task
        if not monitor_task.done():
            monitor_task.cancel()

        return actual_volume

    async def traverse_air_gap(self, nominal_air_gap: float, flow_rate: float, volume_step: float = 10) -> float:
            
            total_air_gap_volume = 0
            total_air_gap_volume += await self.channel.syringe_pump.smart_dispense(nominal_air_gap, flow_rate)
            while not (await self.outlet_bubble_sensor.read()):
                total_air_gap_volume += await self.channel.syringe_pump.smart_dispense(volume_step, flow_rate)
            
            return total_air_gap_volume

    async def detect_air_gap(self, bubble_sensor: BubbleSensorBase, callback: Coroutine, poll_interval: float = 0.1, delay: float = 0.0):
        """Helper method to detect air gap
        """

        liquid_in_line = True
        self.logger.info(f'{self.channel.name}.detect_air_gap: Waiting {delay} s')
        await asyncio.sleep(delay)
        try:
            while liquid_in_line:
                _, liquid_in_line = await asyncio.gather(asyncio.sleep(poll_interval), bubble_sensor.read())

            self.logger.info(f'{self.channel.name}.detect_air_gap: Air detected, activating callback')            
            await callback
        except asyncio.CancelledError:
            callback.close()

async def async_cancel(task: asyncio.Task) -> bool:
    return task.cancel()