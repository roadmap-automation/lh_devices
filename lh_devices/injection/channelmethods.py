"""Standalone methods for a ROADMAP channel"""

import asyncio
from dataclasses import dataclass

from ..methods import MethodBase
from ..waste import WasteInterfaceBase

from .channel import RoadmapChannelBase

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
        await self.waste_tracker.submit_carrier(self.channel.layout.carrier_well, volume=number_of_primes * self.channel.syringe_pump.syringe_volume / 1000)

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
        await self.waste_tracker.submit_carrier(self.channel.layout.carrier_well, pump_volume / 1000)
        self.channel.layout.carrier_well.volume -= pump_volume / 1000        

        # Prime loop
        await self.channel.primeloop()
        await self.waste_tracker.submit_carrier(self.channel.layout.carrier_well, self.channel.syringe_pump.syringe_volume / 1000)

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
            await self.waste_tracker.submit_carrier(self.channel.layout.carrier_well, actual_volume0 / 1000)
        else:
            actual_volume0 = 0.0
        actual_volume = await self.channel.syringe_pump.smart_dispense(pump_volume - min_pump_volume, pump_flow_rate, 5)
        await self.waste_tracker.submit_carrier(self.channel.layout.carrier_well, actual_volume / 1000)
        self.logger.info(f'{self.channel.name}.{method.name}: Actually injected {actual_volume + actual_volume0} uL')

        # Switch to prime loop mode and flush
        await self.channel.primeloop()
        await self.waste_tracker.submit_carrier(self.channel.layout.carrier_well, self.channel.syringe_pump.syringe_volume / 1000)
        await self.channel.syringe_pump.run_until_idle(self.channel.syringe_pump.home())

        self.release_all()