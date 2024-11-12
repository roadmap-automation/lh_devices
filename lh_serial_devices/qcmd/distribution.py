import asyncio
import logging
import json

from dataclasses import dataclass
from typing import List
from aiohttp import web
from aiohttp.web import Application

from .recorder import QCMDRecorder

from ..device import ValvePositionerBase
from ..hamilton.HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump, HamiltonSerial, SyringePumpwithBubbleSensor, SMDSensoronHamiltonDevice
from ..valve import LoopFlowValve, SyringeLValve, DistributionValve
from ..components import InjectionPort, FlowCell, Node
from ..distribution import DistributionSingleValve
from ..gsioc import GSIOC, GSIOCMessage
from ..assemblies import AssemblyBasewithGSIOC, AssemblyBase, InjectionChannelBase, Network, connect_nodes, Mode, NestedAssemblyBase, AssemblyMode
from ..bubblesensor import BubbleSensorBase
from ..webview import run_socket_app
from ..methods import MethodBaseDeadVolume

class QCMDDistributionChannel(InjectionChannelBase):
    """Channel for a simple QCMD array"""

    def __init__(self, flow_cell: FlowCell, injection_node: Node | None = None, name: str = '') -> None:
        self.flow_cell = flow_cell
        super().__init__([], injection_node, name)

        self.network = Network([self.flow_cell])

class QCMDDirectInject(MethodBaseDeadVolume):
    """Directly inject from LH to a QCMD instrument through a distribution valve
    """

    def __init__(self, channel: QCMDDistributionChannel, gsioc: GSIOC) -> None:
        super().__init__(gsioc, channel.devices)
        self.channel = channel
        self.dead_volume_mode: str = 'Waste'

    @dataclass
    class MethodDefinition(MethodBaseDeadVolume.MethodDefinition):
        
        name: str = "DirectInject"

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        # Connect to GSIOC communications and wait for trigger
        self.logger.info(f'{self.channel.name}.{method.name}: Connecting to GSIOC')
        self.connect_gsioc()

        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for initial trigger')
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
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Waste mode')
        await self.channel.change_mode('Waste')

        # Wait for trigger to switch to {method.name} mode (LH performs injection)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for second trigger')
        await self.wait_for_trigger()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Inject mode')
        await self.channel.change_mode('Inject')

        # Wait for trigger to switch to LHPrime mode (fast injection of extra volume + final air gap)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for third trigger')
        await self.wait_for_trigger()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Waste mode')
        await self.channel.change_mode('Waste')

        # Wait for trigger to switch to Standby mode (this may not be necessary)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for fourth trigger')
        await self.wait_for_trigger()

        # switch to standby mode
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        # At this point, liquid handler is done, release communications
        self.disconnect_gsioc()
        self.release_all()

class QCMDDirectInjectBubbleSensor(MethodBaseDeadVolume):
    """Directly inject from LH to a QCMD instrument through a distribution valve
    """

    def __init__(self, channel: QCMDDistributionChannel, gsioc: GSIOC, inlet_bubble_sensor: BubbleSensorBase, outlet_bubble_sensor: BubbleSensorBase) -> None:
        super().__init__(gsioc, channel.devices)
        self.channel = channel
        self.inlet_bubble_sensor = inlet_bubble_sensor
        self.outlet_bubble_sensor = outlet_bubble_sensor
        self.dead_volume_mode: str = 'Waste'

    @dataclass
    class MethodDefinition(MethodBaseDeadVolume.MethodDefinition):
        
        name: str = "DirectInject"

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        self.reserve_all()

        method = self.MethodDefinition(**kwargs)

        # Connect to GSIOC communications and wait for trigger
        self.logger.info(f'{self.channel.name}.{method.name}: Connecting to GSIOC')
        self.connect_gsioc()

        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for initial trigger')
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
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Waste mode')
        await self.channel.change_mode('Waste')

        # Wait for another trigger, which indicates that the LH is going to start asking after the bubble status
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for trigger to traverse air gap')
        await self.wait_for_trigger()
        await self.outlet_bubble_sensor.initialize()
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

        # Wait for trigger to switch to {method.name} mode (LH performs injection)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for second trigger')
        await self.wait_for_trigger()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Inject mode')
        await self.channel.change_mode('Inject')

        # Wait for trigger to switch to LHPrime mode (fast injection of extra volume + final air gap)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for third trigger')
        await self.wait_for_trigger()

        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Waste mode')
        await self.channel.change_mode('Waste')

        # Wait for trigger to switch to Standby mode (this may not be necessary)
        self.logger.info(f'{self.channel.name}.{method.name}: Waiting for fourth trigger')
        await self.wait_for_trigger()

        # switch to standby mode
        self.logger.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        # At this point, liquid handler is done, release communications
        self.disconnect_gsioc()
        self.release_all()

class QCMDDistributionAssembly(NestedAssemblyBase, AssemblyBase):
    """Assembly of QCMD channels with a distribution system, no injection system"""

    def __init__(self,
                 channels: List[QCMDDistributionChannel],
                 distribution_system: DistributionSingleValve,
                 gsioc: GSIOC,
                 inlet_bubble_sensor: BubbleSensorBase,
                 outlet_bubble_sensor: BubbleSensorBase,
                 name='') -> None:
        NestedAssemblyBase.__init__(self, [], channels + [distribution_system], name)
        #AssemblyBase.__init__(self, self.devices, name)

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
            ch.methods.update({
                               'DirectInject': QCMDDirectInject(ch, gsioc),
                               'DirectInjectBubbleSensor': QCMDDirectInjectBubbleSensor(ch, gsioc, inlet_bubble_sensor, outlet_bubble_sensor)
                               })
            ch.methods['DirectInject'].devices += distribution_system.devices
            ch.methods['DirectInjectBubbleSensor'].devices += distribution_system.devices
            ch.modes['Inject'] = distribution_system.modes[str(3 + i)]
            ch.modes['Waste'] = distribution_system.modes['7']
            ch.modes['Standby'] = distribution_system.modes['8']

        self.channels = channels
        self.distribution_system = distribution_system

    async def initialize(self) -> None:
        """Initialize the loop as a unit and the distribution valve separately"""
        await asyncio.gather(*[ch.initialize() for ch in self.channels], self.distribution_system.initialize())
        await self.trigger_update()

    def run_channel_method(self, channel: int, method_name: str, method_data: dict) -> None:
        return self.channels[channel].run_method(method_name, method_data)
    
    def create_web_app(self, template='roadmap.html') -> Application:
        app = super().create_web_app(template=template)
        routes = web.RouteTableDef()

        @routes.post('/SubmitTask')
        async def handle_task(request: web.Request) -> web.Response:
            # TODO: turn task into a dataclass; parsing will change
            task = await request.json()
            channel: int = task['channel']
            method_name: str = task['method_name']
            method_data: dict = task['method_data']
            if self.channels[channel].is_ready(method_name):
                self.run_channel_method(channel, method_name, method_data)
                await self.trigger_update()
                return web.Response(text='accepted', status=200)
            
            return web.Response(text='busy', status=200)
        
        @routes.get('/GetTaskData')
        async def get_task(request: web.Request) -> web.Response:
            # TODO: turn task into a dataclass; parsing will change
            task = await request.json()
            task_id = task['id']

            # TODO: actually return task data
            # TODO: Determine what task data we want to save. Logging? success? Any returned errors?
            return web.Response(text=json.dumps({'id': task_id}), status=200)
        
        app.add_routes(routes)

        for i, channel in enumerate(self.channels):
            app.add_subapp(f'/{i}/', channel.create_web_app())

        return app

class QCMDLoop(AssemblyBasewithGSIOC):

    """TODO: Add distribution valve to init and to modes. Can also reduce # of modes because
        distribution valve is set once at the beginning of the method and syringe pump smart
        dispense takes care of aspirate/dispense"""

    def __init__(self, loop_valve: ValvePositionerBase,
                       syringe_pump: SyringePumpwithBubbleSensor,
                       flow_cell: FlowCell,
                       sample_loop: FlowCell,
                       injection_node: Node | None = None,
                       qcmd_address: str = 'localhost',
                       qcmd_port: int = 5011,
                       name: str = '') -> None:
        
        # Devices
        self.loop_valve = loop_valve
        self.syringe_pump = syringe_pump
        self.flow_cell = flow_cell
        self.sample_loop = sample_loop
        self.injection_node = injection_node
        super().__init__([loop_valve, syringe_pump], name=name)

        # Measurement device
        self.recorder = QCMDRecorder(http_address=f'http://{qcmd_address}:{qcmd_port}', name=f'{self.name}.QCMDRecorder')

        # Define node connections for dead volume estimations
        self.network = Network(self.devices + [self.flow_cell, self.sample_loop])

        # Event that signals when the LH is done
        self.release_liquid_handler: asyncio.Event = asyncio.Event()

        # Dead volume queue
        self.dead_volume: asyncio.Queue = asyncio.Queue(1)

        # Bubble sensor volume offset
        self.bubble_sensor_offset = 90

        # Measurement modes
        self.modes = {'Standby': Mode({loop_valve: 0,
                                       syringe_pump: 0},
                                       final_node=syringe_pump.valve.nodes[2]),
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
        
        # Control locks
        # Measurement lock indicates that a measurement is occurring and the cell should not be
        # exchanged or disturbed
        self.measurement_lock: asyncio.Lock = asyncio.Lock()

        # Channel lock indicates that the hardware in the channel is being used.
        self.channel_lock: asyncio.Lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Overwrites base initialization to ensure valves and pumps are in appropriate mode for homing syringe"""

        # initialize loop valve
        await self.loop_valve.initialize()

        # move to a position where loop goes to waste
        await self.loop_valve.move_valve(self.modes['PumpPrimeLoop'].valves[self.loop_valve])

        # initialize syringe pump. If plunger not homed, this will push solution into the loop
        await self.syringe_pump.initialize()

        # If syringe pump was already initialized, plunger may not be homed. Force it to home.
        await self.change_mode('PumpPrimeLoop')
        await self.syringe_pump.home()

        # change to standby mode
        await self.change_mode('Standby')

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:

        # overwrites base class handling of dead volume
        if data.data == 'V':
            dead_volume = await self.dead_volume.get()
            #self.logger.info(f'Sending dead volume {dead_volume}')
            response = f'{dead_volume:0.0f}'
        else:
            response = await super().handle_gsioc(data)
        
        return response
    
    async def QCMDRecord(self,
                         tag_name: str = '',
                         sleep_time: str | float = 0, # seconds
                         record_time: str | float = 0 # seconds
                        ) -> None:
        """QCMDRecord standalone method. Locks measurements"""

        sleep_time = float(sleep_time)
        record_time = float(record_time)

        # Locks the measurement system and records data
        await self.measurement_lock.acquire()
        await self.record(tag_name, record_time, sleep_time)

    async def record(self, tag_name: str, record_time: float, sleep_time: float):
        # helper function that performs the measurement and then releases the lock
        # this allows the lock to be passed to the record function
        await self.recorder.record(tag_name, record_time, sleep_time)
        self.measurement_lock.release()

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

    async def PrimeLoop(self,
                        n_prime: int | str = 1 # number of repeats
                         ) -> None:
        """PrimeLoop standalone method"""

        n_prime = int(n_prime)

        async with self.channel_lock:
            await self.primeloop(n_prime)

    async def LoopInject(self,
                         pump_volume: str | float = 0, # uL
                         pump_flow_rate: str | float = 1, # mL/min
                         excess_volume: str | float = 0, #uL
                         tag_name: str = '',
                         sleep_time: str | float = 0, # seconds
                         record_time: str | float = 0 # seconds
                         ) -> None:
        """LoopInject method, synchronized via GSIOC to liquid handler"""

        pump_volume = float(pump_volume)
        pump_flow_rate = float(pump_flow_rate) * 1000 / 60 # convert to uL / s
        sleep_time = float(sleep_time)
        record_time = float(record_time)

        # Clear liquid handler event if it isn't already cleared
        self.release_liquid_handler.clear()

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.get_dead_volume(self.injection_node, 'LoadLoop')

        # blocks if there's already something in the dead volume queue
        await self.dead_volume.put(dead_volume)
        self.logger.info(f'{self.name}.LoopInject: dead volume set to {dead_volume}')

        # locks the channel so any additional calling processes have to wait
        async with self.channel_lock:

            # switch to standby mode
            self.logger.info(f'{self.name}.LoopInject: Switching to Standby mode')            
            await self.change_mode('Standby')

            # Wait for trigger to switch to LoadLoop mode
            self.logger.info(f'{self.name}.LoopInject: Waiting for first trigger')
            await self.wait_for_trigger()
            self.logger.info(f'{self.name}.LoopInject: Switching to LoadLoop mode')
            await self.change_mode('LoadLoop')

            # Wait for trigger to switch to PumpAspirate mode
            self.logger.info(f'{self.name}.LoopInject: Waiting for second trigger')
            await self.wait_for_trigger()

            # At this point, liquid handler is done
            self.release_liquid_handler.set()

            self.logger.info(f'{self.name}.LoopInject: Switching to PumpPrimeLoop mode')
            await self.change_mode('PumpPrimeLoop')

            # smart dispense the volume required to move plug quickly through loop
            self.logger.info(f'{self.name}.LoopInject: Moving plug through loop, total injection volume {self.sample_loop.get_volume() - (pump_volume + excess_volume)} uL')
            await self.syringe_pump.smart_dispense(self.sample_loop.get_volume() - (pump_volume + excess_volume), self.syringe_pump.max_dispense_flow_rate)

            # waits until any current measurements are complete. Note that this could be done with
            # "async with measurement_lock" but then QCMDRecord would have to grab the lock as soon as it
            # was released, and there may be a race condition with other subroutines.
            # This function allows QCMDRecord to release the lock when it is done.
            self.logger.info(f'{self.name}.LoopInject: Waiting to acquire measurement lock')
            await self.measurement_lock.acquire()

            # change to inject mode
            await self.change_mode('PumpInject')
            self.logger.info(f'{self.name}.LoopInject: Injecting {pump_volume} uL at flow rate {pump_flow_rate} uL / s')
            await self.syringe_pump.smart_dispense(pump_volume, pump_flow_rate)

            # start QCMD timer
            self.logger.info(f'{self.name}.LoopInject: Starting QCMD timer for {sleep_time + record_time} seconds')

            # spawn new measurement task that will release measurement lock when complete
            self.run_method(self.record(tag_name, record_time, sleep_time))

            # Prime loop
            await self.primeloop(volume=1000)

    async def LoopInjectwithBubble(self,
                         pump_volume: str | float = 0, # uL
                         pump_flow_rate: str | float = 1, # mL/min
                         air_gap: str | float = 0, #uL
                         tag_name: str = '',
                         sleep_time: str | float = 0, # seconds
                         record_time: str | float = 0 # seconds
                         ) -> None:
        """LoopInjectwithBubble method, synchronized via GSIOC to liquid handler"""

        pump_volume = float(pump_volume)
        pump_flow_rate = float(pump_flow_rate) * 1000 / 60 # convert to uL / s
        sleep_time = float(sleep_time)
        record_time = float(record_time)

        # Clear liquid handler event if it isn't already cleared
        self.release_liquid_handler.clear()

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.get_dead_volume(self.injection_node, 'LoadLoop')

        # blocks if there's already something in the dead volume queue
        await self.dead_volume.put(dead_volume)
        self.logger.info(f'{self.name}.LoopInjectwithBubble: dead volume set to {dead_volume}')

        # locks the channel so any additional calling processes have to wait
        async with self.channel_lock:

            # power the bubble sensor
            await self.syringe_pump.set_digital_output(1, True)

            # switch to standby mode
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Switching to Standby mode')            
            await self.change_mode('Standby')

            # Wait for trigger to switch to LoadLoop mode
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Waiting for first trigger')
            await self.wait_for_trigger()
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Switching to LoadLoop mode')
            await self.change_mode('LoadLoop')

            # Wait for trigger to switch to PumpAspirate mode
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Waiting for second trigger')
            await self.wait_for_trigger()

            # At this point, liquid handler is done
            self.release_liquid_handler.set()

            self.logger.info(f'{self.name}.LoopInjectwithBubble: Switching to PumpPrimeLoop mode')
            await self.change_mode('PumpPrimeLoop')

            # smart dispense the volume required to move plug quickly through loop, interrupting if sensor 1 goes low
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Moving plug through loop until air gap detected, total injection volume {self.sample_loop.get_volume() - (pump_volume)} uL')
            actual_volume = await self.syringe_pump.smart_dispense(self.sample_loop.get_volume() - pump_volume, self.syringe_pump.max_dispense_flow_rate, 5)
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Actually injected {actual_volume} uL')

            async def traverse_air_gap(nominal_air_gap: float, flow_rate: float = self.syringe_pump.max_dispense_flow_rate, volume_step: float = 10) -> float:
                
                total_air_gap_volume = 0
                total_air_gap_volume += await self.syringe_pump.smart_dispense(nominal_air_gap, flow_rate)
                while not (await self.syringe_pump.get_digital_input(1)):
                    total_air_gap_volume += await self.syringe_pump.smart_dispense(volume_step, flow_rate)
                
                return total_air_gap_volume

            # dispense air gap + bubble offset without interruption
            #await self.syringe_pump.smart_dispense(air_gap + self.bubble_sensor_offset, self.syringe_pump.max_dispense_flow_rate)
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Traversing air gap...')
            total_air_gap_volume = await traverse_air_gap(air_gap, self.syringe_pump.max_dispense_flow_rate)
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Total air gap volume: {total_air_gap_volume} uL')
            actual_volume = await self.syringe_pump.smart_dispense(self.bubble_sensor_offset + 30, self.syringe_pump.max_dispense_flow_rate)
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Bubble sensor offset dispensed: {actual_volume} uL')

            # waits until any current measurements are complete. Note that this could be done with
            # "async with measurement_lock" but then QCMDRecord would have to grab the lock as soon as it
            # was released, and there may be a race condition with other subroutines.
            # This function allows QCMDRecord to release the lock when it is done.
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Waiting to acquire measurement lock')
            await self.measurement_lock.acquire()

            # change to inject mode
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Injecting {pump_volume} uL at flow rate {pump_flow_rate} uL / s')
            await self.change_mode('PumpInject')
            actual_volume = await self.syringe_pump.smart_dispense(pump_volume, pump_flow_rate, 5)
            # TODO: Replace 20 with a bubble_sensor_offset buffer value (+ for initial injection, - for this one)
            extra_volume = min(pump_volume - actual_volume, self.bubble_sensor_offset - 20)
            if extra_volume > 0:
                await self.syringe_pump.smart_dispense(extra_volume, pump_flow_rate)
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Injected {actual_volume} uL at flow rate {pump_flow_rate} uL / s, plus extra_volume {extra_volume}')

            # start QCMD timer
            self.logger.info(f'{self.name}.LoopInjectwithBubble: Starting QCMD timer for {sleep_time + record_time} seconds')

            # spawn new measurement task that will release measurement lock when complete
            self.run_method(self.record(tag_name, record_time, sleep_time))

            # Flush loop with any excess volume in the pump
            await self.change_mode('PumpPrimeLoop')
            await self.syringe_pump.run_syringe_until_idle(self.syringe_pump.home())

            # Prime loop
            await self.primeloop(volume=1000)

    async def LHInject(self,
                         tag_name: str = '',
                         sleep_time: str | float = 0, # seconds
                         record_time: str | float = 0 # seconds
                         ) -> None:
        """LH Direct Inject method, synchronized via GSIOC to liquid handler"""

        sleep_time = float(sleep_time)
        record_time = float(record_time)

        # Clear liquid handler event if it isn't already cleared
        self.release_liquid_handler.clear()

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.get_dead_volume(self.injection_node, 'LHInject')

        # blocks if there's already something in the dead volume queue
        await self.dead_volume.put(dead_volume)
        self.logger.info(f'{self.name}.LHInject: dead volume set to {dead_volume}')

        # locks the channel so any additional calling processes have to wait
        async with self.channel_lock:

             # switch to standby mode
            self.logger.info(f'{self.name}.LHInject: Switching to Standby mode')            
            await self.change_mode('Standby')

            # Wait for trigger to switch to LHPrime mode (fast injection of air gap + dead volume + extra volume)
            self.logger.info(f'{self.name}.LHInject: Waiting for first trigger')
            await self.wait_for_trigger()
            self.logger.info(f'{self.name}.LHInject: Switching to LHPrime mode')
            await self.change_mode('LHPrime')

            # waits until any current measurements are complete. Note that this could be done with
            # "async with measurement_lock" but then QCMDRecord would have to grab the lock as soon as it
            # was released, and there may be a race condition with other subroutines.
            # This function allows QCMDRecord to release the lock when it is done.
            self.logger.info(f'{self.name}.LHInject: Waiting to acquire measurement lock')
            await self.measurement_lock.acquire()

            # Wait for trigger to switch to LHInject mode (LH performs injection)
            self.logger.info(f'{self.name}.LHInject: Waiting for second trigger')
            await self.wait_for_trigger()

            self.logger.info(f'{self.name}.LHInject: Switching to LHInject mode')
            await self.change_mode('LHInject')

            # Wait for trigger to switch to LHPrime mode (fast injection of extra volume + final air gap)
            self.logger.info(f'{self.name}.LHInject: Waiting for third trigger')
            await self.wait_for_trigger()

            self.logger.info(f'{self.name}.LHInject: Switching to LHPrime mode')
            await self.change_mode('LHPrime')

            # start QCMD timer
            self.logger.info(f'{self.name}.LHInject: Starting QCMD timer for {sleep_time + record_time} seconds')

            # spawn new measurement task that will release measurement lock when complete
            self.run_method(self.record(tag_name, record_time, sleep_time))

            # Wait for trigger to switch to Standby mode (this may not be necessary)
            self.logger.info(f'{self.name}.LHInject: Waiting for fourth trigger')
            await self.wait_for_trigger()

            # Clear liquid handler event if it isn't already cleared
            self.release_liquid_handler.set()

            self.logger.info(f'{self.name}.LHInject: Switching to Standby mode')            
            await self.change_mode('Standby')

            #self.logger.info(f'{self.name}.LHInject: Priming loop')

            # Prime loop
            #await self.primeloop()

    async def LHInjectwithBubble(self,
                         tag_name: str = '',
                         sleep_time: str | float = 0, # seconds
                         record_time: str | float = 0 # seconds
                         ) -> None:
        """LH Direct Inject method, synchronized via GSIOC to liquid handler"""

        """TODO:
                0. Check if we really need this. Problem is that we might want different loading and injection flow rates, and if we detect the air gap and switch the valve
                    too early, we will hit the cell with the high flow rate.
                    Better to rely on volumes on the front end (use the bubble sensor to test how well the dead volume calculation works for direct injection).
                    Use the bubble sensor to make sure the back end air gap doesn't get injected into the cell
                    (even at the injection flow rate, when the air gap hits the bubble sensor at the inlet we immediately switch to waste).
                1. In this scheme, all we have to do is switch the valve to waste as soon as the second air gap is detected at the inlet.
                2. Have to make sure it doesn't hang indefinitely; perhaps gather the H command with a sleep of the expected length (would need injection volume and flow rate as parameters)
                    and send R when the timer expires. Won't do anything if it's not hung up on the H command.
                
        """

        sleep_time = float(sleep_time)
        record_time = float(record_time)

        # Clear liquid handler event if it isn't already cleared
        self.release_liquid_handler.clear()

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.get_dead_volume(self.injection_node, 'LHInject')

        # blocks if there's already something in the dead volume queue
        await self.dead_volume.put(dead_volume)
        self.logger.info(f'{self.name}.LHInject: dead volume set to {dead_volume}')

        # locks the channel so any additional calling processes have to wait
        async with self.channel_lock:

             # switch to standby mode
            self.logger.info(f'{self.name}.LHInject: Switching to Standby mode')            
            await self.change_mode('Standby')

            # Wait for trigger to switch to LHPrime mode (fast injection of air gap + dead volume + extra volume)
            self.logger.info(f'{self.name}.LHInject: Waiting for first trigger')
            await self.wait_for_trigger()
            self.logger.info(f'{self.name}.LHInject: Switching to LHPrime mode')
            await self.change_mode('LHPrime')

            # waits until any current measurements are complete. Note that this could be done with
            # "async with measurement_lock" but then QCMDRecord would have to grab the lock as soon as it
            # was released, and there may be a race condition with other subroutines.
            # This function allows QCMDRecord to release the lock when it is done.
            self.logger.info(f'{self.name}.LHInject: Waiting to acquire measurement lock')
            await self.measurement_lock.acquire()

            # Wait for trigger to switch to LHInject mode (LH performs injection)
            self.logger.info(f'{self.name}.LHInject: Waiting for second trigger')
            await self.wait_for_trigger()

            self.logger.info(f'{self.name}.LHInject: Switching to LHInject mode')
            await self.change_mode('LHInject')

            # Wait for trigger to switch to LHPrime mode (fast injection of extra volume + final air gap)
            self.logger.info(f'{self.name}.LHInject: Waiting for third trigger')
            await self.wait_for_trigger()

            self.logger.info(f'{self.name}.LHInject: Switching to LHPrime mode')
            await self.change_mode('LHPrime')

            # start QCMD timer
            self.logger.info(f'{self.name}.LHInject: Starting QCMD timer for {sleep_time + record_time} seconds')

            # spawn new measurement task that will release measurement lock when complete
            self.run_method(self.record(tag_name, record_time, sleep_time))

            # Wait for trigger to switch to Standby mode (this may not be necessary)
            self.logger.info(f'{self.name}.LHInject: Waiting for fourth trigger')
            await self.wait_for_trigger()

            # Clear liquid handler event if it isn't already cleared
            self.release_liquid_handler.set()

            self.logger.info(f'{self.name}.LHInject: Switching to Standby mode')            
            await self.change_mode('Standby')

            #self.logger.info(f'{self.name}.LHInject: Priming loop')

            # Prime loop
            #await self.primeloop()

class QCMDSystem(NestedAssemblyBase, AssemblyBasewithGSIOC):
    """QCMD System comprising one QCMD loop and one distribution valve
        (unnecessarily complex but testbed for ROADMAP multichannel assembly)"""
    
    def __init__(self, distribution_valve: ValvePositionerBase, qcmd_loop: QCMDLoop, injection_port: InjectionPort, name: str = 'QCMDSystem') -> None:
        NestedAssemblyBase.__init__(self, [distribution_valve], [qcmd_loop], name)
        AssemblyBasewithGSIOC.__init__(self, self.devices, name)

        self.injection_port = injection_port
        self.distribution_valve = distribution_valve

        # Distribution lock indicates that the distribution valve is being used
        self.distribution_lock: asyncio.Lock = asyncio.Lock()

        # Build network
        self.network = Network(qcmd_loop.network.devices + [distribution_valve, injection_port])

        self.modes = {'Standby': Mode(valves={distribution_valve: 8}), # don't do anything to the qcmd_loop for standby
                      'LoopInject': Mode(valves={distribution_valve: 1}),
                      'LHInject': Mode(valves={distribution_valve: 2})}

        # for dead volume tracing, update qcmd loop with entire network and with injection node
        qcmd_loop.network = self.network
        qcmd_loop.injection_node = injection_port.nodes[0]
        self.qcmd_loop = qcmd_loop

    async def initialize(self) -> None:
        """Initialize the loop as a unit and the distribution valve separately"""
        await asyncio.gather(self.qcmd_loop.initialize(), self.distribution_valve.initialize())

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:

        # defers dead volume calculation to qcmd_loop
        if data.data == 'V':
            dead_volume = await self.qcmd_loop.dead_volume.get()
            #self.logger.info(f'Sending dead volume {dead_volume}')
            response = f'{dead_volume:0.0f}'
        elif data.data.startswith('{'):
            response = await super().handle_gsioc(data)
        else:
            response = await self.qcmd_loop.handle_gsioc(data)
        
        return response
    
    async def LoopInject(self,
                         **kwargs
                         ) -> None:
        """LoopInject method, synchronized via GSIOC to liquid handler"""

        # start new method if distribution lock is free
        async with self.distribution_lock:

            # switch to appropriate mode
            await self.change_mode('LoopInject')

            # spawn Loop Injection task
            asyncio.create_task(self.qcmd_loop.LoopInject(**kwargs))

            # wait for liquid handler to be released
            await self.qcmd_loop.release_liquid_handler.wait()
            self.qcmd_loop.release_liquid_handler.clear()

            # switch to standby mode
            await self.change_mode('Standby')

    async def LoopInjectwithBubble(self,
                         **kwargs
                         ) -> None:
        """LoopInject method, synchronized via GSIOC to liquid handler"""

        # start new method if distribution lock is free
        async with self.distribution_lock:

            # switch to appropriate mode
            await self.change_mode('LoopInject')

            # spawn Loop Injection task
            asyncio.create_task(self.qcmd_loop.LoopInjectwithBubble(**kwargs))

            # wait for liquid handler to be released
            await self.qcmd_loop.release_liquid_handler.wait()
            self.qcmd_loop.release_liquid_handler.clear()

            # switch to standby mode
            await self.change_mode('Standby')

    async def LHInject(self,
                         **kwargs
                         ) -> None:
        """LHInject method, synchronized via GSIOC to liquid handler"""

        # start new method if distribution lock is free
        async with self.distribution_lock:

            # switch to appropriate mode
            await self.change_mode('LHInject')

            # spawn LH Direct Injection task
            asyncio.create_task(self.qcmd_loop.LHInject(**kwargs))

            # wait for liquid handler to be released
            await self.qcmd_loop.release_liquid_handler.wait()
            self.qcmd_loop.release_liquid_handler.clear()

            # switch to standby mode
            await self.change_mode('Standby')

    async def PrimeLoop(self, **kwargs) -> None:
        """PrimeLoop method"""

        # spawn LH Direct Injection task
        asyncio.create_task(self.qcmd_loop.PrimeLoop(**kwargs))


async def qcmd_loop():
    gsioc = GSIOC(62, 'COM13', 19200)
    ser = HamiltonSerial(port='COM5', baudrate=38400)
    mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='loop_valve_positioner')
    sp = HamiltonSyringePump(ser, '0', SyringeLValve(4, name='syringe_LValve'), 5000., False, name='syringe_pump')
    sp.max_dispense_flow_rate = 5 * 1000 / 60
    sp.max_aspirate_flow_rate = 15 * 1000 / 60
    ip = InjectionPort('LH_injection_port')
    fc = FlowCell(139, 'flow_cell')
    sampleloop = FlowCell(5060., 'sample_loop')

    # connect syringe pump valve port 2 to LH injection port
    connect_nodes(ip.nodes[0], mvp.valve.nodes[3], 144)

    # connect syringe pump valve port 3 to sample loop
    connect_nodes(sp.valve.nodes[3], sampleloop.inlet_node, 0.0)

    # connect sample loop to loop valve port 1
    connect_nodes(mvp.valve.nodes[1], sampleloop.outlet_node, 0.0)

    # connect cell inlet to loop valve port 2
    connect_nodes(mvp.valve.nodes[2], fc.inlet_node, 0.0)

    # connect cell outlet to loop valve port 5
    connect_nodes(mvp.valve.nodes[5], fc.outlet_node, 0.0)

    qcmd_channel = QCMDLoop(mvp, sp,fc, sampleloop, injection_node=ip.nodes[0], name='QCMD Channel')
    qcmd_channel.network.add_device(ip)

    #lh = SimLiquidHandler(qcmd_channel)

    try:
        #print(qcmd_channel.get_dead_volume(qcmd_channel.injection_node, 'LHInject'))
        #print(qcmd_channel.get_dead_volume(qcmd_channel.injection_node, 'LoadLoop'))
        await sp.initialize()
        await qcmd_channel.initialize()
        #await qcmd_channel.change_mode('PumpPrimeLoop')
        #await asyncio.sleep(1)
        #await qcmd_channel.primeloop(2)
        #await mvp.initialize()
        #await mvp.run_until_idle(mvp.move_valve(1))
        #await sp.initialize()
        #await sp.run_until_idle(sp.move_absolute(0, sp._speed_code(sp.max_dispense_flow_rate)))

        gsioc_task = asyncio.create_task(qcmd_channel.initialize_gsioc(gsioc))

        await gsioc_task
    finally:
        logging.info('Cleaning up...')
        await qcmd_channel.recorder.session.close()

async def qcmd_distribution():
    gsioc = GSIOC(62, 'COM13', 19200)
    ser = HamiltonSerial(port='COM9', baudrate=38400)
    #ser = HamiltonSerial(port='COM3', baudrate=38400)
    dvp = HamiltonValvePositioner(ser, '2', DistributionValve(8, name='distribution_valve'), name='Distribution Valve')
    mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='Loop Valve')
    sp = SyringePumpwithBubbleSensor(ser, '0', SyringeLValve(4, name='syringe_LValve'), 5000., False, name='Syringe Pump')
    sp.max_dispense_flow_rate = 5 * 1000 / 60
    sp.max_aspirate_flow_rate = 15 * 1000 / 60
    ip = InjectionPort('LH_injection_port')
    fc = FlowCell(139, 'flow_cell')
    sampleloop = FlowCell(4994., 'sample_loop')

    qcmd_channel = QCMDLoop(mvp, sp, fc, sampleloop, injection_node=ip.nodes[0], name='QCMD Channel')

    # connect LH injection port to distribution port valve 0
    connect_nodes(ip.nodes[0], dvp.valve.nodes[0], 124 + 50)

    # connect distribution valve port 1 to syringe pump valve node 2 (top)
    connect_nodes(dvp.valve.nodes[1], sp.valve.nodes[2], 73 + 20)

    # connect distribution valve port 2 to loop valve node 3 (top right)
    connect_nodes(dvp.valve.nodes[2], mvp.valve.nodes[3], 90 + 50)

    # connect syringe pump valve port 3 to sample loop
    connect_nodes(sp.valve.nodes[3], sampleloop.inlet_node, 0.0)

    # connect sample loop to loop valve port 1
    connect_nodes(mvp.valve.nodes[1], sampleloop.outlet_node, 0.0)

    # connect cell inlet to loop valve port 2
    connect_nodes(mvp.valve.nodes[2], fc.inlet_node, 0.0)

    # connect cell outlet to loop valve port 5
    connect_nodes(mvp.valve.nodes[5], fc.outlet_node, 0.0)

    qcmd_system = QCMDSystem(dvp, qcmd_channel, ip, name='QCMD System')
    app = qcmd_system.create_web_app(template='roadmap.html')
    runner = await run_socket_app(app, 'localhost', 5003)
    print(json.dumps(await qcmd_system.get_info()))
    #lh = SimLiquidHandler(qcmd_channel)

    try:
        #qcmd_system.distribution_valve.valve.move(2)
        await qcmd_system.initialize()
        await sp.set_digital_output(1, True)
        await sp.set_digital_output(2, True)
        #await asyncio.sleep(2)
        #await qcmd_channel.change_mode('PumpPrimeLoop')

        #await qcmd_channel.primeloop(2)
        #await qcmd_system.change_mode('LoopInject')
        #await qcmd_channel.change_mode('LoadLoop')
        #await asyncio.sleep(2)
        #await qcmd_system.change_mode('LoopInject')
        #await asyncio.sleep(2)
        #await qcmd_system.change_mode('Standby')

        #await qcmd_channel.change_mode('PumpPrimeLoop')
        #await mvp.initialize()
        #await mvp.run_until_idle(mvp.move_valve(1))
        #await sp.initialize()
        #await sp.run_until_idle(sp.move_absolute(0, sp._speed_code(sp.max_dispense_flow_rate)))

        gsioc_task = asyncio.create_task(qcmd_system.initialize_gsioc(gsioc))

        await gsioc_task
    finally:
        logging.info('Cleaning up...')
        asyncio.gather(qcmd_channel.recorder.session.close(),
                       runner.cleanup())

async def qcmd_single_distribution():
    gsioc = GSIOC(62, 'COM13', 19200)
    ser = HamiltonSerial(port='COM9', baudrate=38400)
    #ser = HamiltonSerial(port='COM3', baudrate=38400)
    dvp = HamiltonValvePositioner(ser, '2', DistributionValve(8, name='distribution_valve'), name='Distribution Valve')
    #sp = SyringePumpwithBubbleSensor(ser, '0', SyringeLValve(4, name='syringe_LValve'), 5000., False, name='Syringe Pump')
    ip = InjectionPort('LH_injection_port')
    fc0 = FlowCell(139, 'flow_cell')
    fc1 = FlowCell(139, 'flow_cell')
    inlet_bubble_sensor = SMDSensoronHamiltonDevice(dvp, 2, 1)
    outlet_bubble_sensor = SMDSensoronHamiltonDevice(dvp, 1, 0)

    ch0 = QCMDDistributionChannel(fc0, injection_node=ip.nodes[0], name='QCMD Channel 0')
    ch1 = QCMDDistributionChannel(fc0, injection_node=ip.nodes[0], name='QCMD Channel 1')
    distribution_system = DistributionSingleValve(dvp, ip, 'Distribution System')

    # connect LH injection port to distribution port valve 0
    connect_nodes(ip.nodes[0], dvp.valve.nodes[0], 124 + 50)

    # connect distribution valve port 3 to channel 0 flow cell
    connect_nodes(dvp.valve.nodes[3], fc0.inlet_node, 73 + 20)

    # connect distribution valve port 3 to channel 0 flow cell
    connect_nodes(dvp.valve.nodes[4], fc1.inlet_node, 83 + 20)

    qcmd_system = QCMDDistributionAssembly([ch0, ch1], distribution_system, gsioc, inlet_bubble_sensor, outlet_bubble_sensor, name='QCMD MultiChannel System')
    app = qcmd_system.create_web_app(template='roadmap.html')
    runner = await run_socket_app(app, 'localhost', 5003)
    print(json.dumps(await qcmd_system.get_info()))
    #lh = SimLiquidHandler(qcmd_channel)

    try:
        await qcmd_system.initialize()
        await gsioc.listen()

        # testing: curl -X POST http://localhost:5003/SubmitTask -d "{\"channel\": 0, \"method_name\": \"DirectInjectBubbleSensor\", \"method_data\": {}}"
    except asyncio.CancelledError:
        pass
    finally:
        logging.info('Cleaning up...')
        asyncio.gather(
                       runner.cleanup())

if __name__=='__main__':

    import datetime

    if False:
        logging.basicConfig(handlers=[
                                logging.FileHandler(datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_qcmd_recorder_log.txt'),
                                logging.StreamHandler()
                            ],
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO)

        asyncio.run(main(), debug=True)
    else:
        logging.basicConfig(
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO)
        asyncio.run(qcmd_distribution(), debug=True)
        #asyncio.run(sptest())
