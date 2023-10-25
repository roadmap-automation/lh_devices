from typing import Any, Coroutine, List
import aiohttp
import asyncio
import logging
from HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump, HamiltonSerial
from valve import LoopFlowValve, SyringeLValve, DistributionValve
from components import InjectionPort, FlowCell, Node
from gsioc import GSIOC, GSIOCMessage
from assemblies import AssemblyBasewithGSIOC, Network, connect_nodes, Mode
from liquid_handler import SimLiquidHandler

class Timer:
    """Basic timer. Essentially serves as a sleep but only allows one instance to run."""

    def __init__(self, name='Timer') -> None:
        self.name = name
        self.timer_running: asyncio.Event = asyncio.Event()

    async def start(self, wait_time: float = 0.0) -> bool:
        """Executes timer.

        Returns:
            bool: True if successful, False if not.
        """

        # don't start another timer if one is already running
        if not self.timer_running.is_set():
            self.timer_running.set()
            await asyncio.sleep(wait_time)
            self.timer_running.clear()
            return True
        else:
            logging.warning(f'{self.name}: Timer is already running, ignoring start command')
            return False

class QCMDRecorder(Timer):
    """QCMD-specific timer. At end of timing interval, sends HTTP request to QCMD to record tag."""

    def __init__(self, qcmd_address: str = 'localhost', qcmd_port: int = 5011, name='QCMDRecorder') -> None:
        super().__init__(name)
        self.session = aiohttp.ClientSession(f'http://{qcmd_address}:{qcmd_port}')

    async def record(self, tag_name: str = '', record_time: float = 0.0, sleep_time: float = 0.0) -> None:
        """Executes timer and sends record command to QCMD. Call by sending
            {"method": "record", {**kwargs}} over GSIOC.
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # calculate total wait time
        wait_time = record_time + sleep_time

        # wait the full time
        if await self.start(wait_time):

            post_data = {'command': 'set_tag',
                        'value': {'tag': tag_name,
                                'delta_t': record_time}}

            logging.info(f'{self.session._base_url}/QCMD/ => {post_data}')

            # send an http request to QCMD server
            try:
                async with self.session.post('/QCMD/', json=post_data, timeout=10) as resp:
                    response_json = await resp.json()
                    logging.info(f'{self.session._base_url}/QCMD/ <= {response_json}')
            except (ConnectionRefusedError, aiohttp.ClientConnectorError):
                logging.error(f'request to {self.session._base_url}/QCMD/ failed: connection refused')

class QCMDRecorderDevice(AssemblyBasewithGSIOC):
    """QCMD recording device."""

    def __init__(self, qcmd_address: str = 'localhost', qcmd_port: int = 5011, name='QCMDRecorderDevice') -> None:
        super().__init__([], name)
        self.recorder = QCMDRecorder(qcmd_address, qcmd_port, f'{self.name}.QCMDRecorder')

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:
        """Handles GSIOC message but deals with Q more robustly than the base method"""

        if data.data == 'Q':
            response = 'busy' if self.recorder.timer_running.is_set() else 'idle'
        else:
            response = await super().handle_gsioc(data)

        return response

    async def QCMDRecord(self, tag_name: str = '', record_time: str | float = 0.0, sleep_time: str | float = 0.0) -> None:
        """Executes timer and sends record command to QCMD. Call by sending
            {"method": "record", {**kwargs}} over GSIOC.
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # wait the full time
        await self.recorder.record(tag_name, record_time, sleep_time)

class QCMDLoop(AssemblyBasewithGSIOC):

    """TODO: Add distribution valve to init and to modes. Can also reduce # of modes because
        distribution valve is set once at the beginning of the method and syringe pump smart
        dispense takes care of aspirate/dispense"""

    def __init__(self, loop_valve: HamiltonValvePositioner,
                       syringe_pump: HamiltonSyringePump,
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
        self.recorder = QCMDRecorder(qcmd_address, qcmd_port, f'{self.name}.QCMDRecorder')

        # Define node connections for dead volume estimations
        self.network = Network(self.devices + [self.flow_cell, self.sample_loop])

        # Event that signals when the LH is done
        self.release_liquid_handler: asyncio.Event = asyncio.Event()

        # Dead volume queue
        self.dead_volume: asyncio.Queue = asyncio.Queue(1)

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
            #logging.info(f'Sending dead volume {dead_volume}')
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

    async def record(self, tag_name: str, sleep_time: float, record_time: float):
        # helper function that performs the measurement and then releases the lock
        # this allows the lock to be passed to the record function
        await self.recorder.record(tag_name, sleep_time, record_time)
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
        logging.info(f'{self.name}.LoopInject: dead volume set to {dead_volume}')

        # locks the channel so any additional calling processes have to wait
        async with self.channel_lock:

            # switch to standby mode
            logging.info(f'{self.name}.LoopInject: Switching to Standby mode')            
            await self.change_mode('Standby')

            # Wait for trigger to switch to LoadLoop mode
            logging.info(f'{self.name}.LoopInject: Waiting for first trigger')
            await self.wait_for_trigger()
            logging.info(f'{self.name}.LoopInject: Switching to LoadLoop mode')
            await self.change_mode('LoadLoop')

            # Wait for trigger to switch to PumpAspirate mode
            logging.info(f'{self.name}.LoopInject: Waiting for second trigger')
            await self.wait_for_trigger()

            # At this point, liquid handler is done
            self.release_liquid_handler.set()

            logging.info(f'{self.name}.LoopInject: Switching to PumpPrimeLoop mode')
            await self.change_mode('PumpPrimeLoop')

            # smart dispense the volume required to move plug quickly through loop
            logging.info(f'{self.name}.LoopInject: Moving plug through loop, total injection volume {self.sample_loop.get_volume() - (pump_volume + excess_volume)} uL')
            await self.syringe_pump.smart_dispense(self.sample_loop.get_volume() - (pump_volume + excess_volume), self.syringe_pump.max_dispense_flow_rate)

            # waits until any current measurements are complete. Note that this could be done with
            # "async with measurement_lock" but then QCMDRecord would have to grab the lock as soon as it
            # was released, and there may be a race condition with other subroutines.
            # This function allows QCMDRecord to release the lock when it is done.
            logging.info(f'{self.name}.LoopInject: Waiting to acquire measurement lock')
            await self.measurement_lock.acquire()

            # change to inject mode
            await self.change_mode('PumpInject')
            logging.info(f'{self.name}.LoopInject: Injecting {pump_volume} uL at flow rate {pump_flow_rate} uL / s')
            await self.syringe_pump.smart_dispense(pump_volume, pump_flow_rate)

            # start QCMD timer
            logging.info(f'{self.name}.LoopInject: Starting QCMD timer for {sleep_time + record_time} seconds')

            # spawn new measurement task that will release measurement lock when complete
            self.run_method(self.record(tag_name, sleep_time, record_time))

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
        logging.info(f'{self.name}.LHInject: dead volume set to {dead_volume}')

        # locks the channel so any additional calling processes have to wait
        async with self.channel_lock:

             # switch to standby mode
            logging.info(f'{self.name}.LHInject: Switching to Standby mode')            
            await self.change_mode('Standby')

            # Wait for trigger to switch to LHPrime mode (fast injection of air gap + dead volume + extra volume)
            logging.info(f'{self.name}.LHInject: Waiting for first trigger')
            await self.wait_for_trigger()
            logging.info(f'{self.name}.LHInject: Switching to LHPrime mode')
            await self.change_mode('LHPrime')

            # waits until any current measurements are complete. Note that this could be done with
            # "async with measurement_lock" but then QCMDRecord would have to grab the lock as soon as it
            # was released, and there may be a race condition with other subroutines.
            # This function allows QCMDRecord to release the lock when it is done.
            logging.info(f'{self.name}.LHInject: Waiting to acquire measurement lock')
            await self.measurement_lock.acquire()

            # Wait for trigger to switch to LHInject mode (LH performs injection)
            logging.info(f'{self.name}.LHInject: Waiting for second trigger')
            await self.wait_for_trigger()

            logging.info(f'{self.name}.LHInject: Switching to LHInject mode')
            await self.change_mode('LHInject')

            # Wait for trigger to switch to LHPrime mode (fast injection of extra volume + final air gap)
            logging.info(f'{self.name}.LHInject: Waiting for third trigger')
            await self.wait_for_trigger()

            logging.info(f'{self.name}.LHInject: Switching to LHPrime mode')
            await self.change_mode('LHPrime')

            # start QCMD timer
            logging.info(f'{self.name}.LHInject: Starting QCMD timer for {sleep_time + record_time} seconds')

            # spawn new measurement task that will release measurement lock when complete
            self.run_method(self.record(tag_name, sleep_time, record_time))

            # Wait for trigger to switch to Standby mode (this may not be necessary)
            logging.info(f'{self.name}.LHInject: Waiting for fourth trigger')
            await self.wait_for_trigger()

            # Clear liquid handler event if it isn't already cleared
            self.release_liquid_handler.set()

            logging.info(f'{self.name}.LHInject: Switching to Standby mode')            
            await self.change_mode('Standby')

            #logging.info(f'{self.name}.LHInject: Priming loop')

            # Prime loop
            #await self.primeloop()

class QCMDSystem(AssemblyBasewithGSIOC):
    """QCMD System comprising one QCMD loop and one distribution valve
        (unnecessarily complex but testbed for ROADMAP multichannel assembly)"""
    
    def __init__(self, distribution_valve: HamiltonValvePositioner, qcmd_loop: QCMDLoop, injection_port: InjectionPort, name: str = 'QCMDSystem') -> None:
        super().__init__(devices=qcmd_loop.devices + [distribution_valve], name=name)

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
            #logging.info(f'Sending dead volume {dead_volume}')
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
    ser = HamiltonSerial(port='COM6', baudrate=38400)
    #ser = HamiltonSerial(port='COM3', baudrate=38400)
    dvp = HamiltonValvePositioner(ser, '2', DistributionValve(8, name='distribution_valve'), name='distribution_valve_positioner')
    mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='loop_valve_positioner')
    sp = HamiltonSyringePump(ser, '0', SyringeLValve(4, name='syringe_LValve'), 5000., False, name='syringe_pump')
    sp.max_dispense_flow_rate = 5 * 1000 / 60
    sp.max_aspirate_flow_rate = 15 * 1000 / 60
    ip = InjectionPort('LH_injection_port')
    fc = FlowCell(139, 'flow_cell')
    sampleloop = FlowCell(5060., 'sample_loop')

    qcmd_channel = QCMDLoop(mvp, sp, fc, sampleloop, injection_node=ip.nodes[0], name='QCMD Channel')

    # connect LH injection port to distribution port valve 0
    connect_nodes(ip.nodes[0], dvp.valve.nodes[0], 124 + 20)

    # connect distribution valve port 1 to syringe pump valve node 2 (top)
    connect_nodes(dvp.valve.nodes[1], sp.valve.nodes[2], 73 + 20)

    # connect distribution valve port 2 to loop valve node 3 (top right)
    connect_nodes(dvp.valve.nodes[2], mvp.valve.nodes[3], 82 + 20)

    # connect syringe pump valve port 3 to sample loop
    connect_nodes(sp.valve.nodes[3], sampleloop.inlet_node, 0.0)

    # connect sample loop to loop valve port 1
    connect_nodes(mvp.valve.nodes[1], sampleloop.outlet_node, 0.0)

    # connect cell inlet to loop valve port 2
    connect_nodes(mvp.valve.nodes[2], fc.inlet_node, 0.0)

    # connect cell outlet to loop valve port 5
    connect_nodes(mvp.valve.nodes[5], fc.outlet_node, 0.0)

    qcmd_system = QCMDSystem(dvp, qcmd_channel, ip, name='QCMD System')

    #lh = SimLiquidHandler(qcmd_channel)

    try:
        #qcmd_system.distribution_valve.valve.move(2)
        await qcmd_system.initialize()
        await asyncio.sleep(2)
        await qcmd_channel.change_mode('PumpPrimeLoop')
        await qcmd_channel.primeloop(2)
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
        await qcmd_channel.recorder.session.close()

async def main():
    gsioc = GSIOC(62, 'COM13', 19200)
    qcmd_recorder = QCMDRecorderDevice('localhost', 5011)
    try:
        await qcmd_recorder.initialize_gsioc(gsioc)
    finally:
        await qcmd_recorder.recorder.session.close()

async def sptest():
    ser = HamiltonSerial(port='COM5', baudrate=38400)
    sp = HamiltonSyringePump(ser, '0', SyringeLValve(4, name='syringe_LValve'), 5000, False, name='syringe_pump')
    #await sp.initialize()
    #await sp.get_syringe_position()
    #await sp.move_absolute(sp.syringe_position, sp._speed_code(10 * 1000 / 60))
    #await sp.move_valve(4)
    #await sp.aspirate(2500, 10*1000/60)    
    #sp.max_dispense_flow_rate = 20 * 1000 / 60
    #await sp.run_until_idle(sp.move_valve(sp.valve.dispense_position))
    #print(sp.valve.dispense_position)
    #await sp.run_until_idle(sp.home())
    #await sp.smart_dispense(sp.syringe_volume, sp.max_dispense_flow_rate)

    mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='loop_valve_positioner')
    await mvp.run_until_idle(mvp.initialize())

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
