from typing import Any, Coroutine, List
import aiohttp
import asyncio
import logging
from HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump, HamiltonSerial
from valve import LoopFlowValve, SyringeLValve
from components import InjectionPort, FlowCell
from gsioc import GSIOC, GSIOCMessage
from assemblies import AssemblyBasewithGSIOC, Network, connect_nodes
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
                       injection_port: InjectionPort,
                       flow_cell: FlowCell,
                       sample_loop: FlowCell,
                       qcmd_address: str = 'localhost',
                       qcmd_port: int = 5011,
                       name: str = '') -> None:
        
        # Devices
        self.loop_valve = loop_valve
        self.syringe_pump = syringe_pump
        self.injection_port = injection_port
        self.flow_cell = flow_cell
        self.sample_loop = sample_loop
        super().__init__([loop_valve, syringe_pump], name=name)

        # Measurement device
        self.recorder = QCMDRecorder(qcmd_address, qcmd_port, f'{self.name}.QCMDRecorder')

        # Define node connections for dead volume estimations
        self.network = Network([self.loop_valve, self.syringe_pump, self.injection_port, self.flow_cell, self.sample_loop])

        # Trigger for use in async methods
        self.trigger: asyncio.Event = asyncio.Event()

        # Dead volume queue
        self.dead_volume: asyncio.Queue = asyncio.Queue(1)

        # Measurement modes
        self.modes = {'Standby': 
                        {loop_valve: 0,
                        syringe_pump: 0,
                        'dead_volume_nodes': [injection_port.nodes[0], syringe_pump.valve.nodes[2]]},
                    'LoadLoop': 
                        {loop_valve: 1,
                        syringe_pump: 3,
                        'dead_volume_nodes': [injection_port.nodes[0], syringe_pump.valve.nodes[2]]},
                    'PumpAspirate': 
                        {loop_valve: 0,
                        syringe_pump: 1,
                        'dead_volume_nodes': [injection_port.nodes[0], syringe_pump.valve.nodes[2]]},
                    'PumpPrimeLoop': 
                        {loop_valve: 1,
                        syringe_pump: 4,
                        'dead_volume_nodes': [injection_port.nodes[0], syringe_pump.valve.nodes[2]]},
                    'PumpInject':
                        {loop_valve: 2,
                        syringe_pump: 4,
                        'dead_volume_nodes': [injection_port.nodes[0], syringe_pump.valve.nodes[2]]},
                    'LHPrime':
                        {loop_valve: 2,
                        syringe_pump: 0,
                        'dead_volume_nodes': [injection_port.nodes[0], loop_valve.valve.nodes[3]]},
                    'LHInject':
                        {loop_valve: 1,
                        syringe_pump: 0,
                        'dead_volume_nodes': [injection_port.nodes[0], loop_valve.valve.nodes[3]]},
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
        await self.loop_valve.move_valve(self.modes['PumpPrimeLoop'][self.loop_valve])

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

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.get_dead_volume('LoadLoop')

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

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.get_dead_volume('LHInject')

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

            logging.info(f'{self.name}.LHInject: Switching to Standby mode')            
            await self.change_mode('Standby')

            #logging.info(f'{self.name}.LHInject: Priming loop')

            # Prime loop
            #await self.primeloop()

async def qcmd_loop():
    gsioc = GSIOC(62, 'COM13', 19200)
    ser = HamiltonSerial(port='COM5', baudrate=38400)
    mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='loop_valve_positioner')
    sp = HamiltonSyringePump(ser, '0', SyringeLValve(4, name='syringe_LValve'), 5000., False, name='syringe_pump')
    sp.max_dispense_flow_rate = 5 * 1000 / 60
    sp.max_aspirate_flow_rate = 15 * 1000 / 60
    ip = InjectionPort('LH_injection_port')
    fc = FlowCell(0.139, 'flow_cell')
    sampleloop = FlowCell(5060., 'sample_loop')

    # connect syringe pump valve port 2 to LH injection port
    connect_nodes(ip.nodes[0], sp.valve.nodes[2], 144)

    # connect syringe pump valve port 3 to sample loop
    connect_nodes(sp.valve.nodes[3], sampleloop.inlet_node, 0.0)

    # connect sample loop to loop valve port 1
    connect_nodes(mvp.valve.nodes[1], sampleloop.outlet_node, 0.0)

    # connect cell inlet to loop valve port 2
    connect_nodes(mvp.valve.nodes[2], fc.inlet_node, 0.0)

    # connect cell outlet to loop valve port 5
    connect_nodes(mvp.valve.nodes[5], fc.outlet_node, 0.0)

    qcmd_channel = QCMDLoop(mvp, sp, ip, fc, sampleloop, name='QCMD Channel')

    #lh = SimLiquidHandler(qcmd_channel)

    try:
        await qcmd_channel.initialize()
        #await qcmd_channel.change_mode('PumpPrimeLoop')
        #await asyncio.sleep(1)
        #await qcmd_channel.primeloop(2)
        #await mvp.initialize()
        #await mvp.run_until_idle(mvp.move_valve(1))
        #await sp.initialize()
        #await sp.run_until_idle(sp.move_absolute(0, sp._speed_code(sp.max_dispense_flow_rate)))

        gsioc_task = asyncio.create_task(qcmd_channel.initialize_gsioc(gsioc))
        #await sp.run_until_idle(sp.move_absolute(0, sp._speed_code(sp.max_dispense_flow_rate)))
        #sp.max_dispense_flow_rate = 20 * 1000 / 60
        #await qcmd_channel.primeloop(1)
        #sp.max_dispense_flow_rate = 1 * 1000 / 60
        #await sp.run_until_idle(sp.smart_dispense(5000, sp.max_dispense_flow_rate))

        # run some loop inject methods sequentially
        #for i in range(4):
        #    await lh.LoopInject(200, 3, 100, f'hello{i}', 60, 60),
            #await lh.LHInject(200, 1, 100, f'hello{i}', 60, 60),

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
                            level=logging.DEBUG)

        asyncio.run(main(), debug=True)
    else:
        logging.basicConfig(
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO)
        asyncio.run(qcmd_loop(), debug=True)
        #asyncio.run(sptest())
