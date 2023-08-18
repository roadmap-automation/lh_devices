from typing import Any, Coroutine, List
import aiohttp
import asyncio
import logging
from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump, HamiltonSerial
from valve import LoopFlowValve, LValve
from components import InjectionPort, FlowCell
from gsioc import GSIOCDeviceBase, GSIOC, GSIOCMessage
from assemblies import AssemblyBasewithGSIOC, Network, connect_nodes

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
            async with self.session.post('/QCMD/', json=post_data) as resp:
                response_json = await resp.json()
                logging.info(f'{self.session._base_url}/QCMD/ <= {response_json}')

class QCMDRecorderDevice(GSIOCDeviceBase):
    """QCMD recording device."""

    def __init__(self, gsioc: GSIOC, qcmd_address: str = 'localhost', qcmd_port: int = 5011, name='QCMDRecorderDevice') -> None:
        super().__init__(gsioc, name)
        self.recorder = QCMDRecorder(qcmd_address, qcmd_port, f'{self.name}.QCMDRecorder')

    async def QCMDRecord(self, tag_name: str = '', record_time: str | float = 0.0, sleep_time: str | float = 0.0) -> None:
        """Executes timer and sends record command to QCMD. Call by sending
            {"method": "record", {**kwargs}} over GSIOC.
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # wait the full time
        await self.recorder.record(tag_name, record_time, sleep_time)

class QCMDLoop(AssemblyBasewithGSIOC):

    def __init__(self, gsioc: GSIOC,
                       loop_valve: HamiltonValvePositioner,
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
        super().__init__([loop_valve, syringe_pump], gsioc, name=name)
        self.max_flow_rate = self.syringe_pump.max_flow_rate

        # Measurement device
        self.recorder = QCMDRecorder(qcmd_address, qcmd_port, f'{self.name}.QCMDRecorder')

        # Define node connections for dead volume estimations
        self.dead_volume = None
        self.network = Network([self.loop_valve, self.syringe_pump, self.injection_port, self.flow_cell, self.sample_loop])
        
        # connect syringe pump valve port 2 to LH injection port
        connect_nodes(self.network._port_to_node_map[injection_port.inlet_port], syringe_pump.valve.nodes[2], 156)

        # connect syringe pump valve port 3 to sample loop
        connect_nodes(syringe_pump.valve.nodes[3], self.network._port_to_node_map[sample_loop.inlet_port], 0.0)

        # connect sample loop to loop valve port 1
        connect_nodes(loop_valve.valve.nodes[1], self.network._port_to_node_map[sample_loop.outlet_port], 0.0)

        # connect cell inlet to loop valve port 2
        connect_nodes(loop_valve.valve.nodes[2], self.network._port_to_node_map[flow_cell.inlet_port], 0.0)

        # connect cell outlet to loop valve port 5
        connect_nodes(loop_valve.valve.nodes[5], self.network._port_to_node_map[flow_cell.outlet_port], 0.0)

        self.network.update()

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
                    }
        
        # Control locks
        # Measurement lock indicates that a measurement is occurring and the cell should not be
        # exchanged or disturbed
        self.measurement_lock: asyncio.Lock = asyncio.Lock()

        # Channel lock indicates that the hardware in the channel is being used.
        self.channel_lock: asyncio.Lock = asyncio.Lock()

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:

        # overwrites base class handling of dead volume
        if data.data == 'V':
            response = f'{self.dead_volume:0.0f}'
        else:
            response = await super().handle_gsioc(data)
        
        return response
    
    async def QCMDRecord(self,
                         tag_name: str = '',
                         sleep_time: str | float = 0, # seconds
                         record_time: str | float = 0 # seconds
                        ) -> None:
        """QCMDRecord method. Final action is marking task as done,
            thereby releasing GSIOC communications."""

        sleep_time = float(sleep_time)
        record_time = float(record_time)

        # Locks the measurement system if not already locked. This (rather than "with lock")
        # allows the lock to be passed smoothly from a calling function without interruption,
        # or the lock to be acquired if function is used in a standalone fashion.
        if not self.measurement_lock.locked():
            self.measurement_lock.acquire()

        await self.recorder.record(tag_name, record_time, sleep_time)

        self.measurement_lock.release()

        self.gsioc_command_queue.task_done()

    async def primeloop(self,
                        n_prime: int = 1 # number of repeats
                         ) -> None:
        """subroutine for priming the loop method. Primes the loop, but does not do anything with locks"""

        for _ in range(n_prime):
            logging.debug(f'{self.name}.LoopInject: Priming loop with full volume')
            
            # if syringe is not already at 0, move to prime mode and home the syringe
            await self.syringe_pump.get_syringe_position()
            if self.syringe_pump.syringe_position != 0:
                await self.change_mode('PumpPrimeLoop')
                await self.syringe_pump.run_until_idle(self.syringe_pump.home())
            
            # aspirate a full pump volume
            await self.change_mode('PumpAspirate')
            await self.syringe_pump.run_until_idle(self.syringe_pump.aspirate(self.syringe_pump.syringe_volume, self.syringe_pump.max_flow_rate))

            # dispense a full pump volume
            await self.change_mode('PumpPrimeLoop')
            await self.syringe_pump.run_until_idle(self.syringe_pump.dispense(self.syringe_pump.syringe_volume, self.syringe_pump.max_flow_rate))

    async def PrimeLoop(self,
                        n_prime: int | str = 1 # number of repeats
                         ) -> None:
        """PrimeLoop standalone method"""

        n_prime = int(n_prime)

        with self.channel_lock:
            await self.primeloop(n_prime)

    async def LoopInject(self,
                         pump_volume: str | float = 0, # uL
                         pump_flow_rate: str | float = 1, # mL/min
                         air_gap_plus_extra_volume: str | float = 0, #uL
                         tag_name: str = '',
                         sleep_time: str | float = 0, # seconds
                         record_time: str | float = 0 # seconds
                         ) -> None:
        """LoopInject method, synchronized via GSIOC to liquid handler"""

        pump_volume = float(pump_volume)
        pump_flow_rate = float(pump_flow_rate) * 1000 / 60 # convert to uL / s
        sleep_time = float(sleep_time)
        record_time = float(record_time)

        # locks the channel so any additional calling processes have to wait
        with self.channel_lock:

            # switch to standby mode
            await self.change_mode('Standby')

            # Set dead volume
            self.dead_volume = self.network.get_dead_volume(*(node.base_port for node in self.modes['LoadLoop']['dead_volume_nodes']))
            logging.debug(f'{self.name}.LoopInject: dead volume set to {self.dead_volume}')

            # Wait for trigger to switch to LoadLoop mode
            logging.debug(f'{self.name}.LoopInject: Waiting for first trigger')
            await self.trigger.wait()
            logging.debug(f'{self.name}.LoopInject: Switching to LoadLoop mode')
            await self.change_mode('LoadLoop')
            self.trigger.clear()

            # Wait for trigger to switch to PumpAspirate mode
            logging.debug(f'{self.name}.LoopInject: Waiting for second trigger')
            await self.trigger.wait()
            await self.change_mode('PumpAspirate')
            logging.debug(f'{self.name}.LoopInject: Switching to PumpAspirate mode')
            self.trigger.clear()

            # At this point, this method is done with GSIOC; signal command queue
            self.gsioc_command_queue.task_done()

            # aspirate a syringeful
            total_aspiration_volume = self.sample_loop.get_volume() - air_gap_plus_extra_volume
            logging.debug(f'{self.name}.LoopInject: Aspirating {total_aspiration_volume} uL')
            await self.syringe_pump.run_until_idle(self.syringe_pump.aspirate(total_aspiration_volume, self.syringe_pump.max_flow_rate))

            # move quickly through the loop
            logging.debug(f'{self.name}.LoopInject: Switching to PumpPrimeLoop mode')
            await self.change_mode('PumpPrimeLoop')
            logging.debug(f'{self.name}.LoopInject: Moving plug through loop, total injection volume {self.sample_loop.get_volume() - pump_volume + air_gap_plus_extra_volume} uL')
            await self.syringe_pump.run_until_idle(self.syringe_pump.dispense(self.sample_loop.get_volume() - (pump_volume + air_gap_plus_extra_volume), self.syringe_pump.max_flow_rate))
            
            # waits until any current measurements are complete. Note that this could be done with
            # with measurement_lock but then QCMDRecord would have to grab the lock as soon as it
            # was released. This allows QCMDRecord to release the lock when it is done.
            self.measurement_lock.acquire()

            # change to inject mode
            await self.change_mode('PumpInject')
            logging.debug(f'{self.name}.LoopInject: Injecting {pump_volume} uL at flow rate {pump_flow_rate} uL / s')
            await self.syringe_pump.run_until_idle(self.syringe_pump.dispense(pump_volume, pump_flow_rate))

            # start QCMD timer
            logging.debug(f'{self.name}.LoopInject: Starting QCMD timer')
            await self.gsioc_command_queue.put(self.QCMDRecord(tag_name, sleep_time, record_time))

            # Prime loop
            await self.primeloop()

async def qcmd_loop():
    gsioc = GSIOC(62, 'COM4', 19200)
    ser = HamiltonSerial(port='COM5', baudrate=38400)
    mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='loop_valve_positioner')
    sp = HamiltonSyringePump(ser, '0', LValve(4, name='syringe_LValve'), 5000, False, name='syringe_pump')
    sp.max_flow_rate = 5000 * 60 / 1000
    ip = InjectionPort('LH_injection_port')
    fc = FlowCell(0.444, 'flow_cell')
    sampleloop = FlowCell(5000., 'sample_loop')
    qcmd_channel = QCMDLoop(gsioc, mvp, sp, ip, fc, sampleloop, name='QCMD Channel')

    async def sim_gsioc_commands(dev: AssemblyBasewithGSIOC, method_name: str, kwargs):
        await asyncio.sleep(1)
        logging.debug(f'{dev.name}: Method {method_name} requested')
        if hasattr(dev, method_name):
            logging.debug(f'{dev.name}: Starting method {method_name} with kwargs {kwargs}')
            method = getattr(dev, method_name)

        await dev.gsioc_command_queue.put(method(**kwargs))
        await dev.gsioc_command_queue.join()
        logging.info(f'sim_commands: Queue is finally empty!')

    async def set_trigger(trigger: asyncio.Event, sleep_time: float = 0.0):
        await asyncio.sleep(sleep_time)
        trigger.set()
        trigger.clear()

    try:
        await qcmd_channel.initialize_devices()
        gsioc_task = asyncio.create_task(qcmd_channel.initialize_gsioc())
        await asyncio.gather(sim_gsioc_commands(qcmd_channel,
                                                'LoopInject',
                                                {'tag_name': 'hello',
                                                 'record_time': 10,
                                                 'sleep_time': 10,
                                                 'pump_volume': 1000,
                                                 'pump_flow_rate': 3,
                                                 'air_gap_plus_extra_volume': 100}),
                            set_trigger(qcmd_channel.trigger, 20),
                            set_trigger(qcmd_channel.trigger, 30))
        await asyncio.gather(sim_gsioc_commands(qcmd_channel,
                                                'QCMDRecord',
                                                {'tag_name': 'hello',
                                                 'record_time': 10,
                                                 'sleep_time': 10,}))
        await gsioc_task
    finally:
        logging.info('Cleaning up...')
        await qcmd_channel.recorder.session.close()

async def main():
    gsioc = GSIOC(62, 'COM13', 19200)
    qcmd_recorder = QCMDRecorderDevice(gsioc, 'localhost', 5011)
    try:
        await qcmd_recorder.initialize()
    finally:
        await qcmd_recorder.recorder.session.close()

if __name__=='__main__':

    import datetime

    logging.basicConfig(handlers=[
                            logging.FileHandler(datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_qcmd_recorder_log.txt'),
                            logging.StreamHandler()
                        ],
                        format='%(asctime)s.%(msecs)03d %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    asyncio.run(main(), debug=True)