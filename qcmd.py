from typing import Any, Coroutine, List
import aiohttp
import asyncio
import logging
from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump, HamiltonSerial
from valve import LoopFlowValve, LValve
from components import InjectionPort, FlowCell
from gsioc import GSIOCDeviceBase, GSIOC, GSIOCMessage
from assemblies import AssemblyBasewithGSIOC, Network, connect_nodes

class GSIOCTimer(GSIOCDeviceBase):
    """Basic GSIOC Timer. When timer method is received, starts a timer for specified time,
        during which idle queries will return a busy signal. Designed for use with Gilson LH,
        which can pause until timer is no longer idle."""

    def __init__(self, gsioc: GSIOC, name='GSIOCTimer') -> None:
        super().__init__(gsioc)
        self.timer_running: asyncio.Event = asyncio.Event()
        self.name = name

    async def timer(self, wait_time: str | float = 0.0) -> None:
        """Executes timer
        """

        wait_time = float(wait_time)

        # don't start another timer if one is already running
        if not self.timer_running.is_set():
            self.timer_running.set()
            await asyncio.sleep(wait_time)
            self.timer_running.clear()
        else:
            logging.warning(f'{self.name}: Timer is already running...')

class QCMDRecorder(GSIOCTimer):
    """QCMD recording device."""

    def __init__(self, gsioc: GSIOC, qcmd_address: str = 'localhost', qcmd_port: int = 5011, name='QCMDRecorder') -> None:
        super().__init__(gsioc, name)
        self.session = aiohttp.ClientSession(f'http://{qcmd_address}:{qcmd_port}')

    async def timer(self, tag_name: str = '', record_time: str | float = 0.0, sleep_time: str | float = 0.0) -> None:
        """Executes timer and sends record command to QCMD
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # calculate total wait time
        wait_time = record_time + sleep_time

        # wait the full time
        await super().timer(wait_time)

        post_data = {'command': 'set_tag',
                     'value': {'tag': tag_name,
                               'delta_t': record_time}}

        logging.info(f'{self.session._base_url}/QCMD/ => {post_data}')

        # send an http request to QCMD server
        async with self.session.post('/QCMD/', json=post_data) as resp:
            response_json = await resp.json()
            logging.info(f'{self.session._base_url}/QCMD/ <= {response_json}')

class QCMDLoop(AssemblyBasewithGSIOC):

    def __init__(self, gsioc: GSIOC,
                       loop_valve: HamiltonValvePositioner,
                       syringe_pump: HamiltonSyringePump,
                       injection_port: InjectionPort,
                       flow_cell: FlowCell,
                       sample_loop: FlowCell,
                       qcmd_address: str = 'localhost',
                       qcmd_port: int = 5011,
                       name='') -> None:
        
        self.loop_valve = loop_valve
        self.syringe_pump = syringe_pump
        self.injection_port = injection_port
        self.flow_cell = flow_cell
        self.sample_loop = sample_loop
        self.session = aiohttp.ClientSession(f'http://{qcmd_address}:{qcmd_port}')
        self.timer_running: asyncio.Event = asyncio.Event()
        self.dead_volume = None

        super().__init__([loop_valve, syringe_pump], gsioc, name=name)
        self.network = Network([self.loop_valve, self.syringe_pump, self.injection_port, self.flow_cell, self.sample_loop])
        
        # define node connections
        # connect syringe pump valve port 2 to LH injection port
        connect_nodes(self.network._port_to_node_map[injection_port.inlet_port], syringe_pump.valve.nodes[2])

        # connect syringe pump valve port 3 to sample loop
        connect_nodes(syringe_pump.valve.nodes[3], self.network._port_to_node_map[sample_loop.inlet_port], 0.0)

        # connect sample loop to loop valve port 1
        connect_nodes(loop_valve.valve.nodes[1], self.network._port_to_node_map[sample_loop.outlet_port], 0.0)

        # connect cell inlet to loop valve port 2
        connect_nodes(loop_valve.valve.nodes[2], self.network._port_to_node_map[flow_cell.inlet_port], 0.0)

        # connect cell outlet to loop valve port 5
        connect_nodes(loop_valve.valve.nodes[5], self.network._port_to_node_map[flow_cell.outlet_port], 0.0)

        self.network.update()

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
                        syringe_pump: 1,
                        'dead_volume_nodes': [injection_port.nodes[0], syringe_pump.valve.nodes[2]]},
                    'PumpInject':
                        {loop_valve: 2,
                        syringe_pump: 2,
                        'dead_volume_nodes': [injection_port.nodes[0], syringe_pump.valve.nodes[2]]},
                    }

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:
        response = await super().handle_gsioc(data)

        # overwrites base class handling of dead volume
        if data.data == 'V':
            response = f'{self.dead_volume:0.0f}'
        
        return response

    async def timer(self, tag_name: str = '', record_time: float | str = 0.0, sleep_time: float | str = 0.0) -> None:
        """Executes timer and sends record command to QCMD
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # calculate total wait time
        wait_time = record_time + sleep_time

        # wait the full time
        if not self.timer_running.is_set():
            self.timer_running.set()
            await asyncio.sleep(wait_time)
            self.timer_running.clear()
        else:
            logging.warning(f'{self.name}: Timer is already running...')

        post_data = {'command': 'set_tag',
                     'value': {'tag': tag_name,
                               'delta_t': record_time}}

        logging.info(f'{self.session._base_url}/QCMD/ => {post_data}')

        # send an http request to QCMD server
        async with self.session.post('/QCMD/', json=post_data) as resp:
            response_json = await resp.json()
            logging.info(f'{self.session._base_url}/QCMD/ <= {response_json}')

    async def LoopInject(self,
                         pump_volume: str | float = 0, # uL
                         pump_flow_rate: str | float = 1, # mL/min
                         air_gap_plus_extra_volume: str | float = 0, #L
                         tag_name: str = '',
                         sleep_time: str | float = 0, # seconds
                         record_time: str | float = 0 # seconds
                         ) -> None:
        """LoopInject method, synchronized via GSIOC to liquid handler"""

        pump_volume = float(pump_volume)
        pump_flow_rate = float(pump_flow_rate) * 60 / 1000 # convert to uL / s
        sleep_time = float(sleep_time)
        record_time = float(record_time)

        max_flow_rate = 5000 / 1000 * 60 # uL / s

        # switch to standby mode
        await self.change_mode('Standby')

        # Set dead volume
        self.dead_volume = self.network.get_dead_volume(*self.modes['LoadLoop']['dead_volume_nodes'])
        logging.debug(f'{self.name}.LoopInject: dead volume set to {self.dead_volume}')

        # Wait for trigger to switch to LoadLoop mode
        logging.debug(f'{self.name}.LoopInject: Waiting for first trigger')
        await self.trigger.wait()
        logging.debug(f'{self.name}.LoopInject: Switching to LoadLoop mode')
        await self.change_mode('LoadLoop')
        await self.trigger.clear()

        # Wait for trigger to switch to PumpAspirate mode
        logging.debug(f'{self.name}.LoopInject: Waiting for second trigger')
        await self.trigger.wait()
        await self.change_mode('PumpAspirate')
        logging.debug(f'{self.name}.LoopInject: Switching to PumpAspirate mode')
        await self.trigger.clear()

        # aspirate a syringeful
        await self.syringe_pump.aspirate(self.syringe_pump.syringe_volume, max_flow_rate)

        # move quickly through the loop
        logging.debug(f'{self.name}.LoopInject: Switching to PumpPrimeLoop mode')
        await self.change_mode('PumpPrimeLoop')
        logging.debug(f'{self.name}.LoopInject: Moving plug through loop, total injection volume {self.sample_loop.get_volume() - pump_volume + air_gap_plus_extra_volume} uL')
        await self.syringe_pump.dispense(self.sample_loop.get_volume() - pump_volume + air_gap_plus_extra_volume, max_flow_rate)
        
        # change to inject mode
        await self.change_mode('PumpInject')
        logging.debug(f'{self.name}.LoopInject: Injecting {pump_volume} uL at flow rate {pump_flow_rate} uL / s')
        await self.syringe_pump.dispense(pump_volume, pump_flow_rate)

        # start QCMD timer
        logging.debug(f'{self.name}.LoopInject: Starting QCMD timer')
        asyncio.ensure_future(self.gsioc_command_queue.put(self.timer(tag_name, record_time, sleep_time), asyncio.get_event_loop()))

        # Prime loop
        logging.debug(f'{self.name}.LoopInject: Priming loop with full volume')
        await self.change_mode('PumpAspirate')
        await self.syringe_pump.aspirate(self.syringe_pump.syringe_volume, max_flow_rate)
        await self.change_mode('PumpPrimeLoop')
        await self.syringe_pump.dispense(self.syringe_pump.syringe_volume, max_flow_rate)

async def qcmd_loop():
    gsioc = GSIOC(62, 'COM13', 19200)
    ser = HamiltonSerial(port='COM5', baudrate=38400)
    mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='loop_valve_positioner')
    sp = HamiltonSyringePump(ser, '0', LValve(name='syringe_LValve'), 5000, False, name='syringe_pump')
    ip = InjectionPort('LH_injection_port')
    fc = FlowCell(0.444, 'flow_cell')
    sampleloop = FlowCell(5000., 'sample_loop')
    qcmd_channel = QCMDLoop(gsioc, mvp, sp, ip, fc, sampleloop, name='QCMD Channel')
    try:
        await qcmd_channel.initialize()
    finally:
        logging.info('Cleaning up...')
        qcmd_channel.session.close()

async def main():
    gsioc = GSIOC(62, 'COM13', 19200)
    qcmd_recorder = QCMDRecorder(gsioc, 'localhost', 5011)
    try:
        await qcmd_recorder.initialize()
    finally:
        logging.info('Cleaning up...')
        qcmd_recorder.session.close()

if __name__=='__main__':

    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    asyncio.run(main(), debug=True)