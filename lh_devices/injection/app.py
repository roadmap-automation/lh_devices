import asyncio
import datetime
import logging
import pathlib

from aiohttp.web_app import Application as Application

from ..distribution import DistributionSingleValve
from ..hamilton.HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump, SMDSensoronHamiltonDevice
from ..hamilton.HamiltonComm import HamiltonSerial
from ..valve import LoopFlowValve, DistributionValve, SyringeLValve
from ..webview import run_socket_app
from ..gilson.gsioc import GSIOC
from ..components import InjectionPort, FlowCell
from ..connections import connect_nodes
from .injectionsystem import RoadmapChannelBubbleSensor, RoadmapChannelAssembly

LOG_PATH = pathlib.Path(__file__).parent.parent.parent / 'logs'
HISTORY_PATH = pathlib.Path(__file__).parent.parent.parent / 'history'

async def run_injection_system():
    # serial communications setup
    gsioc = GSIOC(62, 'COM13', 19200)
    ser = HamiltonSerial(port='COM9', baudrate=38400)

    # device setup
    dvp = HamiltonValvePositioner(ser, '2', DistributionValve(8, name='distribution_valve'), name='Distribution Valve')

    mvp0 = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve0'), name='Loop Valve 0')
    outlet_bubble_sensor0 = SMDSensoronHamiltonDevice(mvp0, 2, 1)
    inlet_bubble_sensor0 = SMDSensoronHamiltonDevice(mvp0, 1, 0)
    sp0 = HamiltonSyringePump(ser, '0', SyringeLValve(4, name='syringe_LValve0'), 5000., False, name='Syringe Pump 0')

    mvp1 = HamiltonValvePositioner(ser, '4', LoopFlowValve(6, name='loop_valve1'), name='Loop Valve 1')
    outlet_bubble_sensor1 = SMDSensoronHamiltonDevice(mvp1, 2, 1)
    inlet_bubble_sensor1 = SMDSensoronHamiltonDevice(mvp1, 1, 0)
    sp1 = HamiltonSyringePump(ser, '3', SyringeLValve(4, name='syringe_LValve1'), 5000., False, name='Syringe Pump 1')

    mvp2 = HamiltonValvePositioner(ser, '6', LoopFlowValve(6, name='loop_valve2'), name='Loop Valve 2')
    outlet_bubble_sensor2 = SMDSensoronHamiltonDevice(mvp2, 2, 1)
    inlet_bubble_sensor2 = SMDSensoronHamiltonDevice(mvp2, 1, 0)
    sp2 = HamiltonSyringePump(ser, '5', SyringeLValve(4, name='syringe_LValve2'), 5000., False, name='Syringe Pump 2')

    for sp in [sp0, sp1, sp2]:
        sp.max_dispense_flow_rate = 5 * 1000 / 60
        sp.max_aspirate_flow_rate = 15 * 1000 / 60
    
    ip = InjectionPort('LH_injection_port')

    fc0 = FlowCell(139, 'flow_cell0')
    fc1 = FlowCell(139, 'flow_cell1')
    fc2 = FlowCell(139, 'flow_cell2')

    sampleloop0 = FlowCell(5060., 'sample_loop0')
    sampleloop1 = FlowCell(5060., 'sample_loop1')
    sampleloop2 = FlowCell(5000., 'sample_loop2')

    channel_0 = RoadmapChannelBubbleSensor(mvp0, sp0, fc0, sampleloop0, injection_node=ip.nodes[0], inlet_bubble_sensor=inlet_bubble_sensor0, outlet_bubble_sensor=outlet_bubble_sensor0, name='Channel 0')
    channel_1 = RoadmapChannelBubbleSensor(mvp1, sp1, fc1, sampleloop1, injection_node=ip.nodes[0], inlet_bubble_sensor=inlet_bubble_sensor1, outlet_bubble_sensor=outlet_bubble_sensor1, name='Channel 1')
    channel_2 = RoadmapChannelBubbleSensor(mvp2, sp2, fc2, sampleloop2, injection_node=ip.nodes[0], inlet_bubble_sensor=inlet_bubble_sensor2, outlet_bubble_sensor=outlet_bubble_sensor2, name='Channel 2')

    distribution_system = DistributionSingleValve(dvp, ip, 'Distribution System')

    # connect LH injection port to distribution port valve 0
    connect_nodes(ip.nodes[0], dvp.valve.nodes[0], 262 + 20)

    # loop inject: connect distribution valve port 1 to syringe pump valve node 2 (top)
    connect_nodes(dvp.valve.nodes[1], sp0.valve.nodes[2], 73 + 20)
    connect_nodes(dvp.valve.nodes[3], sp1.valve.nodes[2], 90 + 20)
    connect_nodes(dvp.valve.nodes[5], sp2.valve.nodes[2], 90 + 20)

    # direct inject: connect distribution valve port 2 to loop valve node 3 (top right)
    # volume ~ tubing volume + 60 uL for bubble sensor
    connect_nodes(dvp.valve.nodes[2], mvp0.valve.nodes[3], 100)
    connect_nodes(dvp.valve.nodes[4], mvp1.valve.nodes[3], 180)
    connect_nodes(dvp.valve.nodes[6], mvp2.valve.nodes[3], 180)

    # connect syringe pump valve port 3 to sample loop
    connect_nodes(sp0.valve.nodes[3], sampleloop0.inlet_node, 0.0)
    connect_nodes(sp1.valve.nodes[3], sampleloop1.inlet_node, 0.0)
    connect_nodes(sp2.valve.nodes[3], sampleloop2.inlet_node, 0.0)

    # connect sample loop to loop valve port 1
    connect_nodes(mvp0.valve.nodes[1], sampleloop0.outlet_node, 0.0)
    connect_nodes(mvp1.valve.nodes[1], sampleloop1.outlet_node, 0.0)
    connect_nodes(mvp2.valve.nodes[1], sampleloop2.outlet_node, 0.0)

    # connect cell inlet to loop valve port 2
    connect_nodes(mvp0.valve.nodes[2], fc0.inlet_node, 0.0)
    connect_nodes(mvp1.valve.nodes[2], fc1.inlet_node, 0.0)
    connect_nodes(mvp2.valve.nodes[2], fc2.inlet_node, 0.0)

    # connect cell outlet to loop valve port 5
    connect_nodes(mvp0.valve.nodes[5], fc0.outlet_node, 0.0)
    connect_nodes(mvp1.valve.nodes[5], fc1.outlet_node, 0.0)
    connect_nodes(mvp2.valve.nodes[5], fc2.outlet_node, 0.0)

    qcmd_system = RoadmapChannelAssembly([channel_0, channel_1, channel_2],
                                            distribution_system=distribution_system,
                                            gsioc=gsioc,
                                            database_path=HISTORY_PATH / 'injection_system.db',
                                            name='MultiChannel Injection System')
    
    app = qcmd_system.create_web_app(template='roadmap.html')
    runner = await run_socket_app(app, 'localhost', 5003)

    try:
        await qcmd_system.initialize()
        gsioc_task = asyncio.create_task(gsioc.listen())
        await asyncio.Event().wait()

    finally:
        logging.info('Closing Multichannel Injection System...')
        gsioc_task.cancel()
        asyncio.gather(
                    runner.cleanup())

if __name__=='__main__':

    logging.basicConfig(handlers=[
                        logging.FileHandler(LOG_PATH / (datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_injection_log.txt')),
                        logging.StreamHandler()
                    ],
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

    asyncio.run(run_injection_system(), debug=True)
