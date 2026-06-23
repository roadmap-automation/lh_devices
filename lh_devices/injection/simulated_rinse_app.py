import asyncio
import datetime
import logging
import pathlib

from ..distribution import DistributionSingleValveTwoSource
from ..hamilton.HamiltonDevice import SimulatedHamiltonValvePositioner, SimulatedHamiltonSyringePump, SimulatedSensoronHamiltonDevice
from ..valve import LoopFlowValve, DistributionValve, SyringeLValve, SyringeYValve, YValve
from ..webview import run_socket_app
from ..components import InjectionPort, FlowCell
from ..connections import connect_nodes
from ..broker_plugin import BrokerWasteInterface, DeviceBrokerWorker
from .channel import RoadmapChannelBubbleSensor
from .injectionsystem import RoadmapChannelAssemblyRinse
from ..rinse.rinsesystem import RinseSystem

DEVICE_ID_INJECTION = 'injection'
DEVICE_ID_RINSE = 'rinse'
DEVICE_ID_DISTRIBUTION = 'distribution'
LOG_PATH = pathlib.Path(__file__).parent.parent.parent / 'logs'
HISTORY_PATH = pathlib.Path(__file__).parent.parent.parent / 'history'

async def run_injection_system():

    # ============== Rinse System setup =========================

    selector_valve = SimulatedHamiltonValvePositioner(DistributionValve(8, name='selector_valve'), name='Selector Valve')
    source_valve = SimulatedHamiltonValvePositioner(DistributionValve(4, name='source_valve'), name='Source Valve')
    syringe_pump = SimulatedHamiltonSyringePump(SyringeYValve(name='syringe_YValve'), 5000., False, name='Syringe Pump')

    syringe_pump.max_dispense_flow_rate = 5 * 1000 / 60
    syringe_pump.max_aspirate_flow_rate = 15 * 1000 / 60

    rinse_loop = FlowCell(5000., 'rinse_loop')

    rinse_ip = InjectionPort('loop_injection_port')
    rinse_ip.injection_port = source_valve.valve.ports[3]
    rinse_ip._generate_nodes()

    connect_nodes(rinse_loop.inlet_node, syringe_pump.valve.nodes[2], 0.0)
    connect_nodes(rinse_loop.outlet_node, source_valve.valve.nodes[0], 0.0)
    connect_nodes(selector_valve.valve.nodes[0], source_valve.valve.nodes[4], 265.0)

    waste_tracker = BrokerWasteInterface()

    rinse_system = RinseSystem(syringe_pump=syringe_pump,
                               source_valve=source_valve,
                               selector_valve=selector_valve,
                               rinse_loop=rinse_loop,
                               loop_injection_port=rinse_ip,
                               direct_injection_port=rinse_ip,
                               layout_path=LOG_PATH / 'rinse_layout.json',
                               database_path=HISTORY_PATH / 'rinse_system.db',
                               waste_tracker=waste_tracker,
                               name='Rinse System',
                               id='rinse_system')

    rinseapp = rinse_system.create_web_app(template='roadmap.html')
    rinse_runner = await run_socket_app(rinseapp, 'localhost', 5014)

    # ============== Distribution System setup ==================

    ip = InjectionPort('LH_injection_port')

    dvp_source = SimulatedHamiltonValvePositioner(YValve(name='source_valve'), name='Distribution Source Valve')
    dvp_selection = SimulatedHamiltonValvePositioner(DistributionValve(8, name='distribution_valve'), name='Distribution Selection Valve')

    distribution_system = DistributionSingleValveTwoSource(source_valve=dvp_source,
                                                           distribution_valve=dvp_selection,
                                                           injection_port=ip,
                                                           name='Distribution System',
                                                           id='distribution_system')

    distribution_app = distribution_system.create_web_app(template='roadmap.html')
    distribution_runner = await run_socket_app(distribution_app, 'localhost', 5002)

    # ============== Injection System setup =====================

    mvp0 = SimulatedHamiltonValvePositioner(LoopFlowValve(6, name='loop_valve0'), name='Loop Valve 0')
    outlet_bubble_sensor0 = SimulatedSensoronHamiltonDevice(mvp0, 2, 1)
    inlet_bubble_sensor0 = SimulatedSensoronHamiltonDevice(mvp0, 1, 0)
    sp0 = SimulatedHamiltonSyringePump(SyringeLValve(4, name='syringe_LValve0'), 5000., False, name='Syringe Pump 0')

    mvp1 = SimulatedHamiltonValvePositioner(LoopFlowValve(6, name='loop_valve1'), name='Loop Valve 1')
    outlet_bubble_sensor1 = SimulatedSensoronHamiltonDevice(mvp1, 2, 1)
    inlet_bubble_sensor1 = SimulatedSensoronHamiltonDevice(mvp1, 1, 0)
    sp1 = SimulatedHamiltonSyringePump(SyringeLValve(4, name='syringe_LValve1'), 5000., False, name='Syringe Pump 1')

    mvp2 = SimulatedHamiltonValvePositioner(LoopFlowValve(6, name='loop_valve2'), name='Loop Valve 2')
    outlet_bubble_sensor2 = SimulatedSensoronHamiltonDevice(mvp2, 2, 1)
    inlet_bubble_sensor2 = SimulatedSensoronHamiltonDevice(mvp2, 1, 0)
    sp2 = SimulatedHamiltonSyringePump(SyringeLValve(4, name='syringe_LValve2'), 5000., False, name='Syringe Pump 2')

    for sp in [sp0, sp1, sp2]:
        sp.max_dispense_flow_rate = 5 * 1000 / 60
        sp.max_aspirate_flow_rate = 15 * 1000 / 60

    fc0 = FlowCell(139, 'flow_cell0')
    fc1 = FlowCell(139, 'flow_cell1')
    fc2 = FlowCell(139, 'flow_cell2')

    sampleloop0 = FlowCell(5060., 'Injection Loop 0')
    sampleloop1 = FlowCell(5060., 'Injection Loop 1')
    sampleloop2 = FlowCell(5000., 'Injection Loop 2')

    channel_0 = RoadmapChannelBubbleSensor(mvp0, sp0, fc0, sampleloop0, injection_node=ip.nodes[0], inlet_bubble_sensor=inlet_bubble_sensor0, outlet_bubble_sensor=outlet_bubble_sensor0, name='Channel 0')
    channel_1 = RoadmapChannelBubbleSensor(mvp1, sp1, fc1, sampleloop1, injection_node=ip.nodes[0], inlet_bubble_sensor=inlet_bubble_sensor1, outlet_bubble_sensor=outlet_bubble_sensor1, name='Channel 1')
    channel_2 = RoadmapChannelBubbleSensor(mvp2, sp2, fc2, sampleloop2, injection_node=ip.nodes[0], inlet_bubble_sensor=inlet_bubble_sensor2, outlet_bubble_sensor=outlet_bubble_sensor2, name='Channel 2')

    # internal distribution system connection
    connect_nodes(dvp_source.valve.nodes[0], dvp_selection.valve.nodes[0], 80 + 20)

    # connect LH and rinse system to distribution system
    connect_nodes(ip.nodes[0], dvp_source.valve.nodes[1], 262 + 20)
    connect_nodes(rinse_ip.nodes[0], dvp_source.valve.nodes[2], 242 + 20)

    # loop inject: connect distribution valve port 1 to syringe pump valve node 2 (top)
    connect_nodes(dvp_selection.valve.nodes[1], sp0.valve.nodes[2], 73 + 20)
    connect_nodes(dvp_selection.valve.nodes[3], sp1.valve.nodes[2], 90 + 20)
    connect_nodes(dvp_selection.valve.nodes[5], sp2.valve.nodes[2], 90 + 20)

    # direct inject: connect distribution valve port 2 to loop valve node 3 (top right)
    connect_nodes(dvp_selection.valve.nodes[2], mvp0.valve.nodes[3], 120)
    connect_nodes(dvp_selection.valve.nodes[4], mvp1.valve.nodes[3], 200)
    connect_nodes(dvp_selection.valve.nodes[6], mvp2.valve.nodes[3], 200)

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

    qcmd_system = RoadmapChannelAssemblyRinse([channel_0, channel_1, channel_2],
                                            distribution_system=distribution_system,
                                            rinse_system=rinse_system,
                                            layout_path=LOG_PATH / 'injection_layout.json',
                                            database_path=HISTORY_PATH / 'injection_system.db',
                                            waste_tracker=waste_tracker,
                                            name='MultiChannel Injection System')

    injection_worker = DeviceBrokerWorker(DEVICE_ID_INJECTION, qcmd_system, local_port=5003, device_type=DEVICE_ID_INJECTION, display_name='Multichannel Injection System', allow_sample_mixing=True)
    injection_worker.waste_interface = waste_tracker
    rinse_worker = DeviceBrokerWorker(DEVICE_ID_RINSE, rinse_system, local_port=5014, device_type=DEVICE_ID_RINSE, display_name='Rinse System', allow_sample_mixing=False)
    rinse_worker.waste_interface = waste_tracker
    distribution_worker = DeviceBrokerWorker(DEVICE_ID_DISTRIBUTION, distribution_system, local_port=5002, device_type=DEVICE_ID_DISTRIBUTION, display_name='Distribution System', allow_sample_mixing=False)

    app = qcmd_system.create_web_app(template='roadmap.html')
    runner = await run_socket_app(app, 'localhost', 5003)

    try:
        await qcmd_system.initialize()

        await asyncio.gather(
            injection_worker.start(),
            rinse_worker.start(),
            distribution_worker.start(),
        )
        await asyncio.Event().wait()

    finally:
        logging.info('Closing Multichannel Injection System...')
        asyncio.gather(
                    runner.cleanup(),
                    rinse_runner.cleanup(),
                    distribution_runner.cleanup())

if __name__ == '__main__':

    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    logging.basicConfig(handlers=[
                        logging.FileHandler(LOG_PATH / (datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_injection_log.txt')),
                        logging.StreamHandler()
                    ],
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

    asyncio.run(run_injection_system(), debug=True)
