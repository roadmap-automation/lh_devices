import asyncio
import datetime
import logging
import pathlib

from aiohttp.web_app import Application as Application

from ..distribution import DistributionSingleValve
from ..hamilton.HamiltonDevice import HamiltonValvePositioner, HamiltonSyringePump, SMDSensoronHamiltonDevice
from ..hamilton.HamiltonComm import HamiltonSerial
from ..valve import LoopFlowValve, DistributionValve, SyringeYValve
from ..webview import run_socket_app
from ..gilson.gsioc import GSIOC
from ..components import InjectionPort, FlowCell
from ..connections import connect_nodes
from ..waste import RoadmapWasteInterface
from .rinsesystem import RinseSystem

LOG_PATH = pathlib.Path(__file__).parent.parent.parent / 'logs'
HISTORY_PATH = pathlib.Path(__file__).parent.parent.parent / 'history'

TMP_PATH = pathlib.Path('~/Documents/tmp').expanduser() / 'rinse_layout.json'

async def run_rinse_system():
    """Assumes the following setup:

        o Syringe pump with a mounted Y-valve.
        o "Selector" valve positioner with an 8-port distribution valve. Each of the ports except the common port and port 8 is 
            connected to a solvent bottle. Port 8 is connected to waste for priming
        o "Source" valve positioner with a 4-port distribution valve and the following connections:
            - common port (0) connected via a sample loop (5 mL) to syringe pump Y-valve outlet
            - port 1 connected to the common port (0) of the selector valve
            - port 2 connected to a source of air
            - port 3 connected to a direct injection distribution valve
            - port 4 connected to a loop injection distribution valve
    """


    # serial communications setup
    ser = HamiltonSerial(port='COM6', baudrate=38400)

    # device setup
    selector_valve = HamiltonValvePositioner(ser, '2', DistributionValve(8, name='selector_valve'), name='Selector Valve')
    source_valve = HamiltonValvePositioner(ser, '1', DistributionValve(4, name='source_valve'), name='Source Valve')
    syringe_pump = HamiltonSyringePump(ser, '0', SyringeYValve(name='syringe_YValve'), 5000., False, name='Syringe Pump')

    for sp in [syringe_pump]:
        sp.max_dispense_flow_rate = 5 * 1000 / 60
        sp.max_aspirate_flow_rate = 15 * 1000 / 60
    
    rinse_loop = FlowCell(5000., 'rinse_loop')

    # connect loop to syringe pump and selector valve
    connect_nodes(rinse_loop.inlet_node, syringe_pump.valve.nodes[2], 0.0)
    connect_nodes(rinse_loop.outlet_node, source_valve.valve.nodes[0], 0.0)

    # connect selector and source valves
    connect_nodes(selector_valve.valve.nodes[0], source_valve.valve.nodes[0], 50.0)

    waste_tracker = RoadmapWasteInterface('http://localhost:5001/Waste/AddWaste/')

    rinse_system = RinseSystem(syringe_pump, source_valve, selector_valve, rinse_loop, source_valve.valve.nodes[3], TMP_PATH, waste_tracker=waste_tracker, name='Rinse System')

    app = rinse_system.create_web_app(template='roadmap.html')
    runner = await run_socket_app(app, 'localhost', 5004)

    try:
        await rinse_system.initialize()
        await asyncio.Event().wait()

    finally:
        logging.info('Closing Rinse System...')
        asyncio.gather(
                    runner.cleanup())

if __name__=='__main__':

    logging.basicConfig(handlers=[
                        logging.FileHandler(LOG_PATH / (datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_rinse_log.txt')),
                        logging.StreamHandler()
                    ],
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

    asyncio.run(run_rinse_system(), debug=True)
