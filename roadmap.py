import logging
from typing import List

from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump
from gsioc import GSIOC
from components import InjectionPort, FlowCell
from assemblies import AssemblyBase, AssemblyBasewithGSIOC, Network
from connections import connect_nodes

class LoopInjectAssembly(AssemblyBase):
    """Basic loop assembly system with one syringe pump and one valve positioner.
    """

    def __init__(self, loop_valve: HamiltonValvePositioner, syringe_pump: HamiltonSyringePump, injection_port: InjectionPort, flow_cell: FlowCell, sample_loop: FlowCell, name='') -> None:
        self.loop_valve = loop_valve
        self.syringe_pump = syringe_pump
        self.injection_port = injection_port
        self.flow_cell = flow_cell
        self.sample_loop = sample_loop

        super().__init__([loop_valve, syringe_pump], name=name)
        self.network = Network([self.loop_valve, self.syringe_pump, self.injection_port, self.flow_cell, self.sample_loop])
        
        # define node connections
        # connect loop valve port 0 to flow cell inlet
        connect_nodes(loop_valve.valve.nodes[0], self.network._port_to_node_map[flow_cell.inlet_port], 58.)

        # connect loop valve port 1 to syringe pump Y valve outlet
        connect_nodes(loop_valve.valve.nodes[1], syringe_pump.valve.nodes[1], 61.)

        # connect loop valve port 2 to loop valve port 5 via 5.5 mL loop
        connect_nodes(loop_valve.valve.nodes[2], self.network._port_to_node_map[sample_loop.inlet_port], 0.0)
        connect_nodes(loop_valve.valve.nodes[5], self.network._port_to_node_map[sample_loop.outlet_port], 0.0)

        # connect loop valve port 3 to waste
        # connect loop valve port 4 to LH injection port
        connect_nodes(loop_valve.valve.nodes[4], self.network._port_to_node_map[injection_port.inlet_port], 313.)

        self.modes = {'LoadLoop': 
                        {loop_valve: 1,
                        syringe_pump: 1,
                        'dead_volume_nodes': [injection_port.nodes[0], loop_valve.valve.nodes[5]]},
                    'PumpInject':
                        {loop_valve: 2,
                        syringe_pump: 2,
                        'dead_volume_nodes': [injection_port.nodes[0], loop_valve.valve.nodes[3]]},
                    'PumpAspirate':
                        {loop_valve: 2,
                        syringe_pump: 1,
                        'dead_volume_nodes': [injection_port.nodes[0], loop_valve.valve.nodes[3]]}
                    }

class MultiChannelAssembly(AssemblyBasewithGSIOC):

    def __init__(self, channels: List[LoopInjectAssembly], distribution_valve: HamiltonValvePositioner, injection_port: InjectionPort, name='') -> None:
        super().__init__([dev for ch in channels for dev in ch.devices], name)

    """TODO:
        1. Make ROADMAP channels akin to QCMD channels
            a. Should know about distribution valve all the way up to injection port (define all
                of these and their connections before connecting them into channels)
            b. Mode definitions should include distribution valve positions, probably
            c. ROADMAP channels should also have a change_direction method that switches the injection
            direction. This can be done once at the beginning of inject methods.
        2. Collect ROADMAP channels into MultiChannel Assembly
            a. handle_gsioc should pass channel-specific commands to those channels and route responses back
            b. implement distribution valve lock so only one channel can use it at a time (or is this
                a channel-level function?)

        """


if __name__=='__main__':

    import asyncio
    from HamiltonComm import HamiltonSerial
    from valve import LoopFlowValve, SyringeYValve

    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)

    if True:
        async def main():
            ser = HamiltonSerial(port='COM5', baudrate=38400)
            mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='loop_valve_positioner')
            sp = HamiltonSyringePump(ser, '0', SyringeYValve(name='syringe_y_valve'), 5000, False, name='syringe_pump')
            ip = InjectionPort('LH_injection_port')
            fc = FlowCell(0.444, 'flow_cell')
            sampleloop = FlowCell(5500., 'sample_loop')
            at = LoopInjectAssembly(loop_valve=mvp, syringe_pump=sp, injection_port=ip, flow_cell=fc, sample_loop=sampleloop, name='LoopInject0')
            
            await at.initialize()
            await asyncio.sleep(3)
            await at.change_mode('PumpInject')
            logging.debug(at.get_dead_volume())
            await asyncio.sleep(3)
            await at.change_mode('LoadLoop')
            logging.debug(at.get_dead_volume())
            await asyncio.sleep(3)
            await at.change_mode('PumpAspirate')
            logging.debug(at.get_dead_volume())
            await asyncio.sleep(3)
            

            #mvp.valve.move(2)
            #logging.debug(mvp.valve.nodes[4].connections, mvp.valve.nodes[5].connections)
            #logging.debug(at.network.get_dead_volume(ip.inlet_port, mvp.valve.nodes[2].base_port))
            #sp.valve.move(2)
            #logging.debug(at.network.get_dead_volume(sp.valve.nodes[0].base_port, sp.valve.nodes[1].base_port))

        asyncio.run(main(), debug=True)
    else:
        ser = HamiltonSerial(port='COM3', baudrate=38400)
        mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='loop_valve_positioner')
        sp = HamiltonSyringePump(ser, '0', SyringeYValve(name='syringe_y_valve'), 5000, False, name='syringe_pump')
        ip = InjectionPort('LH_injection_port')
        fc = FlowCell(444., 'flow_cell')
        sampleloop = FlowCell(5500., 'sample_loop')
        at = LoopInjectAssembly(loop_valve=mvp, syringe_pump=sp, injection_port=ip, flow_cell=fc, sample_loop=sampleloop, name='LoopInject0')
        mvp.valve.move(1)
        logging.debug(mvp.valve.nodes[4].connections, mvp.valve.nodes[5].connections)
        logging.debug(at.network.get_dead_volume(ip.inlet_port, mvp.valve.nodes[3].base_port))
        #sp.valve.move(2)
        #logging.debug(at.network.get_dead_volume(sp.valve.nodes[0].base_port, sp.valve.nodes[1].base_port))