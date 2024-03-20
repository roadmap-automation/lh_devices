import logging
from typing import Coroutine, List, Dict
from dataclasses import dataclass

from assemblies import Mode
from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump
from gsioc import GSIOC
from components import InjectionPort, FlowCell
from assemblies import AssemblyBase, AssemblyBasewithGSIOC, Network
from connections import connect_nodes, Node
from methods import MethodBase

class RoadmapChannelBase(AssemblyBasewithGSIOC):

    def __init__(self, loop_valve: HamiltonValvePositioner,
                       syringe_pump: HamiltonSyringePump,
                       flow_cell: FlowCell,
                       sample_loop: FlowCell,
                       injection_node: Node | None = None,
                       name: str = '') -> None:
        
        # Devices
        self.loop_valve = loop_valve
        self.syringe_pump = syringe_pump
        self.flow_cell = flow_cell
        self.sample_loop = sample_loop
        self.injection_node = injection_node
        super().__init__([loop_valve, syringe_pump], name=name)
        self.methods: Dict[str, MethodBase] = {}

        # Define node connections for dead volume estimations
        self.network = Network(self.devices + [self.flow_cell, self.sample_loop])

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

    def get_dead_volume(self, mode: str | None = None) -> float:
        return super().get_dead_volume(self.injection_node, mode)
    
    def run_method(self, method_name: str, method_kwargs: dict) -> None:
        return super().run_method(self.methods[method_name].run(**method_kwargs))

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

class LoadLoop(MethodBase):
    """Loads the loop of one ROADMAP channel
    """

    def __init__(self, channel: RoadmapChannelBase) -> None:
        super().__init__([channel.syringe_pump, channel.loop_valve])
        self.channel = channel
        self.dead_volume_mode: str = self.MethodDefinition.name

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "LoadLoop"
        pump_volume: str | float = 0, # uL
        excess_volume: str | float = 0, #uL

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        method = self.MethodDefinition(**kwargs)

        pump_volume = float(method.pump_volume)
        excess_volume = float(method.excess_volume)

        # Wait for trigger to switch to LoadLoop mode
        logging.info(f'{method.name}: Waiting for first trigger')
        await self.channel.wait_for_trigger()
        logging.info(f'{method.name}: Switching to LoadLoop mode')
        await self.channel.change_mode('LoadLoop')

        # Wait for trigger to switch to PumpAspirate mode
        logging.info(f'{method.name}: Waiting for second trigger')
        await self.channel.wait_for_trigger()

        # At this point, liquid handler is done
        #self.release_liquid_handler.set()

        logging.info(f'{method.name}: Switching to PumpPrimeLoop mode')
        await self.channel.change_mode('PumpPrimeLoop')

        # smart dispense the volume required to move plug quickly through loop
        logging.info(f'{method.name}: Moving plug through loop, total injection volume {self.channel.sample_loop.get_volume() - (pump_volume + excess_volume)} uL')
        await self.channel.syringe_pump.smart_dispense(self.channel.sample_loop.get_volume() - (pump_volume + excess_volume), self.channel.syringe_pump.max_dispense_flow_rate)

        # switch to standby mode
        logging.info(f'{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

class InjectLoop(MethodBase):
    """Injects the contents of the loop of one ROADMAP channel
    """

    def __init__(self, channel: RoadmapChannelBase) -> None:
        super().__init__([channel.syringe_pump, channel.loop_valve])
        self.channel = channel

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "InjectLoop"
        pump_volume: str | float = 0, # uL
        pump_flow_rate: str | float = 1, # mL/min

    async def run(self, **kwargs):
        """InjectLoop method"""

        method = self.MethodDefinition(**kwargs)

        pump_volume = float(method.pump_volume)
        pump_flow_rate = float(method.pump_flow_rate) * 1000 / 60 # convert to uL / s

        # change to inject mode
        await self.channel.change_mode('PumpInject')
        logging.info(f'{method.name}: Injecting {pump_volume} uL at flow rate {pump_flow_rate} uL / s')
        await self.channel.syringe_pump.smart_dispense(pump_volume, pump_flow_rate)

        # Prime loop
        await self.channel.primeloop(volume=1000)

class RoadmapChannel(RoadmapChannelBase):
    """Roadmap channel with populated methods
    """

    def __init__(self, loop_valve: HamiltonValvePositioner, syringe_pump: HamiltonSyringePump, flow_cell: FlowCell, sample_loop: FlowCell, injection_node: Node | None = None, name: str = '') -> None:
        super().__init__(loop_valve, syringe_pump, flow_cell, sample_loop, injection_node, name)

        self.methods = {'LoadLoop': LoadLoop(self),
                        'InjectLoop': InjectLoop(self)}
        
    def is_ready(self, method_name: str) -> bool:
        """Checks if all devices are unreserved for method

        Args:
            method_name (str): name of method to check

        Returns:
            bool: True if all devices are unreserved
        """

        return self.methods[method_name].is_ready()
        
class MultiChannelAssembly(AssemblyBasewithGSIOC):

    def __init__(self, channels: List[RoadmapChannel], distribution_valve: HamiltonValvePositioner, injection_port: InjectionPort, name='') -> None:
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