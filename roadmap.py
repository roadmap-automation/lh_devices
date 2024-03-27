import json
import logging
from typing import Coroutine, List, Dict
from dataclasses import dataclass

from aiohttp.web_app import Application as Application
from aiohttp import web

from assemblies import Mode
from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump
from gsioc import GSIOC, GSIOCMessage
from components import InjectionPort, FlowCell
from assemblies import AssemblyBase, AssemblyBasewithGSIOC, Network, NestedAssemblyBase
from connections import connect_nodes, Node
from methods import MethodBase

class RoadmapChannelBase(AssemblyBasewithGSIOC):

    def __init__(self, loop_valve: HamiltonValvePositioner,
                       syringe_pump: HamiltonSyringePump,
                       flow_cell: FlowCell,
                       sample_loop: FlowCell,
                       gsioc: GSIOC | None = None,
                       injection_node: Node | None = None,
                       name: str = '') -> None:
        
        # Devices
        self.loop_valve = loop_valve
        self.syringe_pump = syringe_pump
        self.flow_cell = flow_cell
        self.sample_loop = sample_loop
        self.injection_node = injection_node
        self.gsioc = gsioc
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

        if not self.methods[method_name].is_ready():
            logging.error(f'{self.name}: not all devices in {method_name} are available')
        else:
            super().run_method(self.methods[method_name].run(**method_kwargs))

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
        self.dead_volume_mode: str = 'LoadLoop'

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

        # Connect to GSIOC communications
        gsioc_task = asyncio.create_task(self.channel.initialize_gsioc(self.channel.gsioc))

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.channel.get_dead_volume(self.channel.injection_node, self.dead_volume_mode)

        # blocks if there's already something in the dead volume queue
        await self.channel.dead_volume.put(dead_volume)
        logging.info(f'{self.channel.name}.{method.name}: dead volume set to {dead_volume}')

        # Wait for trigger to switch to LoadLoop mode
        logging.info(f'{self.channel.name}.{method.name}: Waiting for first trigger')
        await self.channel.wait_for_trigger()
        logging.info(f'{self.channel.name}.{method.name}: Switching to LoadLoop mode')
        await self.channel.change_mode('LoadLoop')

        # Wait for trigger to switch to PumpAspirate mode
        logging.info(f'{self.channel.name}.{method.name}: Waiting for second trigger')
        await self.channel.wait_for_trigger()

        # At this point, liquid handler is done, release communications
        gsioc_task.cancel()
        #self.release_liquid_handler.set()

        logging.info(f'{self.channel.name}.{method.name}: Switching to PumpPrimeLoop mode')
        await self.channel.change_mode('PumpPrimeLoop')

        # smart dispense the volume required to move plug quickly through loop
        logging.info(f'{self.channel.name}.{method.name}: Moving plug through loop, total injection volume {self.channel.sample_loop.get_volume() - (pump_volume + excess_volume)} uL')
        await self.channel.syringe_pump.smart_dispense(self.channel.sample_loop.get_volume() - (pump_volume + excess_volume), self.channel.syringe_pump.max_dispense_flow_rate)

        # switch to standby mode
        logging.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
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
        logging.info(f'{self.channel.name}.{method.name}: Injecting {pump_volume} uL at flow rate {pump_flow_rate} uL / s')
        await self.channel.syringe_pump.smart_dispense(pump_volume, pump_flow_rate)

        # Prime loop
        await self.channel.primeloop(volume=1000)

class DirectInject(MethodBase):
    """Directly inject from LH to a ROADMAP channel flow cell
    """

    def __init__(self, channel: RoadmapChannelBase) -> None:
        super().__init__([channel.loop_valve])
        self.channel = channel
        self.dead_volume_mode: str = 'LHInject'

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):
        
        name: str = "DirectInject"

    async def run(self, **kwargs):
        """LoadLoop method, synchronized via GSIOC to liquid handler"""

        method = self.MethodDefinition(**kwargs)

        # Connect to GSIOC communications
        gsioc_task = asyncio.create_task(self.channel.initialize_gsioc(self.channel.gsioc))

        # Set dead volume and wait for method to ask for it (might need brief wait in the calling
        # method to make sure this updates in time)
        dead_volume = self.channel.get_dead_volume(self.channel.injection_node, self.dead_volume_mode)

        # blocks if there's already something in the dead volume queue
        await self.channel.dead_volume.put(dead_volume)
        logging.info(f'{self.channel.name}.{method.name}: dead volume set to {dead_volume}')

        # Wait for trigger to switch to LHPrime mode (fast injection of air gap + dead volume + extra volume)
        logging.info(f'{self.channel.name}.{method.name}: Waiting for first trigger')
        await self.channel.wait_for_trigger()
        logging.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await self.channel.change_mode('LHPrime')

        # Wait for trigger to switch to {method.name} mode (LH performs injection)
        logging.info(f'{self.channel.name}.{method.name}: Waiting for second trigger')
        await self.channel.wait_for_trigger()

        logging.info(f'{self.channel.name}.{method.name}: Switching to LHInject mode')
        await self.channel.change_mode('LHInject')

        # Wait for trigger to switch to LHPrime mode (fast injection of extra volume + final air gap)
        logging.info(f'{self.channel.name}.{method.name}: Waiting for third trigger')
        await self.channel.wait_for_trigger()

        logging.info(f'{self.channel.name}.{method.name}: Switching to LHPrime mode')
        await self.channel.change_mode('LHPrime')

        # Wait for trigger to switch to Standby mode (this may not be necessary)
        logging.info(f'{self.channel.name}.{method.name}: Waiting for fourth trigger')
        await self.channel.wait_for_trigger()

        # switch to standby mode
        logging.info(f'{self.channel.name}.{method.name}: Switching to Standby mode')            
        await self.channel.change_mode('Standby')

        # At this point, liquid handler is done, release communications
        gsioc_task.cancel()

class RoadmapChannel(RoadmapChannelBase):
    """Roadmap channel with populated methods
    """

    def __init__(self, loop_valve: HamiltonValvePositioner, syringe_pump: HamiltonSyringePump, flow_cell: FlowCell, sample_loop: FlowCell, gsioc: GSIOC, injection_node: Node | None = None, name: str = '') -> None:
        super().__init__(loop_valve, syringe_pump, flow_cell, sample_loop, gsioc, injection_node, name)

        self.methods = {'LoadLoop': LoadLoop(self),
                        'InjectLoop': InjectLoop(self),
                        'DirectInject': DirectInject(self)}
        
    def is_ready(self, method_name: str) -> bool:
        """Checks if all devices are unreserved for method

        Args:
            method_name (str): name of method to check

        Returns:
            bool: True if all devices are unreserved
        """

        return self.methods[method_name].is_ready()
        
class RoadmapChannelAssembly(NestedAssemblyBase, AssemblyBasewithGSIOC):

    def __init__(self, channels: List[RoadmapChannel], distribution_valve: HamiltonValvePositioner, injection_port: InjectionPort, name='') -> None:
        NestedAssemblyBase.__init__(self, [dev for ch in channels for dev in ch.devices] + [distribution_valve], channels, name)
        AssemblyBasewithGSIOC.__init__(self, self.devices, name)

        """TODO:
            1. distribution valve is generalized to a distribution system
            2. add change_direction capability to ROADMAP channels
        """

        self.injection_port = injection_port
        self.distribution_valve = distribution_valve

        # Build network
        self.network = Network([dev for ch in channels for dev in ch.network.devices] + [distribution_valve, injection_port])

        self.modes = {'Standby': Mode(valves={distribution_valve: 8})}

        # update channels and methods with upstream information
        for i, ch in enumerate(channels):
            ch.network = self.network
            ch.injection_node = injection_port.nodes[0]
            if 'LoadLoop' in ch.modes.keys():
                ch.modes['LoadLoop'].valves.update({distribution_valve: 1 + 2 * i})
            if 'LoadLoop' in ch.methods.keys():
                ch.methods['LoadLoop'].devices.append(distribution_valve)
            if 'DirectInject' in ch.modes.keys():
                ch.modes['DirectInject'].valves.update({distribution_valve: 2 + 2 * i})
            if 'DirectInject' in ch.methods.keys():
                ch.methods['DirectInject'].devices.append(distribution_valve)

        self.channels = channels

    async def initialize(self) -> None:
        """Initialize the loop as a unit and the distribution valve separately"""
        await asyncio.gather(*[ch.initialize() for ch in self.channels], self.distribution_valve.initialize())

    def run_channel_method(self, channel: int, method_name: str, method_data: dict) -> None:
        return self.channels[channel].run_method(method_name, **method_data)
    
    def create_web_app(self, template='roadmap.html') -> Application:
        app = super().create_web_app(template=template)
        routes = web.RouteTableDef()

        @routes.post('/SubmitTask')
        async def handle_task(request: web.Request) -> web.Response:
            # TODO: turn task into a dataclass; parsing will change
            task = request.json()
            channel: int = task['channel']
            method_name: str = task['method_name']
            method_data: dict = task['method_data']
            if self.channels[channel].is_ready(method_name):
                self.run_channel_method(channel, method_name, method_data)
                return web.Response(text='accepted', status=200)
            
            return web.Response(text='busy', status=200)
        
        @routes.get('/GetTaskData')
        async def handle_task(request: web.Request) -> web.Response:
            # TODO: turn task into a dataclass; parsing will change
            task = request.json()
            task_id = task['id']

            # TODO: actually return task data
            # TODO: Determine what task data we want to save. Logging? success? Any returned errors?
            return web.Response(text=json.dumps({'id': task_id}), status=200)
        
        app.add_routes(routes)

        for i, channel in enumerate(self.channels):
            app.add_subapp(f'/{i}/', channel.create_web_app())

        return app

if __name__=='__main__':

    import asyncio
    from HamiltonComm import HamiltonSerial
    from valve import LoopFlowValve, SyringeYValve, DistributionValve, SyringeLValve
    from webview import run_socket_app

    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

    if True:
        async def main():
            gsioc = GSIOC(62, 'COM13', 19200)
            ser = HamiltonSerial(port='COM7', baudrate=38400)
            #ser = HamiltonSerial(port='COM3', baudrate=38400)
            dvp = HamiltonValvePositioner(ser, '2', DistributionValve(8, name='distribution_valve'), name='distribution_valve_positioner')
            mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(6, name='loop_valve'), name='loop_valve_positioner')
            sp = HamiltonSyringePump(ser, '0', SyringeLValve(4, name='syringe_LValve'), 5000., False, name='syringe_pump')
            sp.max_dispense_flow_rate = 5 * 1000 / 60
            sp.max_aspirate_flow_rate = 15 * 1000 / 60
            ip = InjectionPort('LH_injection_port')
            fc = FlowCell(139, 'flow_cell')
            sampleloop = FlowCell(5060., 'sample_loop')

            channel_0 = RoadmapChannel(mvp, sp, fc, sampleloop, injection_node=ip.nodes[0], gsioc=gsioc, name='channel_0')
            channel_1 = RoadmapChannel(mvp, sp, fc, sampleloop, injection_node=ip.nodes[0], gsioc=gsioc, name='channel_1')

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

            qcmd_system = RoadmapChannelAssembly([channel_0, channel_1], dvp, ip, name='MultiChannel System')
            app = qcmd_system.create_web_app(template='roadmap.html')
            runner = await run_socket_app(app, 'localhost', 5003)
            print(json.dumps(await qcmd_system.get_info()))
            #lh = SimLiquidHandler(qcmd_channel)

            try:
                #qcmd_system.distribution_valve.valve.move(2)
                await qcmd_system.initialize()
                await asyncio.sleep(2)
                await channel_0.change_mode('PumpPrimeLoop')
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
                asyncio.gather(
                            runner.cleanup())

            

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