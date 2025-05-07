import asyncio
import json

from aiohttp.web_app import Application as Application
from aiohttp import web
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from autocontrol.status import Status
from autocontrol.task_struct import TaskData

from .autocontrolplugin import AutocontrolPlugin
from .assemblies import Mode, AssemblyBase
from .components import InjectionPort
from .device import DeviceBase, ValvePositionerBase
from .layout import LayoutPlugin
from .methods import MethodBase
from .waste import WasteInterfaceBase

class DistributionBase(AssemblyBase):

    def __init__(self, n_positions, devices: List[DeviceBase], injection_port: InjectionPort, name: str = '', id: str = None) -> None:
        super().__init__(devices, name, id)

        self.injection_port = injection_port
        self.n_positions = n_positions
        self.modes: Dict[str, Mode] = {}

    async def release(self):
        """Releases reservations on all devices
        """
        for dev in self.devices:
            dev.reserved = False
            await dev.trigger_update()

class DistributionSingleValve(DistributionBase):

    def __init__(self, distribution_valve: ValvePositionerBase, injection_port: InjectionPort, name: str = '', id: str = None) -> None:
        super().__init__(distribution_valve.valve.n_positions, [distribution_valve], injection_port, name)

        self.modes = {str(i): Mode({distribution_valve: i}, final_node=distribution_valve.get_nodes()[i]) for i in range(self.n_positions + 1)}
        self.modes.update({'Standby': Mode({distribution_valve: 0})})


# =============== Autocontrol-enabled distribution systems =======================

class InitiateDistribution(MethodBase):

    def __init__(self, distribution_system: DistributionBase, waste_tracker: WasteInterfaceBase = WasteInterfaceBase()):
        super().__init__(distribution_system.devices, waste_tracker)
        self.distribution_system = distribution_system

    @dataclass
    class MethodDefinition(MethodBase.MethodDefinition):

        name: str = "InitiateDistribution"

    async def run(self, **kwargs):
        """Initiates the distribution system
        """

        self.reserve_all()
        await self.trigger_update()

        try:
            # poll every second to see if devices have been released yet
            while self.distribution_system.reserved:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.release_all()

class DistributionSingleValveTwoSource(AutocontrolPlugin, DistributionBase, LayoutPlugin):

    def __init__(self,
                 distribution_valve: ValvePositionerBase,
                 source_valve: ValvePositionerBase,
                 injection_port: InjectionPort,
                 database_path: Path | None = None,
                 name: str = '',
                 id: str = None) -> None:
        AutocontrolPlugin.__init__(self, database_path, id, name)
        DistributionBase.__init__(self, distribution_valve.valve.n_positions, [distribution_valve, source_valve], injection_port, name, id)
        LayoutPlugin.__init__(self, self.id, self.name)

        self.modes = {str(i): Mode({distribution_valve: i}, final_node=distribution_valve.get_nodes()[i]) for i in range(self.n_positions + 1)}
        self.modes.update({'LH': Mode({source_valve: 1}),
                           'Rinse': Mode({source_valve: 3}),
                           'Standby': Mode({source_valve: 0,
                                            distribution_valve: 0})})
        
        self.methods.update({'InitiateDistribution': InitiateDistribution(self)})
        
    async def _get_status(self, request):
        return web.Response(text=json.dumps(dict(status=Status.BUSY if any(dev.reserved for dev in self.devices) else Status.IDLE,
                                        channel_status=[])),
                            status=200)

    def create_web_app(self, template='roadmap.html'):
        app = super().create_web_app(template)

        app.add_routes(LayoutPlugin._get_routes(self))

        return app

    async def get_info(self) -> Dict:
        d = await AutocontrolPlugin.get_info(self)
        d.update(await DistributionBase.get_info(self))

        d['controls'] = d['controls'] | {'release': {'type': 'button',
                                                     'text': 'Release'}}
        
        return d
    
    async def event_handler(self, command: str, data: Dict) -> None:

        if command == 'release':
            await self.release()
        else:
            await DistributionBase.event_handler(self, command, data)
            await AutocontrolPlugin.event_handler(self, command, data)