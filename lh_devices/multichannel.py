import asyncio
import json

from aiohttp.web_app import Application as Application
from aiohttp import web
from dataclasses import asdict
from typing import List

from autocontrol.status import Status
from autocontrol.task_struct import TaskData

from .assemblies import NestedAssemblyBase, AssemblyBase, InjectionChannelBase
from .autocontrolplugin import AutocontrolPlugin
from .history import HistoryDB, DatabasePlugin
from .methods import MethodResult

class MultiChannelAssembly(AutocontrolPlugin, NestedAssemblyBase):
    """Multichannel assembly, with endpoints for task submission, running, and saving"""

    def __init__(self,
                 channels: List[InjectionChannelBase],
                 assemblies: List[AssemblyBase],
                 database_path: str | None = None,
                 name='MultiChannel Assembly') -> None:

        NestedAssemblyBase.__init__(self, [], assemblies=channels + assemblies, name=name)
        AutocontrolPlugin.__init__(self, database_path, self.id, self.name)

        self.channels = channels
        
        if database_path is not None:
            for ch in self.channels:
                ch.method_callbacks.append(self.async_save_to_database)
            
    async def initialize(self) -> None:
        """Initialize the channels"""
        await asyncio.gather(*[ch.initialize() for ch in self.channels])
        await self.trigger_update()
    
    async def _handle_task(self, request: web.Request) -> web.Response:
        """Handles a submitted task"""
        data = await request.json()
        task = TaskData(**data)
        self.logger.info(f'{self.name} received task {task}')
        channel: int = task.channel
        if channel < len(self.channels):
            method = task.method_data['method_list'][0]
            method_name: str = method['method_name']
            method_data: dict = method['method_data']
            #if self.channels[channel].method_runner.is_ready(method_name):
            self.channels[channel].run_method(method_name, method_data, id=str(task.id))
            
            return web.Response(text='accepted', status=200)
            
            #return web.Response(text='busy', status=503)
        
        return web.Response(text=f'error: channel {channel} does not exist', status=400)

    async def _get_status(self, request):
            
            statuses = [Status.BUSY if ch.reserved else Status.IDLE for ch in self.channels]

            return web.Response(text=json.dumps(dict(status=Status.IDLE,
                                          channel_status=statuses)),
                                status=200)

    def create_web_app(self, template='roadmap.html') -> Application:
        app = AutocontrolPlugin.create_web_app(self, template=template)

        for i, channel in enumerate(self.channels):
            app.add_subapp(f'/{i}/', channel.create_web_app(template))
            app.add_subapp(f'/{channel.id}/', channel.create_web_app(template))

        return app

    async def get_info(self) -> dict:
        """Gets object state as dictionary

        Returns:
            dict: object state
        """
        d = await AutocontrolPlugin.get_info(self)
        d.update(await NestedAssemblyBase.get_info(self))
        d.update({'devices': {}})

        return d