import asyncio
import json

from aiohttp.web_app import Application as Application
from aiohttp import web
from dataclasses import asdict
from typing import List

from autocontrol.status import Status
from autocontrol.task_struct import TaskData

from .assemblies import NestedAssemblyBase, AssemblyBase, InjectionChannelBase
from .history import HistoryDB
from .methods import MethodResult

class MultiChannelAssembly(NestedAssemblyBase):
    """Multichannel assembly, with endpoints for task submission, running, and saving"""

    def __init__(self,
                 channels: List[InjectionChannelBase],
                 assemblies: List[AssemblyBase],
                 database_path: str | None = None,
                 name='MultiChannel Assembly') -> None:

        NestedAssemblyBase.__init__(self, [], assemblies=channels + assemblies, name=name)

        self.channels = channels
        
        if database_path is not None:
            for ch in self.channels:
                ch.method_callbacks.append(self.save_to_database)
            
            self.database_path = database_path

    async def save_to_database(self, result: MethodResult):
        """Saves a method result to the database, only if a task id is associated with it

        Args:
            result (MethodResult): result to save
        """

        if result.id is not None:
            with HistoryDB(self.database_path) as db:
                db.smart_insert(result)

    def read_from_database(self, id: str) -> MethodResult | None:
        """Reads a method result from the database

        Args:
            id (str): id of record

        Returns:
            MethodResult | None: MethodResult object if id exists, otherwise None
        """

        with HistoryDB(self.database_path) as db:
            return db.search_id(id)

    async def initialize(self) -> None:
        """Initialize the channels"""
        await asyncio.gather(*[ch.initialize() for ch in self.channels])
        await self.trigger_update()

    def create_web_app(self, template='roadmap.html') -> Application:
        app = super().create_web_app(template=template)
        routes = web.RouteTableDef()

        @routes.post('/SubmitTask')
        async def handle_task(request: web.Request) -> web.Response:
            # TODO: turn task into a dataclass; parsing will change
            # testing: curl -X POST http://localhost:5003/SubmitTask -d "{\"channel\": 0, \"method_name\": \"DirectInjectBubbleSensor\", \"method_data\": {}}"
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
        
        @routes.get('/GetStatus')
        async def get_status(request: web.Request) -> web.Response:
            
            statuses = [Status.BUSY if ch.reserved else Status.IDLE for ch in self.channels]

            return web.Response(text=json.dumps(dict(status=Status.IDLE,
                                          channel_status=statuses)),
                                status=200)

        @routes.get('/GetTaskData')
        async def get_task(request: web.Request) -> web.Response:
            data = await request.json()
            task = TaskData(**data)
            
            record = self.read_from_database(task.id)
            if record is None:
                return web.Response(text=f'error: id {task.id} does not exist', status=400)

            return web.Response(text=json.dumps(asdict(record)), status=200)
        
        app.add_routes(routes)

        for i, channel in enumerate(self.channels):
            app.add_subapp(f'/{i}/', channel.create_web_app(template))
            app.add_subapp(f'/{channel.id}/', channel.create_web_app(template))

        return app

    async def get_info(self) -> dict:
        """Gets object state as dictionary

        Returns:
            dict: object state
        """

        d = await super().get_info()
        d.update({'devices': {}})

        return d