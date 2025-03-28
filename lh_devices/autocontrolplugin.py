import json

from aiohttp.web_app import Application as Application
from aiohttp import web

from dataclasses import asdict
from pathlib import Path

from autocontrol.status import Status
from autocontrol.task_struct import TaskData

from .history import DatabasePlugin
from .methods import MethodPlugin

class AutocontrolPlugin(MethodPlugin, DatabasePlugin):

    def __init__(self, database_path: Path | None = None, id = '', name = ''):
        MethodPlugin.__init__(self, id, name)
        DatabasePlugin.__init__(self, database_path=database_path)

    async def _handle_task(self, request: web.Request) -> web.Response:
        """Handles a submitted task"""
        data = await request.json()
        task = TaskData(**data)
        self.logger.info(f'{self.name} received task {task}')
        method = task.method_data['method_list'][0]
        method_name: str = method['method_name']
        method_data: dict = method['method_data']
        self.run_method(method_name, method_data, id=str(task.id))
                
        return web.Response(text='accepted', status=200)

    async def _get_status(self, request: web.Request) -> web.Response:
        """Status request"""
        return web.Response(text=json.dumps(dict(status=Status.BUSY if len(self.method_runner.active_methods) else Status.IDLE,
                                        channel_status=[])),
                            status=200)

    async def _get_task(self, request: web.Request) -> web.Response:
        """Handles requests for information about a task. Dummy method round-trips the response through a TaskData serialization process."""
        data: dict = await request.json()
        task_id = data.get('task_id', '')
        
        record = self.read_from_database(task_id)
        if record is None:
            return web.Response(text=f'error: id {task_id} does not exist', status=400)

        return web.Response(text=json.dumps(asdict(record)), status=200)
