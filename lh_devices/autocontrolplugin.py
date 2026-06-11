import asyncio
import base64
import json

from aiohttp.web_app import Application as Application
from aiohttp import web

from pathlib import Path

from autocontrol.status import Status
from autocontrol.task_struct import TaskData

from .history import DatabasePlugin
from .methods import MethodPlugin

class AutocontrolPlugin(MethodPlugin, DatabasePlugin):
    """Mixin for single-channel broker-capable assemblies.

    Exposes a single-element `channels` list so DeviceBrokerWorker can treat
    this device the same way it treats multi-channel assemblies.  Multi-channel
    assemblies shadow this property by setting self.channels = [...] in __init__.
    """

    def __init__(self, database_path: Path | None = None, id = '', name = ''):
        MethodPlugin.__init__(self, id, name)
        DatabasePlugin.__init__(self, database_path=database_path)

    @property
    def channels(self) -> list:
        return self.__dict__.get('_channels', [self])

    @channels.setter
    def channels(self, value: list) -> None:
        self.__dict__['_channels'] = value

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
        task_id = request.rel_url.query.get('task_id', '')

        record = self.read_from_database(task_id)
        if record is None:
            return web.Response(text=f'error: id {task_id} does not exist', status=400)

        return web.json_response({"data": record.result})

    async def _get_task_files(self, request: web.Request) -> web.Response:
        task_id = request.rel_url.query.get('task_id', '')
        record = self.read_from_database(task_id)
        if record is None:
            return web.Response(text=f'error: id {task_id} does not exist', status=400)
        method = self.methods.get(record.method_name)
        if method is None:
            return web.Response(text=f'error: method {record.method_name!r} not found', status=400)
        raw = await asyncio.to_thread(method.file_generator, record)
        encoded = {
            name: (data.decode() if name.endswith('.json') else base64.b64encode(data).decode())
            for name, data in raw.items()
        }
        return web.json_response(encoded)
