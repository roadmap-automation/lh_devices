import json
import logging
import socketio

from pathlib import Path
from aiohttp import web

"""
Code to automatically generate a web application with:
1. Individual device status
2. Individual device control

Ideally, also:
1. A view of the entire device network
2. Hyperlinks between connected devices

Approach:
* each device makes its own endpoint page and defines its own control
* top-level code crawls through the network and makes a URL structure that reflects it

"""

TEMPLATE_PATH = Path(__file__).parent / 'static'

sio = socketio.AsyncServer()

class WebNodeBase:

    id: str = ''
    name: str = ''

    def create_web_app(self, template: str) -> web.Application:
        """Creates a web application for this specific assembly by creating a webpage per device

        Args:
            template (str | None): template for root path. Required.

        Returns:
            web.Application: web application for this device
        """

        app = web.Application()
        routes = web.RouteTableDef()

        @routes.get('/')
        async def get_handler(request: web.Request) -> web.Response:
             return web.FileResponse(TEMPLATE_PATH / template)

        routes.static('/src', TEMPLATE_PATH / 'src', follow_symlinks=True)

        @routes.get('/state')
        @routes.get(f'/{self.id}/state')
        async def get_state(request: web.Request) -> web.Response:
            state = await self.get_info()
            return web.Response(text=json.dumps(state), status=200)

        @sio.on(self.id)
        async def event_handler(event, data):
            # starts handling the event
            await self.event_handler(data['command'], data['data'])

        app.add_routes(routes)

        return app

    async def get_info(self) -> dict:
        """Gets object state as dictionary

        Returns:
            dict: object state
        """

        return {'name': self.name,
                'id': self.id,
                'type': None}

    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        logging.info(f'{self.name} received {command} with data {data}')

    async def trigger_update(self):
        """Emits a socketio event with id"""

        await sio.emit(self.id)

async def run_socket_app(app: web.Application, host='localhost', port=5003) -> web.AppRunner:
    """Connects socketio app to aiohttp application and runs it

    Args:
        app (web.Application): aiohttp web application
    """

    sio.attach(app)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    return runner

