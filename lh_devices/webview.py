import aiohttp_cors
import json
import logging
import socketio

from pathlib import Path
from aiohttp import web
from uuid import uuid4

from .logutils import Loggable

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

TEMPLATE_PATH = Path(__file__).parent / 'templates'

sio = socketio.AsyncServer(cors_allowed_origins='*')

class WebNodeBase(Loggable):

    def __init__(self, id: str = '', name: str = ''):
        self.id = str(uuid4()) if id is None else id
        self.name = name
        Loggable.__init__(self)

    def create_web_app(self, template: str = 'roadmap.html') -> web.Application:
        """Creates a web application for this specific web node by creating a webpage per device

        Args:
            template (str | None): template for root path. Required.

        Returns:
            web.Application: web application for this device
        """

        app = web.Application()
        routes = web.RouteTableDef()

        @routes.get('/')
        @routes.get(f'/{self.id}')
        async def get_handler(request: web.Request) -> web.Response:
            return web.FileResponse(TEMPLATE_PATH / template)

        @routes.get('/state')
        @routes.get(f'/{self.id}/state')
        async def get_state(request: web.Request) -> web.Response:
            state = await self.get_info()
            try:
                json_state = json.dumps(state)
            except TypeError:
                print(state)

            return web.Response(text=json_state, status=200)

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

    cors = aiohttp_cors.setup(app, defaults={
   "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*"
    )
    })

    for route in list(app.router.routes()):
        cors.add(route)
   
    sio.attach(app)
    
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    return runner

