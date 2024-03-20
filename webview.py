from aiohttp import web
import socketio

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

sio = socketio.AsyncServer()

async def run_socket_app(app: web.Application, host='localhost', port=5003) -> web.AppRunner:
    """Connects socketio app to aiohttp application and runs it

    Args:
        app (web.Application): aiohttp web application
    """

    sio.attach(app)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    return runner

