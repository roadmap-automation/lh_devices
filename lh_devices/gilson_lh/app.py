"""Gilson LH device service entrypoint.

Starts:
  - LayoutPlugin (aiohttp + socket.io webview with Trilution callbacks)
  - GilsonLHBrokerWorker (AMQP consumer on exchange.instrument)

Port 5001 (matches lh_manager's legacy lhdevice.address).
"""

import asyncio
import datetime
import logging
from pathlib import Path

from lh_devices.layout import LayoutPlugin
from lh_devices.webview import run_socket_app

from .app_config import config
from .broker_plugin import GilsonLHBrokerWorker
from .lhinterface import lh_interface
from .notify import notifier
from .webview import get_routes

LOG_PATH = config.log_path
HOST = 'localhost'
PORT = 5001
DEVICE_ID = 'gilson_lh'


async def run():
    # Notifications
    notifier.load_config(config.notify_path)

    # Layout
    layout_plugin = LayoutPlugin(id=DEVICE_ID, name='Gilson 271 Liquid Handler')
    layout_plugin.layout_path = config.layout_path
    config.persistent_path.mkdir(parents=True, exist_ok=True)
    layout_plugin.load_layout()

    # Broker worker
    broker_worker = GilsonLHBrokerWorker(
        layout_plugin=layout_plugin,
        lh_iface=lh_interface,
        local_port=PORT,
        device_id=DEVICE_ID,
    )

    # Web app: LayoutPlugin base + Trilution callback routes
    app = layout_plugin.create_web_app(template='roadmap.html')
    app.add_routes(get_routes(layout_plugin, lh_interface))

    # Start broker (must happen after layout is loaded)
    await broker_worker.start()

    runner = await run_socket_app(app, HOST, PORT)
    logging.info("Gilson LH service running at http://%s:%d", HOST, PORT)

    try:
        while True:
            await asyncio.sleep(1.0)
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        logging.info("Gilson LH service shutting down.")
        await runner.cleanup()


if __name__ == '__main__':
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                LOG_PATH / (datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_gilson_lh.txt')
            ),
            logging.StreamHandler(),
        ],
        format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    asyncio.run(run(), debug=False)
