"""Gilson LH device service entrypoint.

Starts:
  - lh_interface (LHInterface: AutocontrolPlugin + DeviceBase + LayoutPlugin)
  - GilsonLHBrokerWorker (AMQP consumer on exchange.instrument)

Port 5001 (matches lh_manager's legacy lhdevice.address).
"""

import asyncio
import datetime
import logging

from lh_devices.core.bedlayout import LHBedLayout
from lh_devices.webview import run_socket_app

from ..gilson.gsioc import GSIOC
from .app_config import config
from .broker_plugin import GilsonLHBrokerWorker
from .lhinterface import lh_interface
from .notify import notifier

LOG_PATH = config.log_path
HOST = 'localhost'
PORT = 5009
DEVICE_ID = 'lh'


async def run():
    # Notifications
    notifier.load_config(config.notify_path)

    # lh_interface IS the layout plugin — configure its layout path and load
    lh_interface.layout_path = config.layout_path
    config.persistent_path.mkdir(parents=True, exist_ok=True)
    lh_interface.load_layout()
    if lh_interface.layout is None:
        logging.warning("No layout file at %s — starting with empty layout.", config.layout_path)
        lh_interface.layout = LHBedLayout()

    # GSIOC serial connection (Trilution ↔ gilson_lh ↔ broker)
    gsioc = GSIOC(62, 'COM13', 19200)

    # Broker worker: lh_interface serves as both layout_plugin and lh_iface
    broker_worker = GilsonLHBrokerWorker(
        lh_iface=lh_interface,
        local_port=PORT,
        device_id=DEVICE_ID,
        gsioc=gsioc,
    )

    # Web app: create_web_app() combines AutocontrolPlugin + LayoutPlugin + Trilution routes
    app = lh_interface.create_web_app(template='roadmap.html')

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
