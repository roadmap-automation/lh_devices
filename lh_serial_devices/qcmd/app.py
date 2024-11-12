import asyncio
import datetime
import logging
import pathlib

from .qcmd import QCMDMultiChannelMeasurementDevice
from ..webview import run_socket_app

LOG_PATH = pathlib.Path(__file__).parent.parent.parent / 'logs'

async def qcmd_multichannel_measure():

    measurement_system = QCMDMultiChannelMeasurementDevice('localhost', 5011, qcmd_ids=['13117490', '13110090'], database_path=LOG_PATH / 'qcmd.db')
    await measurement_system.initialize()
    app = measurement_system.create_web_app(template='roadmap.html')
    runner = await run_socket_app(app, 'localhost', 5005)
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        logging.info('Cleaning up...')
        await runner.cleanup()

if __name__ == '__main__':

    logging.basicConfig(handlers=[
                        logging.FileHandler(LOG_PATH / (datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_qcmdmulti_recorder_log.txt')),
                        logging.StreamHandler()
                    ],
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    asyncio.run(qcmd_multichannel_measure(), debug=True)

