import asyncio
import datetime
import logging
import pathlib
import webbrowser

from ..gilson.gsioc import GSIOC
from .recorder import QCMDRecorderDevicewithCamera
from ..camera.camera import FIT0819
from ..webview import run_socket_app

LOG_PATH = pathlib.Path(__file__).parent.parent.parent / 'logs'

async def main():
    gsioc = GSIOC(62, 'COM13', 19200)
    qcmd_recorder = QCMDRecorderDevicewithCamera('localhost', 5011, FIT0819(None), name='QCMD Recorder')
    app = qcmd_recorder.create_web_app(template='roadmap.html')
    runner = await run_socket_app(app, 'localhost', 5006)
    webbrowser.open('http://localhost:5006/')
    try:
        await qcmd_recorder.initialize_gsioc(gsioc)
    except asyncio.CancelledError:
        pass
    finally:
        logging.info('Closing QCMD Recorder...')
        await qcmd_recorder.recorder.session.close()
        await runner.cleanup()

if __name__ == '__main__':
    logging.basicConfig(handlers=[
                            logging.FileHandler(LOG_PATH / (datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_qcmd_recorder_log.txt')),
                            logging.StreamHandler()
                        ],
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    asyncio.run(main(), debug=True)