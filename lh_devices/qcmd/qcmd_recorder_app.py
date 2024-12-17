import asyncio
import datetime
import logging
import pathlib

from ..gilson.gsioc import GSIOC
from .recorder import QCMDRecorderDevice

LOG_PATH = pathlib.Path(__file__).parent.parent.parent / 'logs'

async def main():
    gsioc = GSIOC(62, 'COM13', 19200)
    qcmd_recorder = QCMDRecorderDevice('localhost', 5011, name='QCMD Recorder')
    try:
        await qcmd_recorder.initialize_gsioc(gsioc)
    except asyncio.CancelledError:
        pass
    finally:
        logging.info('Closing QCMD Recorder...')
        await qcmd_recorder.recorder.session.close()

if __name__ == '__main__':
    logging.basicConfig(handlers=[
                            logging.FileHandler(LOG_PATH / (datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_qcmd_recorder_log.txt')),
                            logging.StreamHandler()
                        ],
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    asyncio.run(main(), debug=True)