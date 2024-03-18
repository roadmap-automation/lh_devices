import datetime
import logging
import asyncio

from gsioc import GSIOC
from qcmd import QCMDRecorderDevice

async def main():
    gsioc = GSIOC(62, 'COM13', 19200)
    qcmd_recorder = QCMDRecorderDevice('localhost', 5011)
    try:
        await qcmd_recorder.initialize_gsioc(gsioc)
    finally:
        await qcmd_recorder.recorder.session.close()


logging.basicConfig(handlers=[
                        logging.FileHandler(datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_qcmd_recorder_log.txt'),
                        logging.StreamHandler()
                    ],
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

asyncio.run(main(), debug=True)