import json
import aiohttp
import asyncio
import logging
from gsioc import GSIOCDeviceBase, GSIOC, GSIOCMessage

class GSIOCTimer(GSIOCDeviceBase):
    """Basic GSIOC Timer. When timer command is received, starts a timer for specified time,
        during which idle queries will return a busy signal. Designed for use with Gilson LH,
        which can pause until timer is no longer idle."""

    def __init__(self, gsioc: GSIOC, name='GSIOCTimer') -> None:
        super().__init__(gsioc)
        self.idle = True
        self.wait_time = None
        self.name = name

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:
        """Handles GSIOC messages.
        Args:
            data (GSIOCMessage): GSIOC Message to be parsed / handled

        Returns:
            str: response (only for GSIOC immediate commands)
        """

        if data.data == 'Q':
            # busy query
            return 'idle' if self.idle else 'busy'
        
        elif data.data.startswith('timer: '):
            self.wait_time = float(data.data.split('timer: ', 1)[1])
            await self.gsioc_command_queue.put(asyncio.create_task(self.timer()))
        
        else:
            return 'error: unknown command'
        
        return None

    async def timer(self) -> None:
        """Executes timer
        """

        # don't start another timer if one is already running
        if self.idle:
            self.idle = False
            await asyncio.sleep(self.wait_time)
            self.idle = True
        else:
            logging.warning(f'{self.name}: Timer is already running...')

class QCMDRecorder(GSIOCTimer):
    """QCMD recording device."""

    def __init__(self, gsioc: GSIOC, qcmd_address: str = 'localhost', qcmd_port: int = 5011, name='QCMDRecorder') -> None:
        super().__init__(gsioc, name)
        self.qcmd_address = qcmd_address
        self.qcmd_port = qcmd_port
        self.sleep_time = None
        self.record_time = None
        self.tag_name = None

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:
        """Handles GSIOC messages.
        Args:
            data (GSIOCMessage): GSIOC Message to be parsed / handled

        Returns:
            str: response (only for GSIOC immediate commands)
        """

        if data.data == 'Q':
            # busy query
            return 'idle' if self.idle else 'busy'

        elif data.data.startswith('tag: '):
            self.tag_name = data.data.split('tag: ', 1)[1]

        elif data.data.startswith('sleep: '):
            self.sleep_time = float(data.data.split('sleep: ', 1)[1])

        elif data.data.startswith('record: '):
            self.record_time = float(data.data.split('record: ', 1)[1])

        elif data.data == 'start':
            await self.gsioc_command_queue.put(asyncio.create_task(self.timer()))
        
        else:
            return 'error: unknown command'
        
        return None

    async def timer(self) -> None:
        """Executes timer and sends record command to QCMD
        """

        # calculate total wait time
        self.wait_time = self.record_time + self.sleep_time

        # wait the full time
        await super().timer()

        # send an http request to 
        response = await aiohttp.request(method='POST',
                              url=f'http://{self.qcmd_address}:{self.qcmd_port}/QCMD/',
                              json={'command': 'set_tag',
                                               'value': {'tag': self.tag_name,
                                                         'delta_t': self.record_time}})
        
        logging.info(f'{self}: received response {response}')

async def main():
    gsioc = GSIOC(62, 'COM13', 19200)
    qcmd_recorder = QCMDRecorder(gsioc, 'localhost', 5011)
    await qcmd_recorder.initialize()

if __name__=='__main__':

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG)

    asyncio.run(main(), debug=True)