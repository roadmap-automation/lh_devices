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
        self.sleep_time = None
        self.record_time = None
        self.tag_name = None
        self.session = aiohttp.ClientSession(f'http://{qcmd_address}:{qcmd_port}')

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:
        """Handles GSIOC messages.
        Args:
            data (GSIOCMessage): GSIOC Message to be parsed / handled

        Returns:
            str: response (only for GSIOC immediate commands)
        """

        logging.debug(f'{self.name}: Received {data.data}')

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

        post_data = {'command': 'set_tag',
                     'value': {'tag': self.tag_name,
                               'delta_t': self.record_time}}

        logging.info(f'{self.session._base_url}/QCMD/ => {post_data}')

        # send an http request to QCMD server
        async with self.session.post('/QCMD/', json=post_data) as resp:
            response_json = await resp.json()
            logging.info(f'{self.session._base_url}/QCMD/ <= {response_json}')

async def main():
    gsioc = GSIOC(62, 'COM13', 19200)
    qcmd_recorder = QCMDRecorder(gsioc, 'localhost', 5011)
    try:
        await qcmd_recorder.initialize()
    finally:
        logging.info('Cleaning up...')
        qcmd_recorder.session.close()

if __name__=='__main__':

    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    asyncio.run(main(), debug=True)