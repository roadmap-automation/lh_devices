import asyncio

from aiohttp import ClientSession, ClientConnectionError
from urllib.parse import urlsplit

from ..gsioc import GSIOCMessage
from ..assemblies import AssemblyBasewithGSIOC
from ..logutils import Loggable

class Timer(Loggable):
    """Basic timer. Essentially serves as a sleep but only allows one instance to run."""

    def __init__(self, name='Timer') -> None:
        self.name = name
        self.timer_running: asyncio.Event = asyncio.Event()

        Loggable.__init__(self)

    async def start(self, wait_time: float = 0.0) -> bool:
        """Executes timer.

        Returns:
            bool: True if successful, False if not.
        """

        # don't start another timer if one is already running
        if not self.timer_running.is_set():
            self.timer_running.set()
            await asyncio.sleep(wait_time)
            self.timer_running.clear()
            return True
        else:
            self.logger.warning(f'{self.name}: Timer is already running, ignoring start command')
            return False

class QCMDRecorder(Timer):
    """QCMD-specific timer. At end of timing interval, sends HTTP request to QCMD to record tag."""

    def __init__(self, http_address: str = 'http://localhost:5011/QCMD/0/', name='QCMDRecorder') -> None:
        super().__init__(name)
        url_parts = urlsplit(http_address)
        self.session = ClientSession(f'{url_parts.scheme}://{url_parts.netloc}')
        self.url_path = url_parts.path

    async def record(self, tag_name: str = '', record_time: float = 0.0, sleep_time: float = 0.0) -> None:
        """Executes timer and sends record command to QCMD. Call by sending
            {"method": "record", {**kwargs}} over GSIOC.
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # calculate total wait time
        wait_time = record_time + sleep_time

        # wait the full time
        if await self.start(wait_time):

            post_data = {'command': 'set_tag',
                        'value': {'tag': tag_name,
                                'delta_t': record_time}}

            self.logger.info(f'{self.session._base_url}{self.url_path} => {post_data}')

            # send an http request to QCMD server
            try:
                async with self.session.post(self.url_path, json=post_data, timeout=10) as resp:
                    response_json = await resp.json()
                    self.logger.info(f'{self.session._base_url}{self.url_path} <= {response_json}')
            except (ConnectionRefusedError, ClientConnectionError):
                self.logger.error(f'request to {self.session._base_url}{self.url_path} failed: connection refused')

class QCMDRecorderDevice(AssemblyBasewithGSIOC):
    """QCMD recording device."""

    def __init__(self, qcmd_address: str = 'localhost', qcmd_port: int = 5011, name='QCMDRecorderDevice') -> None:
        super().__init__([], name)
        self.recorder = QCMDRecorder(f'http://{qcmd_address}:{qcmd_port}/QCMD/', f'{self.name}.QCMDRecorder')

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:
        """Handles GSIOC message but deals with Q more robustly than the base method"""

        if data.data == 'Q':
            response = 'busy' if self.recorder.timer_running.is_set() else 'idle'
        else:
            response = await super().handle_gsioc(data)

        return response

    async def QCMDRecord(self, tag_name: str = '', record_time: str | float = 0.0, sleep_time: str | float = 0.0) -> None:
        """Executes timer and sends record command to QCMD. Call by sending
            {"method": "record", {**kwargs}} over GSIOC.
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # wait the full time
        await self.recorder.record(tag_name, record_time, sleep_time)