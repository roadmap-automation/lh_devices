import asyncio
import base64
import datetime
import logging

from pathlib import Path
from typing import Coroutine

from aiohttp import ClientSession, ClientConnectionError
from urllib.parse import urlsplit

from ..camera.camera import CameraDeviceBase, FIT0819
from ..gilson.gsioc import GSIOCMessage
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
            try:
                await asyncio.sleep(wait_time)
            except asyncio.CancelledError:
                pass
            finally:
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
        self.cancel: asyncio.Queue = asyncio.Queue(1)

    def stop(self, hard: bool = False):
        """Stops timer immediately

        Args:
            hard (bool, optional): hard stop (does not do any follow-up actions). Defaults to False.
        """

        if self.cancel.empty():
            self.cancel.put_nowait(hard)

    async def wait(self, wait_time = 0) -> None:
        # reset the cancel queue
        while not self.cancel.empty():
            self.cancel.get_nowait()
        
        if await self.start(wait_time):
            self.cancel.put_nowait(False)
        else:
            self.cancel.put_nowait(True)

    async def record(self, tag_name: str = '', record_time: float = 0.0, sleep_time: float = 0.0) -> None:
        """Executes timer and sends record command to QCMD. Call by sending
            {"method": "record", {**kwargs}} over GSIOC.
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # calculate total wait time
        wait_time = record_time + sleep_time

        # wait the full time, stopping if a cancel signal is received
        wait_task = asyncio.create_task(self.wait(wait_time))
        hard_cancel: bool = await self.cancel.get()
        wait_task.cancel()

        if not hard_cancel:

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

        self.running_tasks = set()

        # Event that is triggered when all methods are completed
        self.event_finished: asyncio.Event = asyncio.Event()        

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:
        """Handles GSIOC message but deals with Q more robustly than the base method"""

        if data.data == 'Q':
            response = 'busy' if self.recorder.timer_running.is_set() else 'idle'
        else:
            response = await super().handle_gsioc(data)
            await self.trigger_update()

        return response

    async def QCMDRecord(self, tag_name: str = '', record_time: str | float = 0.0, sleep_time: str | float = 0.0) -> None:
        """Executes timer and sends record command to QCMD. Call by sending
            {"method": "record", {**kwargs}} over GSIOC.
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # wait the full time
        await self.recorder.record(tag_name, record_time, sleep_time)
        await self.trigger_update()

    def run_method(self, method: Coroutine) -> None:
        """Runs a coroutine method. Designed for complex operations with assembly hardware"""

        # clear finished event because something is now running
        self.event_finished.clear()

        # create a task and add to set to avoid garbage collection
        task = asyncio.create_task(method)
        logging.debug(f'Running task {task} from method {method}')
        self.running_tasks.add(task)

        # register callback upon task completion
        task.add_done_callback(self.method_complete_callback)

    def method_complete_callback(self, result: asyncio.Future) -> None:
        """Callback when method is complete

        Args:
            result (Any): calling method
        """

        self.running_tasks.discard(result)

        # if this was the last method to finish, set event_finished
        if len(self.running_tasks) == 0:
            self.event_finished.set()

    async def get_info(self) -> dict:
        d = await super().get_info()
        d.update({'type': 'device',
                  'state': {'idle': (not self.recorder.timer_running.is_set()),
                            'reserved': self.reserved},
                  'controls': {'interrupt': {'type': 'button',
                                             'text': 'Interrupt'},
                              'cancel': {'type': 'button',
                                             'text': 'Cancel'}}})
        
        return d    

    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'cancel':
            if self.recorder.cancel.empty():
                self.recorder.cancel.put_nowait(True)
        elif command == 'interrupt':
            if self.recorder.cancel.empty():
                self.recorder.cancel.put_nowait(False)

class QCMDRecorderDevicewithCamera(QCMDRecorderDevice):
    """QCMD recording device."""

    def __init__(self,
                 qcmd_address: str = 'localhost',
                 qcmd_port: int = 5011,
                 camera: CameraDeviceBase = CameraDeviceBase(None, None),
                 camera_save_path: Path = Path('~/Documents').expanduser(),
                 name='QCMDRecorderDevice') -> None:
        super().__init__(qcmd_address, qcmd_port, name)
        self.camera = camera
        self.camera_save_path = camera_save_path / datetime.datetime.now().strftime('%Y%m%d_%H.%M.%S')
        self.camera_save_path.mkdir(parents=True, exist_ok=True)

    async def QCMDRecord(self, tag_name: str = '', record_time: str | float = 0.0, sleep_time: str | float = 0.0) -> None:
        """Executes timer and sends record command to QCMD. Call by sending
            {"method": "record", {**kwargs}} over GSIOC.
        """

        record_time = float(record_time)
        sleep_time = float(sleep_time)

        # wait the full time
        await self.recorder.record(tag_name, record_time, sleep_time)
        await self.capture_and_write(tag_name)
        await self.trigger_update()

    async def capture_and_write(self, tag_name: str = ''):
        """Captures an image with the camera and writes the data to a timestamped file
        """
        await self.camera.capture()
        formatted_time = self.camera.timestamp.strftime('%Y%m%d_%H.%M.%S')
        self.logger.info(f'Captured image at {formatted_time} with tag name {tag_name}')
        try:
            fn = formatted_time + f'{tag_name}.png'
            with open(self.camera_save_path / fn, 'wb') as f:
                f.write(base64.b64decode(self.camera.image))
        except OSError:
            self.logger.warning(f'Invalid character in tag name {tag_name}, just printing timestamp')
            fn = formatted_time + '.png'
            with open(self.camera_save_path / fn, 'wb') as f:
                f.write(base64.b64decode(self.camera.image))

if __name__=='__main__':

    async def test_capture():
        qrd = QCMDRecorderDevicewithCamera('localhost', 5001, FIT0819(None))
        await qrd.capture_and_write()
        await qrd.recorder.session.close()

    asyncio.run(test_capture())