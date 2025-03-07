import time
import asyncio

from dataclasses import dataclass
from typing import Dict
from aiohttp import ClientSession, ClientConnectionError
from enum import Enum
from urllib.parse import urlsplit

from ..camera.camera import CameraDeviceBase, FIT0819
from ..device import DeviceBase, PollTimer
from ..assemblies import InjectionChannelBase
from ..methods import MethodBase
from ..multichannel import MultiChannelAssembly

class QCMDState(str, Enum):
    IDLE = 'idle'
    INITIALIZING = 'initializing'
    MEASURING = 'measuring'
    DISCONNECTED = 'disconnected'

class QCMDMeasurementDevice(DeviceBase):
    
    def __init__(self, http_address: str = 'http://localhost:5011/QCMD/0/', device_id: str = None, name='QCMDRecorder') -> None:

        DeviceBase.__init__(self, device_id=device_id, name=name)
        self.poll_interval = 1.0
        self.heartbeat_interval = 30.0

        url_parts = urlsplit(http_address)
        self.session = ClientSession(f'{url_parts.scheme}://{url_parts.netloc}')
        self.url_path = url_parts.path
        self.timeout = 10
        self.qcmd_status: str = QCMDState.DISCONNECTED

        # private attributes for monitoring progress
        self._tag: str | None = None
        self._start: float | None = None
        self._sleep_time: float = 0.0
        self._record_time: float = 0.0
        self._active_sleep_task: asyncio.Task | None = None

        self.result: dict | None = None

        self._heartbeat_task: asyncio.Task = None
        self._updating: bool = False

    async def initialize_device(self):

        self._heartbeat_task = asyncio.create_task(self.check_heartbeat())
        return await super().initialize_device()

    async def check_heartbeat(self) -> None:
        try:
            self.logger.debug(f'{self.name} started check_heartbeat')
            timer = PollTimer(self.heartbeat_interval, self.name + ' monitor PollTimer')
            await self.trigger_update()
            asyncio.create_task(timer.cycle())
            while True:
                await timer.wait_until_set()
                await asyncio.gather(timer.cycle(), self.trigger_update())
            
        except asyncio.CancelledError:
            self.logger.debug(f'{self.name} ended check_heartbeat')

    async def monitor(self) -> None:

        try:
            timer = PollTimer(self.poll_interval, self.name + ' monitor PollTimer')
            await self.trigger_update()
            asyncio.create_task(timer.cycle())
            while not self.idle:
                await timer.wait_until_set()
                await asyncio.gather(timer.cycle(), self.trigger_update())
            
            self.logger.debug(f'{timer.address} ended')
        except asyncio.CancelledError:
            self.logger.debug(f'{timer.address} cancelled')
        finally:
            await self.trigger_update()

    async def _record_with_monitor(self) -> Dict[str, float]:

        # calculate total wait time, set start time
        wait_time = self._record_time + self._sleep_time
        self._start = time.time()
        self.idle = False
        result = 0.0

        # start the monitor
        monitor = asyncio.create_task(self.monitor())

        # wait the full time and catch cancel
        self._active_sleep_task = asyncio.create_task(asyncio.sleep(wait_time))
        try:
            await self._active_sleep_task
        except asyncio.CancelledError:
            self.logger.info(f'{self.name} interrupted')
        finally:
            # cancel the monitor
            self.idle = True
            result = round(time.time() - self._start, 6)
            monitor.cancel()
            self._active_sleep_task = None

        return {'total time': result}

    def _remaining_time_formatted(self) -> tuple[str | None, str | None]:
        time_elapsed = time.time() - self._start if self._start is not None else 0.0

        sleep_time_remaining = max(0.0, self._sleep_time - time_elapsed)
        fmt_sleep = time.strftime('%H:%M:%S' if sleep_time_remaining // 3600 else '%M:%S', time.gmtime(sleep_time_remaining))

        record_time_remaining = max(0.0, min(self._record_time + self._sleep_time - time_elapsed, self._record_time))
        fmt_record = time.strftime('%H:%M:%S' if record_time_remaining // 3600 else '%M:%S', time.gmtime(record_time_remaining))

        return fmt_sleep, fmt_record

    async def _post(self, post_data: dict) -> dict | None:
        """Posts data to self.url

        Args:
            post_data (dict): data to post

        Returns:
            dict: response JSON
        """

        self.logger.debug(f'{self.session._base_url}{self.url_path} => {post_data}')

        response_json: dict | None = None

        # send an http request to QCMD server
        try:
            async with self.session.post(self.url_path, json=post_data, timeout=self.timeout) as resp:
                response_json = await resp.json()
                self.logger.debug(f'{self.session._base_url}{self.url_path} <= {response_json}')
        except (ConnectionRefusedError, ClientConnectionError):
            self.logger.error(f'request to {self.session._base_url}{self.url_path} failed: connection refused')
        except TimeoutError:
            self.logger.error(f'request to {self.session._base_url}{self.url_path} failed: timed out')

        return response_json

    def _reset_state(self):
        self._tag = None
        self._record_time = 0.0
        self._sleep_time = 0.0
        self._start = None        

    async def record(self, record_time: float = 0.0, sleep_time: float = 0.01) -> dict:
        """Recording 

        Args:
            record_time (float, optional): Time to record in seconds. Defaults to 0.0.
            sleep_time (float, optional): Time to sleep before recording in seconds. Defaults to 0.0.
        """

        self._record_time = float(record_time)
        self._sleep_time = float(sleep_time)

        if self._record_time + self._sleep_time > 0:
            result = await self._record_with_monitor()
        else:
            result = {'total time': 0.0}

        # send data request to QCMD interface
        post_data = {'command': 'get_data_slice',
                    'value': {'delta_t': self._record_time}}

        self._reset_state()

        post_result = await self._post(post_data)

        return result | post_result if post_result is not None else result

    async def record_tag(self, tag_name: str = '', record_time: float = 0.0, sleep_time: float = 0.0) -> dict:
        """Executes timer and sends record command to QCMD. Call by sending
            {"method": "record", {**kwargs}} over GSIOC.
        """

        self._record_time = float(record_time)
        self._sleep_time = float(sleep_time)
        self._tag = tag_name

        if self._record_time + self._sleep_time > 0:
            result = await self._record_with_monitor()
        else:
            result = {'total time': 0.0}

        # send data request to QCMD interface
        post_data = {'command': 'set_tag',
                    'value': {'tag': self._tag,
                            'delta_t': self._record_time}}

        self._reset_state()

        post_result = await self._post(post_data)

        return result | post_result if post_result is not None else result

    async def stop_collection(self) -> dict | None:
        """Sends stop signal to QCMD
        """

        post_data = {'command': 'stop',
                     'value': None}

        return await self._post(post_data)

    async def start_collection(self, description: str = '') -> dict | None:
        """Sends start signal to QCMD
        """

        post_data = {'command': 'start',
                     'value': {'description': description}}

        return await self._post(post_data)

    async def trigger_update(self):
        if not self._updating:
            asyncio.create_task(self.update_qcmd_status())
        return await super().trigger_update()

    async def update_qcmd_status(self) -> None:
        """Updates status from QCM server
        """
        self._updating = True
        post_data = {'command': 'get_status',
                     'value': None}
        
        result = await self._post(post_data)
        if result is None:
            self.qcmd_status = QCMDState.DISCONNECTED
        else:
            self.qcmd_status = result.get('result', QCMDState.DISCONNECTED)
        self._updating = False

    async def sleep(self, sleep_time: float = 0.0) -> float:
        """Sleep

        Args:
            sleep_time (float, optional): Time to sleep in seconds. Defaults to 0.0.

        Returns:
            float: actual time spent sleeping
        """

        self._sleep_time = float(sleep_time)

        result = await self._record_with_monitor()
        self._sleep_time = None

        return result

    def reserve(self):
        """Reserve channel"""
        self.reserved = True

    def release(self):
        """Release channel"""
        self.reserved = False

    def interrupt(self):
        """Interrupts current sleep"""

        if self._active_sleep_task is not None:
            self._active_sleep_task.cancel()

    async def get_info(self) -> dict:
        d = await super().get_info()
        sleep_time_remaining, record_time_remaining = self._remaining_time_formatted()
        d.update({'type': 'device',
                  'state': {'idle': self.idle,
                            'reserved': self.reserved,
                            'display': {'QCMD status': self.qcmd_status,
                                        'Tag': self._tag,
                                        'Sleep time remaining': sleep_time_remaining,
                                        'Record time remaining': record_time_remaining}},
                  'controls': {'interrupt': {'type': 'button',
                                             'text': 'Interrupt'},
                               'set_sleep_time': {'type': 'textbox',
                                                  'text': 'Set sleep time (s): '},
                               'set_record_time': {'type': 'textbox',
                                                  'text': 'Set record time (s): '}}})
        
        return d    

    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'interrupt':
            self.interrupt()
        elif command == 'set_sleep_time':
            self._sleep_time = float(data['value'])
            await self.trigger_update()
        elif command == 'set_record_time':
            self._record_time = float(data['value'])
            await self.trigger_update()

class QCMDMeasurementChannel(InjectionChannelBase):

    def __init__(self, qcmd: QCMDMeasurementDevice, name='QCMD Channel'):
        super().__init__([qcmd], name=name)

        self.methods.update({'QCMDRecord': self.QCMDRecord(qcmd),
                'QCMDRecordTag': self.QCMDRecordTag(qcmd),
                'QCMDSleep': self.QCMDSleep(qcmd),
                'QCMDAcceptTransfer': self.QCMDAcceptTransfer(qcmd),
                'QCMDStart': self.QCMDStart(qcmd),
                'QCMDStop': self.QCMDStop(qcmd)})
        
        self.qcmd = qcmd

    async def get_info(self):
        d = await super().get_info()
        d['controls'] = d['controls'] | {'add_tag': {'type': 'textbox',
                                                     'text': 'Add tag: '}}
        
        return d

    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'add_tag':
            self.run_method('QCMDRecordTag', dict(tag_name=data['value'],
                                                  record_time=self.qcmd._record_time,
                                                  sleep_time=self.qcmd._sleep_time))

    class QCMDMethodBase(MethodBase):

        def __init__(self, device: QCMDMeasurementDevice):
            super().__init__(devices=[device])
            self.qcmd = device

        async def start(self, **kwargs):
            self.reserve_all()
            self.idle = False
            result = await super().start(**kwargs)
            self.idle = True
            self.release_all()

            await self.trigger_update()

            return result

    class QCMDSleep(QCMDMethodBase):

        @dataclass
        class MethodDefinition(MethodBase.MethodDefinition):

            name: str = 'QCMDSleep'
            sleep_time: float = 0.0

        async def run(self, **kwargs):

            method = self.MethodDefinition(**kwargs)
            self.reserve_all()
            self.logger.info(f'{self.name}: Starting sleep for {method.sleep_time} s')
            result = await self.qcmd.sleep(method.sleep_time)
            self.logger.info(f'{self.name}: Actual time slept {self.qcmd.result["total time"]} s')
            self.release_all()

            return result

    class QCMDRecord(QCMDMethodBase):

        @dataclass
        class MethodDefinition(MethodBase.MethodDefinition):
            """Recording 

            Args:
                record_time (float, optional): Time to record in seconds. Defaults to 0.0.
                sleep_time (float, optional): Time to sleep before recording in seconds. Defaults to 0.0.
            """
            name: str = 'QCMDRecord'
            record_time: float = 0.0
            sleep_time: float = 0.0

        async def run(self, **kwargs):

            method = self.MethodDefinition(**kwargs)
            self.reserve_all()
            result = await self.qcmd.record(method.record_time, method.sleep_time)
            self.release_all()

            return result

    class QCMDRecordTag(QCMDMethodBase):

        @dataclass
        class MethodDefinition(MethodBase.MethodDefinition):
            """Sets a tag after sleep_time + record_time

            Args:
                tag_name (str, optional): Tag name
                record_time (float, optional): Time to record in seconds. Defaults to 0.0.
                sleep_time (float, optional): Time to sleep before recording in seconds. Defaults to 0.0.
            """
            name: str = 'QCMDRecordTag'
            tag_name: str = ''
            record_time: float = 0.0
            sleep_time: float = 0.0

        async def run(self, **kwargs):

            method = self.MethodDefinition(**kwargs)
            self.reserve_all()
            result = await self.qcmd.record_tag(method.tag_name, method.record_time, method.sleep_time)
            self.release_all()

            return result

    class QCMDAcceptTransfer(QCMDMethodBase):

        @dataclass
        class MethodDefinition(MethodBase.MethodDefinition):

            name: str = 'QCMDAcceptTransfer'
            contents: str = ''

        async def run(self, **kwargs):

            method = self.MethodDefinition(**kwargs)
            self.reserve_all()
            self.logger.info(f'{self.name}: Received transfer of material {method.contents}')
            result = {'contents': method.contents}
            self.release_all()

            return result

    class QCMDStop(QCMDMethodBase):

        @dataclass
        class MethodDefinition(MethodBase.MethodDefinition):

            name: str = 'QCMDStop'

        async def run(self, **kwargs):

            self.reserve_all()
            result = await self.qcmd.stop_collection()
            self.release_all()

            return result

    class QCMDStart(QCMDMethodBase):

        @dataclass
        class MethodDefinition(MethodBase.MethodDefinition):

            name: str = 'QCMDStart'
            description: str = ''

        async def run(self, **kwargs):

            method = self.MethodDefinition(**kwargs)
            self.reserve_all()
            result = await self.qcmd.start_collection(method.description)
            
            # wait for collection
            try:
                timer = PollTimer(self.qcmd.poll_interval, self.name + ' monitor PollTimer')
                await self.qcmd.trigger_update()
                asyncio.create_task(timer.cycle())
                while not (self.qcmd.qcmd_status == QCMDState.MEASURING):
                    await timer.wait_until_set()
                    await asyncio.gather(timer.cycle(), self.qcmd.trigger_update())
                    if self.qcmd.qcmd_status == QCMDState.DISCONNECTED:
                        await self.throw_error('openQCM software not responding; waiting for error to be cleared', critical=False)
                
                self.logger.debug(f'{timer.address} ended')
            except asyncio.CancelledError:
                self.logger.debug(f'{timer.address} cancelled')
            finally:
                await self.qcmd.trigger_update()

            self.release_all()

            return result
        
class QCMDMeasurementChannelwithCamera(QCMDMeasurementChannel):

    def __init__(self, qcmd: QCMDMeasurementDevice, camera: CameraDeviceBase, name='QCMD Channel with Camera'):
        self.camera = camera
        super().__init__(qcmd, name)
        self.devices += [camera]

    class QCMDMethodBasewithCamera(QCMDMeasurementChannel.QCMDMethodBase):

        def __init__(self, device: QCMDMeasurementDevice, camera: CameraDeviceBase):
            super().__init__(device)
            self.camera = camera

    class QCMDCaptureImage(QCMDMethodBasewithCamera):

        @dataclass
        class MethodDefinition(MethodBase.MethodDefinition):

            name: str = 'QCMDCaptureImage'
            description: str = ''

        async def run(self, **kwargs):

            method = self.MethodDefinition(**kwargs)
            self.reserve_all()

            self.camera.capture()

            self.release_all()

            return {'image': self.camera.image}

class QCMDMultiChannelMeasurementDevice(MultiChannelAssembly):
    """QCMD recording device simultaneously recording on multiple QCMD instruments"""

    def __init__(self,
                 qcmd_address: str = 'localhost',
                 qcmd_port: int = 5011,
                 n_channels: int = 1,
                 qcmd_ids: list | None = None,
                 database_path: str | None = None,
                 name='MultiChannel QCMD Measurement Device') -> None:

        if qcmd_ids is not None:
            channels = [QCMDMeasurementChannelwithCamera(QCMDMeasurementDevice(f'http://{qcmd_address}:{qcmd_port}/QCMD/id/{qcmd_id}/',
                                                                          name=f'QCMD Measurement Device {i}, Serial Number {qcmd_id}'),
                                                         camera=FIT0819(None),
                                                    name=f'QCMD Measurement Channel {i}')
                                for i, qcmd_id in enumerate(qcmd_ids)]
        else:
            channels = [QCMDMeasurementChannelwithCamera(QCMDMeasurementDevice(f'http://{qcmd_address}:{qcmd_port}/QCMD/{i}/',
                                                                          name=f'QCMD Measurement Device {i}'),
                                                         camera=None,
                                                    name=f'QCMD Measurement Device {i}')
                                for i in range(n_channels)]

        super().__init__(channels=channels,
                         assemblies=[],
                         database_path=database_path,
                         name=name)
