import asyncio
import base64
import cv2
import datetime
from dataclasses import dataclass, field
from ..device import DeviceBase, DeviceError
from .capturetools import get_input_devices

@dataclass
class CameraListBase:
    cameras: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self.refresh()

    def refresh(self):
        return []

class CameraDeviceBase(DeviceBase):

    def __init__(self, address: str, camera_list: CameraListBase, device_id = None, name = None):
        super().__init__(device_id, name)

        self.address = address
        self.camera_list = camera_list
        self.raw_image: cv2.typing.MatLike = None
        self.image: str = None
        self.timestamp: datetime.datetime = None
        self.properties: dict[int, float] = {}

    def check_if_present(self) -> bool:
        """Checks if camera is present

        Raises:
            DeviceError: Camera not present
        """

        self.camera_list.refresh()
        if self.address not in self.camera_list.cameras.keys():
            return False
        
        return True
    
    def _capture(self) -> None:
        """Capture an image using camera properties
        """

        if self.check_if_present():

            # open camera
            cam = cv2.VideoCapture(self.camera_list.cameras[self.address], apiPreference=cv2.CAP_DSHOW)

            # set properties
            for prop, value in self.properties.items():
                cam.set(prop, value)

            # capture image
            if cam.isOpened():
                result, image = cam.read()
                self.timestamp = datetime.datetime.now()
            else:
                self.clear()
                self.logger.warning(f'{self.name} Camera at address {self.address} cannot be opened')

            # render image as base64 string
            self.raw_image = image
            self.image = base64.b64encode(cv2.imencode('.png', image)[1].tobytes()).decode("utf-8")

            # read properties
            for prop in self.properties.keys():
                self.properties[prop] = cam.get(prop)

            # release camera and return image and read properties
            cam.release()

        else:

            self.clear()
  
    async def capture(self) -> None:
        """Async version of _capture
        """

        if self.idle:
            self.idle = False
            await self.trigger_update()

            await asyncio.to_thread(self._capture)

            self.idle = True
            await self.trigger_update()

    def clear(self) -> None:
        self.image = None
        self.raw_image = None
        self.timestamp = None

# =========== DFRobot FIT0819 Endoscope WebCam ==========

@dataclass
class DFRobotCameraList(CameraListBase):

    def refresh(self):

        names: list[str] = get_input_devices("FriendlyName")
        paths: list[str] = get_input_devices("DevicePath")
        for idx, (name, path) in enumerate(zip(names, paths)):
            if name == 'HD Camera':
                camera_id = path.split('mi_00#')[1].split('&0&0')[0]
                self.cameras[camera_id] = idx

class FIT0819(CameraDeviceBase):

    def __init__(self, address: str, device_id=None, name='DFRobot FIT0819 Endoscope'):
        camera_list = DFRobotCameraList()

        # default to first camera in list if address is None
        if address is None:
            if len(camera_list.cameras):
                address = list(camera_list.cameras.keys())[0]

        super().__init__(address, camera_list, device_id, name)

    async def get_info(self) -> dict:
        d = await super().get_info()
        d.update({'type': 'device',
                  'display': {'Address': '' if self.address is None else self.address},
                  'state': {'idle': self.idle,
                            'reserved': self.reserved,
                            'display': {'Address': self.address,
                                        'Timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp is not None else ''},
                            'image': self.image},
                  'controls': {'address': {'type': 'select',
                                           'options': [''] + list(self.camera_list.cameras.keys()),
                                           'text': 'Camera address: ',
                                           'current': self.address if self.address is not None else ''},
                                'capture': {'type': 'button',
                                            'text': 'Capture image'}}})

        return d    

    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'address':
            self.address = list(self.camera_list.cameras.keys())[data['index'] - 1] if data['index'] > 0 else None
            self.clear()
            await self.trigger_update()
        elif command == 'capture':
            # capture in background. Idle flags prevent multiple acquisitions
            asyncio.create_task(self.capture())
