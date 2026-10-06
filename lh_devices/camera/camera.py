import asyncio
import base64
import cv2
import datetime
import logging


from ..device import DeviceBase, DeviceError
from .capturetools import get_input_devices

class CameraCollectionBase:
    """
    Abstract broker that manages physical camera hardware and binds 
    them to logical camera slots.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.physical_cameras: dict[str, int] = {} # {address: cv2_index}
        self.logical_slots: list['CameraDeviceBase'] = []

    def register_slot(self, camera: 'CameraDeviceBase'):
        """Registers a logical camera slot with the broker."""
        self.logical_slots.append(camera)
        camera.camera_collection = self

    def refresh_hardware(self):
        """
        Polls the OS for current video devices.
        MUST be implemented by specific camera collection subclasses.
        """
        raise NotImplementedError("Subclasses must implement refresh_hardware()")

    def is_address_valid(self, address: str) -> bool:
        return address in self.physical_cameras

    def get_index(self, address: str) -> int:
        return self.physical_cameras.get(address)

    def _verify_assignments(self):
        """Unbinds slots if their physical camera was unplugged."""
        for slot in self.logical_slots:
            if slot.address is not None and slot.address not in self.physical_cameras:
                self.logger.warning(f"Hardware for {slot.name} lost. Unbinding.")
                slot.address = None
                slot.clear()

    def auto_assign_freely(self):
        """Assigns any unassigned physical camera to any empty slot."""
        self.refresh_hardware()
        assigned_addresses = {slot.address for slot in self.logical_slots if slot.address is not None}
        available_addresses = set(self.physical_cameras.keys()) - assigned_addresses
        
        for slot in self.logical_slots:
            if slot.address is None and available_addresses:
                new_address = available_addresses.pop()
                slot.address = new_address
                self.logger.info(f"Auto-assigned {new_address} to {slot.name}")

class CameraDeviceBase(DeviceBase):
    """A logical camera slot that delegates hardware requests to a Collection broker."""
    def __init__(self, name=None, device_id=None):
        super().__init__(device_id, name)
        
        self.address: str | None = None 
        self.camera_collection: CameraCollectionBase | None = None
        
        self.raw_image: cv2.typing.MatLike = None
        self.image: str = None
        self.timestamp: datetime.datetime = None
        self.properties: dict[int, float] = {}

    def is_hardware_assigned(self) -> bool:
        """Checks if a physical camera is currently bound to this slot."""
        if self.address is None or self.camera_collection is None:
            return False
        return self.camera_collection.is_address_valid(self.address)

    def check_if_present(self) -> bool:
        """
        Checks if a physical camera is currently bound to this logical slot.
        Replaces the old refresh-heavy logic.
        """
        if self.address is None or self.camera_collection is None:
            return False
            
        return self.camera_collection.is_address_valid(self.address)

    def _capture(self) -> None:
        """Capture an image using camera properties"""

        if self.check_if_present():
            # 1. Ask the broker for the physical OpenCV integer index
            camera_index = self.camera_collection.get_index(self.address)

            if camera_index is None:
                self.clear()
                self.logger.error(f'{self.name} has a valid address but the broker returned no index.')
                return

            # 2. Open camera using the broker-provided index
            cam = cv2.VideoCapture(camera_index, apiPreference=cv2.CAP_DSHOW)

            # set properties
            for prop, value in self.properties.items():
                cam.set(prop, value)

            # capture image
            if cam.isOpened():
                result, image = cam.read()
                
                if result:
                    self.timestamp = datetime.datetime.now()
                    # render image as base64 string
                    self.raw_image = image
                    self.image = base64.b64encode(cv2.imencode('.png', image)[1].tobytes()).decode("utf-8")

                    # read properties
                    for prop in self.properties.keys():
                        self.properties[prop] = cam.get(prop)
                else:
                    self.clear()
                    self.logger.warning(f'{self.name} opened index {camera_index} but failed to read a frame.')
            else:
                self.clear()
                self.logger.warning(f'{self.name} Camera at index {camera_index} cannot be opened')

            # release camera
            cam.release()

        else:
            self.clear()
            # Optional: Log this at debug level so it doesn't spam your console if a channel is intentionally empty
            self.logger.debug(f'{self.name} ignored capture command: no physical hardware assigned.')
 
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


