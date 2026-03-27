import asyncio
import base64
import cv2
import datetime
import logging

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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




# =========== DFRobot FIT0819 Endoscope WebCam ==========

class FIT0819Collection(CameraCollectionBase):
    """
    Specific collection manager for DFRobot FIT0819 Endoscope cameras.
    Handles the unique DirectShow string parsing for these devices.
    """
    
    def refresh_hardware(self):
        names = get_input_devices("FriendlyName")
        paths = get_input_devices("DevicePath")
        
        new_cameras = {}
        for idx, (name, path) in enumerate(zip(names, paths)):
            if name == 'HD Camera': # This is the specific FIT0819 filter name
                try:
                    # Parse the specific DirectShow DevicePath string format
                    camera_id = path.split('mi_00#')[1].split('&0&0')[0]
                    new_cameras[camera_id] = idx
                except IndexError:
                    self.logger.warning(f"Found 'HD Camera' but failed to parse ID from path: {path}")
                
        self.physical_cameras = new_cameras
        self._verify_assignments()

    async def discover_via_acroname(self, hub_manager, port_map: Dict[int, 'CameraDeviceBase'], mount_delay: float = 3.0):
        """
        Generic Acroname discovery sequence. Relies on the subclass's 
        implementation of refresh_hardware() to identify deltas.
        """
        ports = list(port_map.keys())
        hub_manager.disable_ports(ports)
        await asyncio.sleep(1.0)
        
        # Establish baseline
        self.refresh_hardware()
        baseline = set(self.physical_cameras.keys())

        for port, slot in port_map.items():
            self.logger.info(f"Enabling port {port} for {slot.name}...")
            hub_manager.enable_port(port)
            await asyncio.sleep(mount_delay) 
            
            self.refresh_hardware()
            current = set(self.physical_cameras.keys())
            newly_discovered = current - baseline
            
            if len(newly_discovered) == 1:
                address = newly_discovered.pop()
                slot.address = address
                baseline.add(address)
                self.logger.info(f"Acroname Port {port} mapped to {slot.name} ({address})")
            elif len(newly_discovered) > 1:
                self.logger.warning(f"Port {port}: Multiple new cameras appeared! Found: {newly_discovered}")
            else:
                self.logger.warning(f"Port {port}: No new camera detected.")

class FIT0819(CameraDeviceBase):

    def __init__(self, device_id=None, name='DFRobot FIT0819 Endoscope'):

        super().__init__(name=name, device_id=device_id)

    async def get_info(self) -> dict:
        d = await super().get_info()
        
        # 1. Ask the broker for available hardware addresses (if attached)
        available_addresses = []
        if self.camera_collection is not None:
            # We can rely on the base broker having the physical_cameras dictionary
            available_addresses = list(self.camera_collection.physical_cameras.keys())

        d.update({'type': 'device',
                  'display': {'Address': 'Unassigned' if self.address is None else self.address},
                  'state': {'idle': self.idle,
                            'reserved': self.reserved,
                            'display': {'Address': self.address,
                                        'Timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S') if self.timestamp is not None else ''},
                            'image': self.image},
                  'controls': {'address': {'type': 'select',
                                           'options': [''] + available_addresses,
                                           'text': 'Camera address: ',
                                           'current': self.address if self.address is not None else ''},
                                'capture': {'type': 'button',
                                            'text': 'Capture image'}}})

        return d    

    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface"""

        await super().event_handler(command, data)

        if command == 'address':
            # 2. Assign the address based on the broker's current list
            if self.camera_collection is not None and data['index'] > 0:
                available_addresses = list(self.camera_collection.physical_cameras.keys())
                
                # Safety check to ensure the index is valid
                if data['index'] - 1 < len(available_addresses):
                    self.address = available_addresses[data['index'] - 1]
            else:
                self.address = None
                
            self.clear()
            await self.trigger_update()
            
        elif command == 'capture':
            # capture in background. Idle flags prevent multiple acquisitions
            asyncio.create_task(self.capture())
