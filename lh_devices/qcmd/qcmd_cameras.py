# lh_devices/qcmd/qcmd_cameras.py

import logging
import asyncio
from typing import Dict
from ..camera.camera import CameraDeviceBase, CameraCollectionBase
from ..camera.capturetools import get_input_devices

# If you kept acroname_hub.py in camera/, import it here
from ..camera.acroname_hub import USBHubManager 

class FIT0819(CameraDeviceBase):
    """Specific logical slot for DFRobot Endoscopes used in QCMD."""
    
    def __init__(self, device_id=None, name='DFRobot FIT0819 Endoscope'):
        super().__init__(name=name, device_id=device_id)

    async def get_info(self) -> dict:
        d = await super().get_info()
        
        # Ask the broker for available hardware addresses
        available_addresses = []
        if self.camera_collection is not None:
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
            # Assign the address based on the broker's current list
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


class FIT0819Collection(CameraCollectionBase):
    """Specific broker for finding FIT0819 cameras and mapping them via an Acroname Hub."""
    
    def refresh_hardware(self):
        names = get_input_devices("FriendlyName")
        paths = get_input_devices("DevicePath")
        
        new_cameras = {}
        for idx, (name, path) in enumerate(zip(names, paths)):
            if name == 'HD Camera': 
                try:
                    camera_id = path.split('mi_00#')[1].split('&0&0')[0]
                    new_cameras[camera_id] = idx
                except IndexError:
                    self.logger.warning(f"Failed to parse ID from path: {path}")
                
        self.physical_cameras = new_cameras
        self._verify_assignments()

    async def discover_via_acroname(self, hub_manager: USBHubManager, port_map: Dict[int, FIT0819], mount_delay: float = 3.0):
        """Orchestrates Acroname sequential power-up for FIT0819s."""
        ports = list(port_map.keys())
        hub_manager.disable_ports(ports)
        await asyncio.sleep(1.0)
        
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