import logging
import asyncio
from typing import Dict, List
from .camera import CameraDeviceBase # Importing your camera classes

try:
    import brainstem
    from brainstem.result import Result
except ImportError:
    brainstem = None
    Result = None
    logging.getLogger(__name__).warning("BrainStem library not found. Acroname hub control will not be available.")

class USBHubManager:
    """
    Manager for the Acroname USBHub3+ using the brainstem API.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.hub = None
        self.connected = False

    def connect(self) -> bool:
        if brainstem is None:
            self.logger.error("Cannot connect to USB Hub: brainstem library missing.")
            return False

        self.hub = brainstem.stem.USBHub3p()
        result = self.hub.discoverAndConnect(brainstem.link.Spec.USB)
        
        if result == Result.NO_ERROR:
            self.connected = True
            self.logger.info("Successfully connected to Acroname USBHub3+.")
            return True
        else:
            self.connected = False
            self.logger.error(f"Failed to connect to Acroname hub. Error code: {result}")
            
            # Kill the dangling threads immediately on failure ---
            self.hub.disconnect() 
            del self.hub
            self.hub = None
            return False

    def disconnect(self):
        """Disconnects and destroys the hub object to free threads."""
        if self.hub is not None:
            self.hub.disconnect()
            del self.hub
            self.hub = None
            self.connected = False
            self.logger.info("Disconnected and destroyed Acroname USBHub3+ object.")

    def enable_port(self, port_index: int) -> bool:
        if not self.connected: return False
        
        # The methods take the port_index directly as the argument
        data_res = self.hub.usb.setDataEnable(port_index)
        pwr_res = self.hub.usb.setPowerEnable(port_index)
        
        return data_res == Result.NO_ERROR and pwr_res == Result.NO_ERROR

    def disable_port(self, port_index: int) -> bool:
        if not self.connected: return False
        
        # The methods take the port_index directly as the argument
        data_res = self.hub.usb.setDataDisable(port_index)
        pwr_res = self.hub.usb.setPowerDisable(port_index)
        
        return data_res == Result.NO_ERROR and pwr_res == Result.NO_ERROR

    def disable_ports(self, port_indices: List[int]) -> bool:
        if not self.connected: return False
        all_successful = True
        for port in port_indices:
            if not self.disable_port(port):
                all_successful = False
        return all_successful


class AcronameCameraDiscoveryManager:
    """
    Orchestrates the sequential powering of USB ports to identify and assign
    camera hardware addresses to generic camera instances.
    """
    def __init__(self, hub_manager: USBHubManager, port_to_camera_map: Dict[int, CameraDeviceBase]):
        """
        Args:
            hub_manager: An initialized and connected USBHubManager.
            port_to_camera_map: A dictionary mapping hub port index to the specific Camera instance.
                                Example: {0: camera_ch1, 1: camera_ch2}
        """
        self.logger = logging.getLogger(__name__)
        self.hub = hub_manager
        self.port_map = port_to_camera_map

    async def discover_and_assign(self, mount_delay: float = 3.0):
        """
        Sequentially powers up ports, checks for new camera IDs, and assigns them.
        """
        if not self.hub.connected:
            self.logger.error("Hub not connected. Cannot perform camera discovery.")
            return

        ports_to_manage = list(self.port_map.keys())
        self.logger.info(f"Starting sequential discovery on ports: {ports_to_manage}")

        # Step 1: Turn off all managed camera ports to get a clean slate
        self.hub.disable_ports(ports_to_manage)
        await asyncio.sleep(1.0) # Brief pause for the OS to drop the connections

        # Step 2: Establish the baseline of currently connected cameras (e.g. laptop webcam, etc.)
        baseline_list = DFRobotCameraList()
        baseline_addresses = set(baseline_list.cameras.keys())

        # Step 3: Sequentially enable and assign
        for port, camera_instance in self.port_map.items():
            self.logger.info(f"Enabling port {port} for {camera_instance.name}...")
            self.hub.enable_port(port)
            
            # Wait for Windows/DirectShow to mount the new USB video device
            # 3 seconds is usually safe, but might need tweaking depending on the PC
            await asyncio.sleep(mount_delay)

            # Check the new list
            new_list = DFRobotCameraList()
            current_addresses = set(new_list.cameras.keys())

            # Find the difference
            newly_discovered = current_addresses - baseline_addresses

            if len(newly_discovered) == 1:
                discovered_address = newly_discovered.pop()
                self.logger.info(f"Discovered camera ID {discovered_address} on port {port}. Assigning to {camera_instance.name}.")
                
                # Assign to the camera
                camera_instance.address = discovered_address
                # Also ensure the camera has an up-to-date list object if it needs it internally
                camera_instance.camera_list = new_list 
                
                # Update baseline so we don't 'discover' this one again on the next loop
                baseline_addresses.add(discovered_address)
            
            elif len(newly_discovered) > 1:
                self.logger.warning(f"Port {port} enabled, but multiple new cameras appeared! Expected 1. Found: {newly_discovered}. Skipping assignment.")
            else:
                self.logger.warning(f"Port {port} enabled, but no new camera was detected. Skipping assignment.")

        self.logger.info("Camera discovery and assignment complete.")

    async def discover_via_acroname(self, hub_manager, port_map: dict[int, CameraDeviceBase]):
        """
        Sequentially powers ports to deterministically bind addresses to specific slots.
        port_map: {hub_port_index: logical_camera_slot}
        """
        # 1. Turn off all mapped ports
        hub_manager.disable_ports(list(port_map.keys()))
        await asyncio.sleep(1.0)
        
        # 2. Establish baseline of non-hub cameras
        self.refresh_hardware()
        baseline = set(self.physical_cameras.keys())

        # 3. Sequentially power and bind
        for port, slot in port_map.items():
            hub_manager.enable_port(port)
            await asyncio.sleep(3.0) # Wait for OS mount
            
            self.refresh_hardware()
            current = set(self.physical_cameras.keys())
            newly_discovered = current - baseline
            
            if len(newly_discovered) == 1:
                address = newly_discovered.pop()
                slot.address = address
                baseline.add(address)
                self.logger.info(f"Acroname Port {port} mapped to {slot.name} ({address})")