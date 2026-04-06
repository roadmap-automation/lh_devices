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

        # disable Aether (direct connections only)
        config = self.hub.getConfig()
        config.value.enabled = False
        self.hub.setConfig(config.value)

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
    