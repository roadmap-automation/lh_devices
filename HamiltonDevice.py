from typing import Tuple, List
from HamiltonComm import HamiltonSerial
from valve import ValveBase
from connections import Node

class HamiltonBase:
    """Base class for Hamilton multi-valve positioner (MVP) and syringe pump (PSD) devices.

        Requires:
        serial_instance -- HamiltonSerial instance for communication
        address -- single character string from '0' to 'F' corresponding to the physical
                    address switch position on the device. Automatically converted to the 
                    correct address code.
     
       """

    def __init__(self, serial_instance: HamiltonSerial, address: str) -> None:
        
        self.serial = serial_instance
        self.idle = True
        self.busy_code = '@'
        self.idle_code = '`'
        self.address = address
        self.address_code = chr(int(address, base=16) + int('31', base=16))

    def get_nodes(self) -> List[Node]:

        return []

    async def initialize(self) -> None:
        pass

    async def query(self, cmd: str) -> str:
        """
        Wraps self.serial.query with a trimmed response
        """

        response = await self.serial.query(self.address_code, cmd)
        
        if response:
            return response[2:-1]
        else:
            return None

    async def run_until_idle(self, cmd: str) -> Tuple[str, str]:
        """
        Sends from serial connection and waits until idle
        """

        self.idle = False
        response = await self.query(cmd + 'R')
        if response is not None:

            status_byte = response[0]
            if len(response) > 1:
                response = response[1:]

            error = self.parse_status_byte(status_byte)
            while (not self.idle) & (not error):
                error = await self.update_status()

            return response, error
        else:
            return None, None

    async def update_status(self) -> str | None:
        """
        Polls the status of the device using 'Q'
        """

        response = await self.query('Q')
        error = self.parse_status_byte(response)

        return error

    def parse_status_byte(self, c: str) -> str | None:
        """
        Parses status byte
        """

        error = None
        match c:
            case self.busy_code:
                self.idle = False
            case self.idle_code:
                self.idle = True
            case 'b':
                error = 'Bad command'
            case _ :
                error = f'Error code: {c}'

        if error:
            self.idle = True

        return error

class HamiltonValvePositioner(HamiltonBase):
    """Hamilton MVP4 device
    """

    def __init__(self, serial_instance: HamiltonSerial, address: str, valve: ValveBase) -> None:
        super().__init__(serial_instance, address)

        self.valve = valve
        self.initialized = False

    def get_nodes(self) -> List[Node]:
        
        return self.valve.nodes

    async def initialize(self) -> None:

        # TODO: might be Y!!! Depends on whether left or right is facing front or back.
        response, error = await self.run_until_idle('Z')
        if not error:
            self.initialized = True
        else:
            print(f'Initialization error {error}')

    async def move_valve(self, position: int) -> None:
        """Moves to a particular valve position. See specific valve documentation.

        Args:
            position (int): position to move the valve to
        """

        initial_value = self.valve.position
        
        # this checks for errors
        self.valve.move(position)

        response, error = await self.run_until_idle(f'I{position}')
        if error:
            print(f'Move error {error}')
            self.valve.position = initial_value


class HamiltonSyringePump(HamiltonValvePositioner):
    """Hamilton syringe pump device. Includes both a syringe motor and a built-in valve positioner.
    """

    def __init__(self,
                 serial_instance: HamiltonSerial,
                 address: str,
                 valve: ValveBase,
                 syringe_volume: float = 5000,
                 high_resolution = False
                 ) -> None:
        super().__init__(serial_instance, address, valve)

        # Syringe volume in uL
        self.syringe_volume = syringe_volume

        # default high resolution mode is False
        self._high_resolution = high_resolution

    async def initialize(self) -> None:
        await self.set_high_resolution(self._high_resolution)
        return await super().initialize()

    async def set_high_resolution(self, high_resolution: bool) -> None:
        """Turns high resolution mode on or off

        Args:
            high_resolution (bool): turn high resolution on (True) or off (False)
        """
        
        response, error = await self.run_until_idle(f'N{int(high_resolution)}')
        if error:
            print(f'Error setting resolution: {error}')
        else:
            self._high_resolution = high_resolution

    def _full_stroke(self) -> int:
        """Calculates syringe stroke (# half steps for full volume) based on resolution mode

        Returns:
            float: stroke in half steps
        """

        return 48000 if self._high_resolution else 6000

    def _speed_code(self, desired_flow_rate: float) -> int:
        """Calculates speed code (parameter V, see PSD/4 manual Appendix H) based on desired
            flow rate and syringe parameters

        Args:
            desired_flow_rate (float): desired flow rate in uL / s

        Returns:
            int: V (half-steps per second)
        """

        minV, maxV = 2, 10000

        calcV = float(desired_flow_rate * 6000) / self.syringe_volume

        if calcV < minV:
            print(f'Warning: clipping desired flow rate {desired_flow_rate} to lowest possible value {self._flow_rate(minV)}')
            return minV
        elif calcV > maxV:
            print(f'Warning: clipping desired flow rate {desired_flow_rate} to highest possible value {self._flow_rate(maxV)}')
            return maxV
        else:
            return round(calcV)
        
    def _flow_rate(self, V: int) -> float:
        """Calculates actual flow rate from speed code parameter (V)

        Args:
            V (float): speed code in half-steps / second

        Returns:
            float: flow rate in uL / s
        """

        return float(V * self.syringe_volume) / self._full_stroke()
    
    def _stroke_length(self, desired_volume: float) -> int:
        """Calculates stroke length in steps

        Args:
            desired_volume (float): aspirate or dispense volume in uL

        Returns:
            int: stroke length in number of motor steps
        """

        return round(desired_volume * (self._full_stroke() / 2) / self.syringe_volume)

    async def get_syringe_position(self) -> int:
        """Reads absolute position of syringe

        Returns:
            int: absolute position of syringe in steps
        """

        response = await self.query('?')
        
        return int(response[1:])


    async def aspirate(self, volume: float, flow_rate: float) -> None:
        """Aspirate (Pick-up)

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        stroke_length = self._stroke_length(volume)
        current_position = await self.get_syringe_position()
        max_position = self._full_stroke() / 2
        #print(f'Stroke length: {stroke_length} out of full stroke {self._full_stroke() / 2}')

        if max_position < (stroke_length + current_position):
            print(f'Invalid syringe move from current position {current_position} with stroke length {stroke_length} and maximum position {max_position}')
        else:
            V = self._speed_code(flow_rate)
            #print(f'Speed: {V}')

            response, error = await self.run_until_idle(f'V{V}P{stroke_length}')
            if error:
                print(f'Syringe move error {error}')

    async def dispense(self, volume: float, flow_rate: float) -> None:
        """Dispense

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        stroke_length = self._stroke_length(volume)
        #print(f'Stroke length: {stroke_length} out of full stroke {self._full_stroke() / 2}')
        current_position = await self.get_syringe_position()

        if (current_position - stroke_length) < 0:
            print(f'Invalid syringe move from current position {current_position} with stroke length {stroke_length} and minimum position 0')
        else:
            V = self._speed_code(flow_rate)

            response, error = await self.run_until_idle(f'V{V}D{stroke_length}')
            if error:
                print(f'Syringe move error {error}')

    async def home(self) -> None:
        """Home syringe.
        """

        response, error = await self.run_until_idle(f'A0')
        if error:
            print(f'Syringe homing error {error}')