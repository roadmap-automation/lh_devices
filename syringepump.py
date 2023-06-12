import asyncio
from typing import List

from hamilton import HamiltonSerial
from hamiltonvalve import Port, TValve, HamiltonValvePositioner, ValveBase

class SyringeYValve(TValve):
    """Y valve to sit atop syringe pump. Port 0 is down, and ports are number clockwise.
        Implementation is that of a 3-port T valve.
    """

    def __init__(self, position: int = 1, ports: List[Port] = []) -> None:
        super().__init__(3, position, ports)

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
        
        self._high_resolution = high_resolution
        response, error = await self.run_until_idle(f'N{int(high_resolution)}')
        if error:
            print(f'Error setting resolution: {error}')

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
        print(f'Stroke length: {stroke_length} out of full stroke {self._full_stroke() / 2}')

        if max_position < (stroke_length + current_position):
            print(f'Invalid syringe move from current position {current_position} with stroke length {stroke_length} and maximum position {max_position}')
        else:
            V = self._speed_code(flow_rate)
            print(f'Speed: {V}')

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
        print(f'Stroke length: {stroke_length} out of full stroke {self._full_stroke() / 2}')
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