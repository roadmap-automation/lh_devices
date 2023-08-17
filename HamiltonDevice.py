from typing import Tuple, List
import asyncio
import logging
from HamiltonComm import HamiltonSerial
from valve import ValveBase
from connections import Node

class PollTimer:
    """Async timer for polling delay

    Usage:
    while <condition>:
        # starts timer at the same time as the future
        await asyncio.gather(<future to run periodically>, timer.cycle())

        # ensures that the timer has expired before the future is run again
        await timer.wait_until_set()
    """
    def __init__(self, poll_delay: float, address: str) -> None:
        """Initializate timer

        Args:
            poll_delay (float): poll delay in seconds
            address (str): address code of parent class. Used only for logging
        """
        self.poll_delay = poll_delay
        self.address = address

        # internal event to track polling expiration
        self.expired = asyncio.Event()

    async def cycle(self) -> None:
        """Cycle the timer
        """

        # clear the event
        self.expired.clear()
        logging.debug(f'timer {self.address} started')

        # sleep the required delay
        await asyncio.sleep(self.poll_delay)
        logging.debug(f'timer {self.address} ended')

        # set the event
        self.expired.set()

    async def wait_until_set(self) -> None:
        """Waits until the internal timer has expired"""
        await self.expired.wait()

class HamiltonBase:
    """Base class for Hamilton multi-valve positioner (MVP) and syringe pump (PSD) devices.

        Requires:
        serial_instance -- HamiltonSerial instance for communication
        address -- single character string from '0' to 'F' corresponding to the physical
                    address switch position on the device. Automatically converted to the 
                    correct address code.
     
       """

    def __init__(self, serial_instance: HamiltonSerial, address: str, name=None) -> None:
        
        self.serial = serial_instance
        self.name = name
        self.idle = True
        self.busy_code = '@'
        self.idle_code = '`'
        self.poll_delay = 0.1   # Hamilton-recommended 100 ms delay when polling
        self.address = address
        self.address_code = chr(int(address, base=16) + int('31', base=16))
        self.response_queue: asyncio.Queue = asyncio.Queue()

    def __repr__(self):

        if self.name:
            return self.name
        else:
            return object.__repr__(self)

    def get_nodes(self) -> List[Node]:

        return []

    async def initialize(self) -> None:

        await self.run_until_idle(self.initialize_device())

    async def initialize_device(self) -> None:
        pass

    async def query(self, cmd: str) -> Tuple[str | None, str | None]:
        """Adds command to command queue and waits for response"""
        
        # push command to command queue
        await self.serial.query(self.address_code, cmd, self.response_queue)

        # wait for response
        response = await self.response_queue.get()
        
        # process response
        if response:
            response = response[2:-1]
            response, error = self.parse_status_byte(response)
            return response, error
        else:
            return None, None

    async def poll_until_idle(self) -> None:
        """Polls device until idle

        Returns:
            str: error string
        """

        timer = PollTimer(self.poll_delay, self.address_code)

        while (not self.idle):
            # run update_status and start the poll_delay timer
            await asyncio.gather(self.update_status(), timer.cycle())

            # wait until poll_delay timer has ended before asking for new status.
            await timer.wait_until_set()

    async def run_until_idle(self, cmd: asyncio.Future) -> None:
        """
        Sends from serial connection and waits until idle
        """

        self.idle = False
        await cmd
        await self.poll_until_idle()

    async def update_status(self) -> None:
        """
        Polls the status of the device using 'Q'
        """

        _, error = await self.query('Q')

        # TODO: Handle error
        if error:
            logging.error(f'{self}: Error in update_status: {error}')

    def parse_status_byte(self, response: str) -> str | None:
        """
        Parses status byte
        """
        c = response[0]

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

        return response.split(c, 1)[1], error

class HamiltonValvePositioner(HamiltonBase):
    """Hamilton MVP4 device
    """

    def __init__(self, serial_instance: HamiltonSerial, address: str, valve: ValveBase, name=None) -> None:
        super().__init__(serial_instance, address, name)

        self.valve = valve
        self.initialized = False

    def get_nodes(self) -> List[Node]:
        
        return self.valve.nodes

    async def initialize(self) -> None:
        await super().initialize()
        await self.set_valve_code()
        await self.move_valve(0)

    async def initialize_device(self) -> None:
        """Initialize the device"""

        # TODO: might be Y!!! Depends on whether left or right is facing front or back.
        _, error = await self.query('ZR')
        if not error:
            self.initialized = True
        else:
            logging.error(f'{self}: Initialization error {error}')

    async def set_valve_code(self, code: int | None = None) -> None:
        """Set valve code on MVP/4 device. Supposed to override DIP switches but does not appear to.

        Args:
            code (int | None, optional): Optionally specify valve code on MVP/4 device. Defaults to None.
        """
        
        code = self.valve.hamilton_valve_code if code is None else code

        if code is not None:
            _, error = await self.query(f'h2100{code}R')
            await self.poll_until_idle()
            if not error:
                await self.get_valve_code()
            else:
                logging.error(f'{self}: Valve code could not be set, got error {error}')
        else:
            logging.error(f'{self}: Unknown Hamilton valve code {code}')

    async def get_valve_code(self) -> None:
        """Reads the valve code from the device and checks against internal value
        """

        response, error = await self.query('?21000')
        if not error:
            code = int(response)
            if code != self.valve.hamilton_valve_code:
                logging.error(f'{self}: Valve code {code} from instrument does not match expected {self.valve.hamilton_valve_code}')
        else:
            logging.error(f'{self}: Valve code could not be read, got response {response} and error {error}')

    async def get_valve_position(self) -> None:
        """Reads the valve position from the device and updates the internal value
        """

        response, error = await self.query('?25000')
        if not error:
            angle = int(response)

            # convert to position
            delta_angle = 360 / self.valve.n_ports
            position = angle / delta_angle + 1

            # if non-integer position, check for off position or error
            if position != int(position):
                if angle == (delta_angle // 6) * 3:
                    position = 0
                else:
                    logging.error(f'{self}: valve is at unknown position {position} with angle {angle}')
                    position = None
            else:
                position = int(position)

            # record position
            logging.debug(f'{self}: Valve is at position {position}')
            self.valve.move(position)
        else:
            logging.error(f'{self}: Valve position could not be read, got response {response} and error {error}')

    async def move_valve(self, position: int) -> None:
        """Moves to a particular valve position. See specific valve documentation.

        Args:
            position (int): position to move the valve to
        """

        # this checks for errors
        self.valve.move(position)

        # convert to angle
        delta_angle = 360 / self.valve.n_ports

        # in special case of zero, move valve to "off" position between angles
        angle = delta_angle // 6 * 3 if position == 0 else (position - 1) * delta_angle

        _, error = await self.query(f'h29{angle:03.0f}R')
        await self.poll_until_idle()
        if error:
            logging.error(f'{self}: Move error {error}')

        # check that valve actually moved
        await self.get_valve_position()
        if self.valve.position != position:
            logging.error(f'{self}: Valve did not move to new position {position}, actually at {self.valve.position}')
        else:
            logging.debug(f'{self}: Move successful to position {position}')


class HamiltonSyringePump(HamiltonValvePositioner):
    """Hamilton syringe pump device. Includes both a syringe motor and a built-in valve positioner.
    """

    def __init__(self,
                 serial_instance: HamiltonSerial,
                 address: str,
                 valve: ValveBase,
                 syringe_volume: float = 5000,
                 high_resolution = False,
                 name = None,
                 ) -> None:
        super().__init__(serial_instance, address, valve, name)

        # Syringe volume in uL
        self.syringe_volume = syringe_volume

        # default high resolution mode is False
        self._high_resolution = high_resolution

        # syringe position
        self.syringe_position: int = 0.0

        # min and max V
        self.minV, self.maxV = 2, 10000

        # allow custom max flow rate
        self.max_flow_rate = self._max_flow_rate()
        self.min_flow_rate = self._min_flow_rate()

    async def initialize(self) -> None:
        await super().initialize()
        await self.run_until_idle(self.set_high_resolution(self._high_resolution))
        await self.run_until_idle(self.get_syringe_position())

    async def set_high_resolution(self, high_resolution: bool) -> None:
        """Turns high resolution mode on or off

        Args:
            high_resolution (bool): turn high resolution on (True) or off (False)
        """
        
        response, error = await self.query(f'N{int(high_resolution)}R')
        if error:
            logging.error(f'{self}: Error setting resolution: {error}')
        else:
            self._high_resolution = high_resolution

    def _full_stroke(self) -> int:
        """Calculates syringe stroke (# half steps for full volume) based on resolution mode

        Returns:
            float: stroke in half steps
        """

        return 48000 if self._high_resolution else 6000

    def _min_flow_rate(self) -> int:
        """Calculates minimum flow rate of device
        
        Returns:
            int: min flow rate in uL / s"""
        
        return self._flow_rate(self.minV)
    
    def _max_flow_rate(self) -> int:
        """Calculates maximum flow rate of device
        
        Returns:
            int: max flow rate in uL / s"""
        
        return self._flow_rate(self.maxV)

    def _speed_code(self, desired_flow_rate: float) -> int:
        """Calculates speed code (parameter V, see PSD/4 manual Appendix H) based on desired
            flow rate and syringe parameters

        Args:
            desired_flow_rate (float): desired flow rate in uL / s

        Returns:
            int: V (half-steps per second)
        """

        #calcV = float(desired_flow_rate * 6000) / self.syringe_volume

        if desired_flow_rate < self._min_flow_rate():
            logging.warning(f'{self}: Warning: clipping desired flow rate {desired_flow_rate} to lowest possible value {self._min_flow_rate()}')
            return self.minV
        elif desired_flow_rate > self._max_flow_rate():
            logging.warning(f'{self}: Warning: clipping desired flow rate {desired_flow_rate} to highest possible value {self._max_flow_rate()}')
            return self.maxV
        else:
            return round(float(desired_flow_rate * 6000) / self.syringe_volume)
        
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

        response, error = await self.query('?')
        
        self.syringe_position = int(response)

    async def aspirate(self, volume: float, flow_rate: float) -> None:
        """Aspirate (Pick-up)

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        await self.get_syringe_position()
        stroke_length = self._stroke_length(volume)
        max_position = self._full_stroke() / 2
        logging.debug(f'Stroke length: {stroke_length} out of full stroke {self._full_stroke() / 2}')

        if max_position < (stroke_length + self.syringe_position):
            logging.error(f'{self}: Invalid syringe move from current position {self.syringe_position} with stroke length {stroke_length} and maximum position {max_position}')
            
            # TODO: this is a hack to clear the response queue...need to fix this
            #await self.update_status()
        else:
            V = self._speed_code(flow_rate)
            logging.debug(f'Speed: {V}')

            response, error = await self.query(f'V{V}P{stroke_length}R')
            if error:
                logging.error(f'{self}: Syringe move error {error}')

    async def dispense(self, volume: float, flow_rate: float) -> None:
        """Dispense

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        await self.get_syringe_position()
        stroke_length = self._stroke_length(volume)
        logging.debug(f'Stroke length: {stroke_length} out of full stroke {self._full_stroke() / 2}')

        if (self.syringe_position - stroke_length) < 0:
            logging.error(f'{self}: Invalid syringe move from current position {self.syringe_position} with stroke length {stroke_length} and minimum position 0')
            #await self.update_status()
        else:
            V = self._speed_code(flow_rate)

            response, error = await self.query(f'V{V}D{stroke_length}R')
            if error:
                logging.error(f'{self}: Syringe move error {error}')

    async def home(self) -> None:
        """Home syringe.
        """

        response, error = await self.query(f'A0R')
        if error:
            logging.error(f'{self}: Syringe homing error {error}')

