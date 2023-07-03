from typing import Tuple, List
import asyncio
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
        self.poll_delay = 0.1   # Hamilton-recommended 100 ms delay when polling
        self.address = address
        self.address_code = chr(int(address, base=16) + int('31', base=16))
        self.command_queue: asyncio.Queue = asyncio.Queue()
        self.response_queue: asyncio.Queue = asyncio.Queue()
        self.batch_queue: asyncio.Queue = asyncio.Queue()

    def get_nodes(self) -> List[Node]:

        return []

    async def initialize(self) -> None:

        await self.run_until_idle(self.initialize_device())

    async def initialize_device(self) -> None:
        pass

    async def _serial_query(self) -> str:
        """
        Gets command from command queue and runs a serial query
        """

        # Get command from command queue
        addr, cmd, queue = await self.command_queue.get()

        # Send address, command string, and response queue to serial connection
        await self.serial.query(addr, cmd, queue)
    
    async def run(self, cmd: asyncio.Future) -> None:
        """Runs a method using command and response queues

        Args:
            cmd (asyncio.Future): method to run. Sending command to self.command_queue will
                                    trigger the command to be run. Command should process the
                                    response.
        """

        await asyncio.gather(cmd, self._serial_query())

    async def run_in_batch(self, cmd: asyncio.Future, batch_queue: asyncio.Queue) -> None:
        """Runs a method as part of a batch (called from higher-level devices)

        Args:
            cmd (asyncio.Future): command to run (see self.run)
            batch_queue (asyncio.Queue): queue in which to put the command future
        """

        await batch_queue.put((self.serial, cmd, self.command_queue))

    async def query(self, cmd: str) -> Tuple[str | None, str | None]:
        """Adds command to command queue and waits for response"""
        
        # push command to command queue
        await self.command_queue.put((self.address_code, cmd, self.response_queue))

        # wait for response
        response = await self.response_queue.get()
        
        # process response
        if response:
            response = response[2:-1]
            error = self.parse_status_byte(response)
            return response, error
        else:
            return None, None

    async def poll_until_idle(self) -> None:
        """Polls device until idle

        Returns:
            str: error string
        """
        while (not self.idle):
            await asyncio.gather(self.run(self.update_status()),
                                 asyncio.sleep(self.poll_delay))    # delays before sending next polling request

    async def run_until_idle(self, cmd: asyncio.Future) -> None:
        """
        Sends from serial connection and waits until idle
        """

        self.idle = False
        await self.run(cmd)
        await self.poll_until_idle()

    async def update_status(self) -> None:
        """
        Polls the status of the device using 'Q'
        """

        _, error = await self.query('Q')

        # TODO: Handle error
        if error:
            print(f'Error in update_status: {error}')

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

    async def initialize_device(self) -> None:
        """Initialize the device"""

        # TODO: might be Y!!! Depends on whether left or right is facing front or back.
        _, error = await self.query('ZR')
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
        _, error = await self.query(f'I{position}')
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
        await self.run_until_idle(self.set_high_resolution(self._high_resolution))
        await super().initialize()

    async def set_high_resolution(self, high_resolution: bool) -> None:
        """Turns high resolution mode on or off

        Args:
            high_resolution (bool): turn high resolution on (True) or off (False)
        """
        
        response, error = await self.query(f'N{int(high_resolution)}R')
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

            response, error = await self.query(f'V{V}P{stroke_length}R')
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

            response, error = await self.query(f'V{V}D{stroke_length}R')
            if error:
                print(f'Syringe move error {error}')

    async def home(self) -> None:
        """Home syringe.
        """

        response, error = await self.query(f'A0R')
        if error:
            print(f'Syringe homing error {error}')

# ======== batch running tools
# by having these separate from the individual classes, they can be used for both single and
# multiple devices
#
""" TODO: This seems unnecessarily complicated. Requires calling syntax of the type:
    await batch_run([dev.run_in_batch(dev.<command>(), batch_queue) for dev in devices], batch_queue)

    It would be nice if batch_run could just accept dev.<command>() and the device in question 
    would be automatically retrieved. This would allow batch_run to generate its own temporary
    batch_queue and simplify everything.
"""

async def _batch_serial_query(ncmds: int, batch_queue: asyncio.Queue) -> None:
    """Handles batch of serial queries

    Args:
        ncmds (int): number of serial queries in the batch
        batch_queue (asyncio.Queue): batch queue containing the queries
    """

    # create a container for parsing serial requests by serial port
    unique_serial: dict[HamiltonSerial, List[asyncio.Queue]] = {}
    tasks = []

    # loop through commands in batch request
    for _ in range(ncmds):
        # get serial port, command future, and the command queue
        item: Tuple[HamiltonSerial, asyncio.Future, asyncio.Queue] = await batch_queue.get()
        ser, cmd, command_queue = item

        # unique serial ports get a new entry
        if ser not in unique_serial.keys():
            unique_serial[ser] = []
        
        # maintqain a list of command queues for each item
        unique_serial[ser].append(command_queue)

        # maintain a list of tasks
        tasks.append(cmd)

    # run concurrently the tasks and the command / response handler
    tasks.append(_batch_run_tasks(unique_serial))
    await asyncio.gather(*tasks)

async def _batch_run_tasks(unique_serial: dict[HamiltonSerial, List[asyncio.Queue]]) -> None:
    """Command / response handler for batch requests

    Args:
        unique_serial (dict[HamiltonSerial, List[asyncio.Queue]]): parsed serial requests from
            _batch_serial_query
    """

    # loop through serial connections
    for ser, queuelist in unique_serial.items():
        addrlist = []
        cmdstrlist = []
        rqlist = []

        # for each request in the serial connection, get the command queue
        for command_queue in queuelist:

            # get query information from command queue
            item: Tuple[int, str, asyncio.Queue] =  await command_queue.get()
            addr, cmdstr, response_queue = item

            # populate lists for batch_query
            addrlist.append(addr)
            cmdstrlist.append(cmdstr)
            rqlist.append(response_queue)
        
        # execute the batch query
        await ser.batch_query(addrlist, cmdstrlist, rqlist)

async def batch_run(cmds: List[asyncio.Future], batch_queue: asyncio.Queue) -> None:
    """Run a batch of async commands of type run_in_batch. Might involve different serial ports

    Args:
        cmds (List[asyncio.Future]): List of asyncio futures from HamiltonBase.run_in_batch
        batch_queue (asyncio.Queue): Queue to handle batches (sync with run_in_batch)
    """

    tasks = cmds
    tasks.append(_batch_serial_query(len(cmds), batch_queue))

    await asyncio.gather(*tasks)
