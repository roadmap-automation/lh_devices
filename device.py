import asyncio
import copy
import logging
from uuid import uuid4
from typing import List, Literal
from connections import Node
from dataclasses import dataclass
from valve import ValveState, ValveBase

@dataclass
class DeviceError:
    error: str | None = None
    retry: bool = False

    def __post_init__(self):
        self._clear_error_event: asyncio.Event = asyncio.Event()

    async def pause_until_clear(self) -> None:
        await self._clear_error_event.wait()

    def clear(self, retry: bool | None = None) -> None:
        self.error = None
        if retry is not None:
            self.retry = retry

        self._clear_error_event.set()
        self._clear_error_event.clear()

    def __repr__(self):
        return f'DeviceError: {self.error}'

@dataclass
class DeviceState:
    id: str
    name: str
    idle: bool
    initialized: bool
    reserved: bool
    error: DeviceError

class DeviceBase:
    """Base device class
    """

    def __init__(self, id: str | None = None, name: str | None = None) -> None:
        
        if id is None:
            self.id = str(uuid4())
        self.name=name
        self.idle: bool = True
        self.initialized: bool = False
        self.reserved = False   # like a lock; allows reserving the device before running a method
        self.error: DeviceError = DeviceError()
        self.poll_delay = 0.1

    def __repr__(self):

        if self.name:
            return self.name
        else:
            return object.__repr__(self)

    @property
    def state(self) -> DeviceState:
        """Gets the current state
        """

        return DeviceState(id=self.id,
                           name=self.name,
                           idle=self.idle,
                           initialized=self.initialized,
                           reserved=self.reserved,
                           error=self.error)

    def get_nodes(self) -> List[Node]:

        return []

    async def is_initialized(self) -> bool:
        """Query device to get initialization state

        Returns:
            bool: True if device is initialized, else False
        """

        return self.initialized

    async def initialize(self) -> None:
        """Initialize device only if not already initialized
        """

        if not await self.is_initialized():
            await self.run_until_idle(self.initialize_device())

    async def initialize_device(self) -> None:
        
        self.idle = True
        self.initialized = True

    async def update_status(self) -> None:
        """
        Returns the status of the device. Intended to be subclassed
        for real devices
        """

        pass

    async def poll_until_idle(self) -> None:
        """Polls device until idle

        Returns:
            str: error string
        """

        timer = PollTimer(self.poll_delay, self.name)

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
        # TODO: Figure out whether this is the best updating strategy
        await self.trigger_update()
        await self.poll_until_idle()
        await self.trigger_update()

    async def run_async(self, cmd: asyncio.Future) -> None:
        """
        Sends from serial connection but does not update idle status
        """

        idle_value = copy.copy(self.idle)
        await cmd
        await self.trigger_update()
        self.idle = idle_value

    async def reset(self) -> None:
        """Resets the device
        """

        self.idle = True
        self.initialized = False

    async def stop(self) -> None:
        """Stops the device (terminates command buffer)
        """

        self.idle = True

    async def resume(self) -> None:
        """Resumes the device from next unexecuted command
        """

        pass

    async def trigger_update(self):
        """Sends signal for update"""

        pass

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


# ========= Valve positioners ===========
@dataclass
class ValvePositionerState(DeviceState):
    valve_state: ValveState

class ValvePositionerBase(DeviceBase):

    def __init__(self, valve: ValveBase, id = None, name = None):
        DeviceBase.__init__(self, id=id, name=name)

        self.valve = valve

    @property
    def state(self) -> ValvePositionerState:
        """Gets the current state
        """

        return ValvePositionerState(id=self.id,
                                    name=self.name,
                                    idle=self.idle,
                                    initialized=self.initialized,
                                    reserved=self.reserved,
                                    error=self.error,
                                    valve_state=self.valve.state)

    def get_nodes(self) -> List[Node]:
        
        return self.valve.nodes
   
    async def move_valve(self, position: int) -> None:
        """Moves to a particular valve position. See specific valve documentation.

        Args:
            position (int): position to move the valve to
        """

        if self.valve.validate_move(position):
            self.valve.move(position)

# ========= Syringe pump ===========
@dataclass
class SyringeState:
    position: float             # in ul
    syringe_volume:  float      # in ul
    speed: float                # in ul / s
    volume_units: Literal['ml', 'ul'] = 'ml'
    time_units: Literal['s', 'min'] = 'min'

@dataclass
class SyringePumpState(DeviceState):
    syringe_state: SyringeState

class SyringePumpBase(DeviceBase):

    def __init__(self, syringe_volume: float, id = None, name = None):
        super().__init__(id, name)

        self.max_aspirate_flow_rate = 15
        self.max_dispense_flow_rate = 15
        self.min_flow_rate = 0.1

        self.syringe_volume: float = syringe_volume
        self.syringe_position: float = 0.0

        self._speed = self.min_flow_rate

    @property
    def syringe_speed(self):
        """Read only property returning the syringe speed"""

        return self._speed

    @property
    def syringe_state(self) -> SyringeState:
        """Gets the current state of the syringe

        Returns:
            SyringeState: state of the syringe
        """

        return SyringeState(self.syringe_position / 1000,
                            self.syringe_volume / 1000,
                            self.syringe_speed * 60. / 1000.,
                            volume_units='ml',
                            time_units='min')

    @property
    def state(self):
        """Gets the current state
        """

        return SyringePumpState(id=self.id,
                                name=self.name,
                                idle=self.idle,
                                initialized=self.initialized,
                                reserved=self.reserved,
                                error=self.error,
                                syringe_state=self.syringe_state)

    async def update_syringe_status(self) -> int:
        """Reads absolute position of syringe

        Returns:
            int: absolute position of syringe in steps
        """

        pass

    async def aspirate(self, volume: float, flow_rate: float) -> None:
        """Aspirate (Pick-up)

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        pass

    async def dispense(self, volume: float, flow_rate: float) -> None:
        """Dispense

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        pass

    async def smart_dispense(self, volume: float, dispense_flow_rate: float, interrupt_index: int | None = None) -> float:
        """Smart dispense, including both aspiration at max flow rate, dispensing at specified
            flow rate, and the ability to handle a volume that is larger than the syringe volume
            
        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
            interrupt_index (int | None, optional): See move_absolute. Defaults to None.
            
        Returns:
            float: volume actually dispensed in uL
            """
        
        return volume

    async def move_absolute(self, position: int, interrupt_index: int | None = None) -> None:
        """Low-level method for moving the syringe to an absolute position

        Args:
            position (int): syringe position in steps
            interrupt_index (int | None, optional): index of condition to interrupt syringe movement.
                See Hamilton PSD manual for codes. Defaults to None.

        """

        self.syringe_position = position

    async def set_speed(self, flow_rate: float) -> str:
        """Sets syringe speed to a specified flow rate

        Args:
            flow_rate (float): flow rate in uL / s
        """

        self._speed = flow_rate

    async def home(self) -> None:
        """Homes syringe using maximum flow rate
        """

        await self.set_speed(self.max_dispense_flow_rate)
        await self.move_absolute(0)

@dataclass
class SyringePumpValvePositionerState(SyringePumpState, ValvePositionerState):
    ...

class SyringePumpValvePositioner(ValvePositionerBase, SyringePumpBase):

    def __init__(self, valve, syringe_volume: float, id=None, name=None):
        ValvePositionerBase.__init__(self, valve, id=id, name=name)
        SyringePumpBase.__init__(self, syringe_volume, id=self.id, name=self.name)

    @property
    def state(self):

        return SyringePumpValvePositionerState(
                id=self.id,
                name=self.name,
                idle=self.idle,
                initialized=self.initialized,
                reserved=self.reserved,
                error=self.error,
                valve_state=self.valve.state,
                syringe_state=SyringeState(self.syringe_position / 1000,
                                           self.syringe_volume / 1000,
                                           self.syringe_speed * 60. / 1000.,
                                           volume_units='ml',
                                           time_units='min'
                                           )
                )

