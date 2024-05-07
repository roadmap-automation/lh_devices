import asyncio
import logging
import copy
import time

from typing import Tuple, List
from uuid import uuid4

from HamiltonComm import HamiltonSerial
from HamiltonDevice import HamiltonBase, HamiltonSyringePump, PollTimer
from valve import ValveBase, SyringeValveBase

class BubbleSensorBase:
    """Base class for bubble sensor
    """

    def __init__(self, id: str | None = None, name: str = '') -> None:
        self.id = id if id is not None else str(uuid4())
        self.name = name

    async def initialize(self) -> None:
        """Initializes the bubble sensor (e.g. powers it on, connects to it)
        """

        raise NotImplementedError

    async def read(self) -> bool:
        """Reads the bubble sensor and returns its value

        Returns:
            bool: bubble sensor value
        """

        raise NotImplementedError
    
class SMDSensoronHamiltonDevice(BubbleSensorBase):

    def __init__(self, device: HamiltonBase, digital_input: int, power_digital_output: int, id: str | None = None, name: str = '') -> None:
        super().__init__(id, name)
        self.device = device
        self.digital_input = digital_input
        self.power_digital_output = power_digital_output

    async def initialize(self) -> None:
        return await self.device.run_until_idle(self.device.set_digital_output(self.power_digital_output, True))

    async def read(self) -> bool:
        """Read bubble sensor

        Returns:
            bool: true if liquid in line; false if air
        """

        return await self.device.get_digital_input(self.digital_input)

class SyringePumpwithBubbleSensor(HamiltonSyringePump):
    """Syringe pump with one or more bubble sensors driven by digital outputs and addressing digital inputs of the same index.
    """

    def __init__(self, serial_instance: HamiltonSerial, address: str, valve: SyringeValveBase, syringe_volume: float = 5000, high_resolution=False, name=None) -> None:
        super().__init__(serial_instance, address, valve, syringe_volume, high_resolution, name)

    async def update_buffer_status(self) -> bool:
        """Reads buffer status.

        Returns:
            bool: whether buffer is empty. False on error (typically because device is busy)
        """

        response, error = await self.query('F')

        return True if response == 1 else False

    async def time_until_buffer_empty(self, cmd: asyncio.Future) -> float:
        """
        Sends from serial connection and waits until buffer empty.
            Typical usage: run_until_buffer_empty('H1R')

        Returns:
            float: time in seconds since command was initiated
        """

        start_time = time.time()

        await cmd

        timer = PollTimer(self.poll_delay, self.address_code)

        buffer_empty = False

        while (not buffer_empty):
            # run update_status and start the poll_delay timer
            buffer_empty, _ = await asyncio.gather(self.update_buffer_status(), timer.cycle())

            # wait until poll_delay timer has ended before asking for new status.
            await timer.wait_until_set()

        return time.time() - start_time