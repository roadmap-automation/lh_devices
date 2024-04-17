import asyncio
import logging
import copy
import time

from typing import Tuple, List

from HamiltonComm import HamiltonSerial
from HamiltonDevice import HamiltonSyringePump, PollTimer
from valve import ValveBase, SyringeValveBase

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