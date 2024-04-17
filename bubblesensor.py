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

    # TODO: Make a better base class for move_absolute and smart_dispense so the code is not copied here

    async def move_absolute(self, position: int, V_rate: int, interrupt_index: int | None = None) -> None:
        """Low-level method for moving the syringe to an absolute position using
            the V speed code

        Args:
            position (int): syringe position in steps
            V_rate (int): movement rate
            interrupt_index (int | None, optional): index of condition to interrupt syringe movement.
                See Hamilton PSD manual for codes. Defaults to None.
        """

        interrupt_string = '' if interrupt_index is None else f'i{interrupt_index}'

        response, error = await self.query(f'V{V_rate}' + interrupt_string + f'A{position}R')
        if error:
            logging.error(f'{self}: Syringe move error {error} for move to position {position} with V {V_rate}')

    async def smart_dispense(self, volume: float, dispense_flow_rate: float, interrupt_index: int | None = None) -> float:
        """Smart dispense, including both aspiration at max flow rate, dispensing at specified
            flow rate, and the ability to handle a volume that is larger than the syringe volume.
            Allows interruptions via bubble sensors and returns total volume actually dispensed
            
        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
            interrupt_index (int | None, optional): See move_absolute. Defaults to None.
            
        Returns:
            float: volume actually dispensed in uL"""

        # check that aspiration and dispense positions are defined
        if (not hasattr(self.valve, 'aspirate_position')) | (not hasattr(self.valve, 'dispense_position')):
            logging.error(f'{self.name}: valve must have aspirate_position and dispense_position defined to use smart_dispense')
            return 0
        if (self.valve.aspirate_position is None) | (self.valve.dispense_position is None):
            logging.error(f'{self.name}: aspirate_position and dispense_position must be set to use smart_dispense')
            return 0
        
        # convert speeds to V factors
        V_aspirate = self._speed_code(self.max_aspirate_flow_rate)
        V_dispense = self._speed_code(dispense_flow_rate)

        # calculate total volume in steps
        total_steps = self._stroke_length(volume)
        logging.debug(f'{self.name}: smart dispense requested {total_steps} steps')
        if total_steps <= 0:
            logging.warning(f'{self.name}: volume is not positive, smart_dispense terminating')
            return 0

        # calculate max number of steps
        full_stroke = self._full_stroke() // 2

        # update current syringe position (usually zero)
        await self.update_syringe_status()
        syringe_position = self.syringe_position
        current_position = copy.copy(syringe_position)
        logging.debug(f'{self.name}: smart dispense, syringe at {current_position}')

        # calculate number of aspirate/dispense operations and volume per operation
        # if there is already enough volume in the syringe, just do a single dispense
        total_steps_dispensed = 0
        if current_position >= total_steps:
            # switch valve and dispense
            logging.debug(f'{self.name}: smart dispense dispensing {total_steps} at V {V_dispense}')
            await self.run_until_idle(self.move_valve(self.valve.dispense_position))
            await self.run_syringe_until_idle(self.move_absolute(current_position - total_steps, V_dispense, interrupt_index))
            total_steps_dispensed += current_position - self.syringe_position
        else:
            # number of full_volume loops plus remainder
            stroke_steps = [full_stroke] * (total_steps // full_stroke) + [total_steps % full_stroke]
            for stroke in stroke_steps:
                if stroke > 0:
                    logging.debug(f'{self.name}: smart dispense aspirating {stroke - current_position} at V {V_aspirate}')
                    # switch valve and aspirate
                    await self.run_until_idle(self.move_valve(self.valve.aspirate_position))
                    await self.run_syringe_until_idle(self.move_absolute(stroke, V_aspirate))
                    position_after_aspirate = copy.copy(self.syringe_position)

                    # switch valve and dispense; run_syringe_until_idle updates self.syringe_position
                    logging.debug(f'{self.name}: smart dispense dispensing all at V {V_dispense}')
                    await self.run_until_idle(self.move_valve(self.valve.dispense_position))
                    await self.run_syringe_until_idle(self.move_absolute(0, V_dispense, interrupt_index))
                    position_change = position_after_aspirate - self.syringe_position
                    total_steps_dispensed += position_change
                    
                    if (stroke == position_change):
                        # update current position and go to next step
                        current_position = copy.copy(self.syringe_position)
                    else:
                        # stop! do not do continued strokes because the interrupt was triggered
                        break

        return self._volume_from_stroke_length(total_steps_dispensed)

