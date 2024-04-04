import asyncio
import logging
import copy
import json

from enum import Enum
from typing import Tuple, List
from uuid import uuid4
from aiohttp import web

from HamiltonComm import HamiltonSerial
from HamiltonDevice import HamiltonSyringePump
from valve import ValveBase, SyringeValveBase
from connections import Node
from webview import sio, WebNodeBase

class SyringePumpwithBubbleSensor(HamiltonSyringePump):
    """Syringe pump with one or more bubble sensors driven by digital outputs and addressing digital inputs of the same index.
    """

    def __init__(self, serial_instance: HamiltonSerial, address: str, valve: SyringeValveBase, syringe_volume: float = 5000, high_resolution=False, name=None) -> None:
        super().__init__(serial_instance, address, valve, syringe_volume, high_resolution, name)

    async def get_digital_input(self, digital_input: int) -> bool:
        """Gets value of a digital input.

        Args:
            digital_input (int): Index (either 1 or 2)

        Returns:
            bool: return value
        """

        query_code = 12 + digital_input

        response, error = await self.query(f"?{query_code}")

        return bool(int(response))

    async def set_digital_output(self, digital_output: int, value: bool) -> None:
        """Activates digital output corresponding to its index. Reads current digital output state and
            makes the appropriate adjustment

        Args:
            sensor_index (int): Digital output that drives the bubble sensor
        """

        state = list(await self.get_digital_outputs())
        state[digital_output] = value

        await self.set_digital_outputs(state)

    async def set_digital_outputs(self, digital_outputs: Tuple[bool, bool, bool]) -> None:
        """Sets the three digital outputs, e.g. (True, False, False)

        Returns:
            Tuple[bool, bool, bool]: Tuple of the three digital output values
        """

        binary_string = ''.join(map(str, map(int, digital_outputs)))

        response, error = await self.query(f'J{int(binary_string, 2)}R')

    async def get_digital_outputs(self) -> Tuple[bool, bool, bool]:
        """Gets digital output values

        Returns:
            List[bool]: List of the three digital outputs
        """

        response, error = await self.query(f'?37000')
        binary_string = format(int(response), '03b')

        return tuple([bool(digit) for digit in binary_string])
    
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
            return
        if (self.valve.aspirate_position is None) | (self.valve.dispense_position is None):
            logging.error(f'{self.name}: aspirate_position and dispense_position must be set to use smart_dispense')
            return
        
        # convert speeds to V factors
        V_aspirate = self._speed_code(self.max_aspirate_flow_rate)
        V_dispense = self._speed_code(dispense_flow_rate)

        # calculate total volume in steps
        total_steps = self._stroke_length(volume)
        logging.debug(f'{self.name}: smart dispense requested {total_steps} steps')
        if total_steps <= 0:
            logging.warning(f'{self.name}: volume is not positive, smart_dispense terminating')
            return

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

