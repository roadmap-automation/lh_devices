import asyncio
import logging
import copy
import random
import time

from typing import Tuple
from aiohttp import web
from dataclasses import dataclass, asdict

from .HamiltonComm import HamiltonSerial
from ..bubblesensor import BubbleSensorBase
from ..device import (DeviceBase, DeviceState, ValvePositionerBase,
                    ValvePositionerState, PollTimer,
                    SyringePumpValvePositionerState,
                    SyringePumpValvePositioner, SyringeState,
                    DeviceError)
from ..valve import ValveBase, SyringeValveBase
from ..webview import WebNodeBase

@dataclass
class HamiltonDeviceState(DeviceState):
    digital_outputs: Tuple[bool, bool, bool]

class HamiltonBase(DeviceBase):
    """Base class for Hamilton multi-valve positioner (MVP) and syringe pump (PSD) devices.

        Requires:
        serial_instance -- HamiltonSerial instance for communication
        address -- single character string from '0' to 'F' corresponding to the physical
                    address switch position on the device. Automatically converted to the 
                    correct address code.
     
       """

    def __init__(self, serial_instance: HamiltonSerial, address: str, device_id: str = None, name: str = None) -> None:

        DeviceBase.__init__(self, device_id=device_id, name=name)        
        self.serial = serial_instance
        self.digital_outputs: Tuple[bool, bool, bool] = (False, False, False)
        self.busy_code = '@'
        self.idle_code = '`'
        self.poll_delay = 0.1   # Hamilton-recommended 100 ms delay when polling
        self.address = address
        self.address_code = chr(int(address, base=16) + int('31', base=16))
        self.response_queue: asyncio.Queue = asyncio.Queue()

    @property
    def state(self) -> HamiltonDeviceState:
        """Gets the current state
        """

        return HamiltonDeviceState(id=self.id,
                           name=self.name,
                           idle=self.idle,
                           initialized=self.initialized,
                           reserved=self.reserved,
                           error=self.error,
                           digital_outputs=self.digital_outputs)

    async def initialize(self) -> None:
        """Initialize device only if not already initialized
        """

        await super().initialize()
        
        await self.run_until_idle(self.set_digital_outputs((False, False, False)))

    async def initialize_device(self) -> None:
        pass

    async def query(self, cmd: str) -> Tuple[str | None, DeviceError | None]:
        """Adds command to command queue and waits for response"""
        
        # push command to command queue
        await self.serial.query(self.address_code, cmd, self.response_queue)

        # wait for response
        response = await self.response_queue.get()
        
        # process response
        if response:
            response = response[2:-1]
            response = self.parse_status_byte(response)
            if self.error.error is not None:
                self.logger.error(f'{self} error: {self.error}, waiting for clear, retry = {self.error.retry}')
                await self.trigger_update()
                await self.error.pause_until_clear()
                self.logger.info(f'{self} error cleared')
                if self.error.retry:
                    self.logger.info(f'{self} error: retrying command {cmd}')
                    response, error = await self.query(cmd)

            return response, self.error
        else:
            self.error.error = 'No response to query'
            self.error.retry = True
            return None, self.error

    async def update_status(self) -> None:
        """
        Polls the status of the device using 'Q'
        """

        await self.query('Q')

    def parse_status_byte(self, response: str) -> Tuple[str | None, DeviceError | None]:
        """
        Parses status byte
        """
        c = response[0]

        error = DeviceError()
        match c:
            case self.busy_code:
                self.idle = False
            case self.idle_code:
                self.idle = True
            case 'b':
                self.error.error = 'Bad command'
                self.error.retry = False
            case _ :
                self.error.error = f'Error code: {c}'
                self.error.retry = True

        return response.split(c, 1)[1]

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

        state = list(self.digital_outputs)
        state[digital_output] = value

        await self.set_digital_outputs(tuple(state))

    async def set_digital_outputs(self, digital_outputs: Tuple[bool, bool, bool]) -> None:
        """Sets the three digital outputs, e.g. (True, False, False)

        Returns:
            Tuple[bool, bool, bool]: Tuple of the three digital output values
        """

        binary_string = ''.join(map(str, map(int, digital_outputs[::-1])))

        response, error = await self.query(f'J{int(binary_string, 2)}R')
        self.digital_outputs = digital_outputs

    async def get_digital_outputs(self) -> Tuple[bool, bool, bool]:
        """Gets digital output values

        Returns:
            List[bool]: List of the three digital outputs
        """

        response, error = await self.query(f'?37000')
        binary_string = format(int(response), '03b')

        digital_outputs = tuple([bool(digit) for digit in binary_string[::-1]])
        self.digital_outputs = digital_outputs

        return digital_outputs

    def create_web_app(self, template='roadmap.html') -> web.Application:
        """Creates a web application for this specific device

        Returns:
            web.Application: web application for this device
        """

        return super().create_web_app(template)

    async def get_info(self) -> dict:
        """Gets object state as dictionary

        Returns:
            dict: object state
        """

        d = await super().get_info()

        d.update({
                'type': 'device',
                'config': {
                           'com_port': self.serial.port if self.serial is not None else None,
                           'address': self.address},
                'state': asdict(self.state),
                'controls': {
                             #'test error': {'type': 'textbox',
                             #               'text': 'test error'},
                             'reset': {'type': 'button',
                                       'text': 'Reset'},
                             'stop': {'type': 'button',
                                      'text': 'Stop'},
                             'resume': {'type': 'button',
                                        'text': 'Resume Next'},
                             'clear error': {'type': 'button',
                                             'text': 'Clear Error'},
                             #'retry': {'type': 'button',
                             #          'text': 'Clear Error and Retry'},
                                       }})
        
        return d

    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'reset':
            await self.run_until_idle(self.reset())
            await self.initialize()
        elif command == 'clear error':
            self.error.clear()
        elif command == 'retry':
            self.error.clear(retry=True)
        elif command == 'test error':
            await self.query(data['value'])
            await self.trigger_update()
        elif command == 'stop':
            await self.run_until_idle(self.stop())
        elif command == 'resume':
            await self.resume() # probably do not want to run this until idle because device will not be idle after resuming
        elif command == 'set_digital_output':
            await self.run_until_idle(self.set_digital_output(int(data['number']), bool(data['value'])))

    async def reset(self) -> None:
        """Resets the device
        """

        response, error = await self.query('h30003R')
        self.initialized = False

    async def stop(self) -> None:
        """Stops the device (terminates command buffer)
        """

        response, error = await self.query('T')

    async def resume(self) -> None:
        """Resumes the device from next unexecuted command
        """

        response, error = await self.query('R')

    async def trigger_update(self) -> None:
        """Trigger update using WebNodeBase inheritance
        """

        return await WebNodeBase.trigger_update(self)

class SimulatedHamiltonBase(DeviceBase):
    """Base class for Hamilton multi-valve positioner (MVP) and syringe pump (PSD) devices.
       """

    def __init__(self, speed_multiplier: float = 1.0, device_id=None, name=None) -> None:

        DeviceBase.__init__(self, device_id=device_id, name=name)        
        self.digital_outputs: Tuple[bool, bool, bool] = (False, False, False)
        self.digital_inputs: Tuple[bool, bool] = (False, False)
        self.speed_multiplier = speed_multiplier

    @property
    def state(self) -> HamiltonDeviceState:
        """Gets the current state
        """

        return HamiltonDeviceState(id=self.id,
                           name=self.name,
                           idle=self.idle,
                           initialized=self.initialized,
                           reserved=self.reserved,
                           error=self.error,
                           digital_outputs=self.digital_outputs)

    async def initialize(self) -> None:
        """Initialize device only if not already initialized
        """

        await super().initialize()
        
        await self.set_digital_outputs((False, False, False))

    async def initialize_device(self) -> None:
        self.initialized = True
        self.idle = True

    async def update_status(self) -> None:
        """
        Polls the status of the device using 'Q'
        """

        return self.idle

    async def get_digital_input(self, digital_input: int) -> bool:
        """Gets value of a digital input.

        Args:
            digital_input (int): Index (either 1 or 2)

        Returns:
            bool: return value
        """

        return self.digital_inputs[digital_input - 1]

    async def set_digital_output(self, digital_output: int, value: bool) -> None:
        """Activates digital output corresponding to its index. Reads current digital output state and
            makes the appropriate adjustment

        Args:
            sensor_index (int): Digital output that drives the bubble sensor
        """

        state = list(self.digital_outputs)
        state[digital_output] = value

        await self.set_digital_outputs(tuple(state))
        self.idle = True

    async def set_digital_outputs(self, digital_outputs: Tuple[bool, bool, bool]) -> None:
        """Sets the three digital outputs, e.g. (True, False, False)

        Returns:
            Tuple[bool, bool, bool]: Tuple of the three digital output values
        """

        self.digital_outputs = digital_outputs
        self.idle = True

    async def get_digital_outputs(self) -> Tuple[bool, bool, bool]:
        """Gets digital output values

        Returns:
            List[bool]: List of the three digital outputs
        """

        return self.digital_outputs

    async def get_info(self) -> dict:
        """Gets object state as dictionary

        Returns:
            dict: object state
        """

        d = await super().get_info()

        d.update({
                'type': 'device',
                'config': {},
                'state': asdict(self.state),
                'controls': {'reset': {'type': 'button',
                                     'text': 'Reset'},
                             'stop': {'type': 'button',
                                     'text': 'Stop'},
                             'resume': {'type': 'button',
                                     'text': 'Resume Next'}}})
        
        return d

    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'reset':
            await self.run_until_idle(self.reset())
            await self.initialize()
        elif command == 'stop':
            await self.run_until_idle(self.stop())
        elif command == 'resume':
            await self.resume()
        elif command == 'set_digital_output':
            await self.run_until_idle(self.set_digital_output(int(data['number']), bool(data['value'])))

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

        self.idle = False

    async def trigger_update(self) -> None:
        """Trigger update using WebNodeBase inheritance
        """

        return await WebNodeBase.trigger_update(self)


@dataclass
class HamiltonValvePositionerState(HamiltonDeviceState, ValvePositionerState):
    ...

class HamiltonValvePositioner(HamiltonBase, ValvePositionerBase):
    """Hamilton MVP4 device
    """

    def __init__(self, serial_instance: HamiltonSerial, address: str, valve: ValveBase, device_id=None, name=None) -> None:
        HamiltonBase.__init__(self, serial_instance, address, device_id=device_id, name=name)
        ValvePositionerBase.__init__(self, valve, self.id, self.name)

    @property
    def state(self) -> HamiltonValvePositionerState:
        """Gets the current state
        """

        return HamiltonValvePositionerState(id=self.id,
                           name=self.name,
                           idle=self.idle,
                           initialized=self.initialized,
                           reserved=self.reserved,
                           error=self.error,
                           digital_outputs=self.digital_outputs,
                           valve_state=self.valve.state)

    async def initialize(self) -> None:
        await super().initialize()
        await self.set_valve_code()
        await self.move_valve(0)

    async def initialize_device(self) -> None:
        """Initialize the device"""

        _, error = await self.query('ZR')
        if error.error is not None:
            self.logger.error(f'{self}: Initialization error {error}')
        else:
            self.initialized = True

    async def is_initialized(self) -> bool:
        """Query device to get initialization state

        Returns:
            bool: True if device is initialized, else False
        """

        status = await self.get_valve_status()

        return True if status[-1]=='0' else False

    async def get_valve_status(self) -> str:
        """Gets full status string of device

        Returns:
            str: six-bit binary string with status
        """
        
        response, error = await self.query('?20000')

        return format(int(response), '06b')

    async def set_valve_code(self, code: int | None = None) -> None:
        """Set valve code on MVP/4 device.
            Syringe codes are not used (absolute angles are used) but syringe will check for various
            positions that it believes to be incorrect and throw errors, so correct valve code is important.

        Args:
            code (int | None, optional): Optionally specify valve code on MVP/4 device. Defaults to None.
        """
        
        code = self.valve.hamilton_valve_code if code is None else code

        if code is not None:
            _, error = await self.query(f'h2100{code}R')
            await self.poll_until_idle()
            if error.error is None:
                await self.get_valve_code()
            else:
                self.logger.error(f'{self}: Valve code could not be set, got error {error}')
        else:
            self.logger.error(f'{self}: Unknown Hamilton valve code {code}')

    async def get_valve_code(self) -> None:
        """Reads the valve code from the device and checks against internal value
        """

        response, error = await self.query('?21000')
        if error.error is None:
            code = int(response)
            if code != self.valve.hamilton_valve_code:
                self.logger.error(f'{self}: Valve code {code} from instrument does not match expected {self.valve.hamilton_valve_code}')
        else:
            self.logger.error(f'{self}: Valve code could not be read, got response {response} and error {error}')

    async def get_valve_position(self) -> None:
        """Reads the valve position from the device and updates the internal value
        """

        response, error = await self.query('?25000')
        if error.error is None:
            angle = int(response)

            # convert to position
            delta_angle = 360 / self.valve.n_positions
            position = angle / delta_angle + 1

            # if non-integer position, check for off position or error
            if position != int(position):
                if angle == (delta_angle // 6) * 3:
                    position = 0
                else:
                    self.logger.error(f'{self}: valve is at unknown position {position} with angle {angle}')
                    position = None
            else:
                position = int(position)

            # record position
            self.logger.debug(f'{self}: Valve is at position {position}')
            self.valve.move(position)
        else:
            self.logger.error(f'{self}: Valve position could not be read, got response {response} and error {error}')

    async def move_valve(self, position: int) -> None:
        """Moves to a particular valve position. See specific valve documentation.

        Args:
            position (int): position to move the valve to
        """

        # this checks for errors
        if self.valve.validate_move(position):

            # convert to angle
            delta_angle = 360 / self.valve.n_positions

            #print(self.valve.n_positions, delta_angle, delta_angle // 6 * 3, (position - 1) * delta_angle)

            # in special case of zero, move valve to "off" position between angles
            angle = delta_angle // 6 * 3 if position == 0 else (position - 1) * delta_angle

            _, error = await self.query(f'h29{angle:03.0f}R')
            await self.poll_until_idle()
            if error.error is not None:
                self.logger.error(f'{self}: Move error {error}')

            # check that valve actually moved
            await self.get_valve_position()
            if self.valve.position != position:
                self.logger.error(f'{self}: Valve did not move to new position {position}, actually at {self.valve.position}')
            else:
                self.logger.debug(f'{self}: Move successful to position {position}')

    async def get_info(self) -> dict:
        """Gets information about valve positioner

        Returns:
            dict: information dictionary
        """
        info = await super().get_info()
        controls = {'move_valve': {'type': 'select',
                                   'text': 'Move Valve: ',
                                   'options': [str(i) for i in range(self.valve.n_positions + 1)],
                                   'current': str(self.valve.position)}
                   }
        info['controls'] = info['controls'] | controls
        return info
    
    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'move_valve':
            newposition = data['index']
            await self.run_until_idle(self.move_valve(int(newposition)))


class SimulatedHamiltonValvePositioner(SimulatedHamiltonBase, ValvePositionerBase):
    """Simulated Hamilton MVP4 device
    """

    def __init__(self, valve: ValveBase, device_id=None, name=None) -> None:
        SimulatedHamiltonBase.__init__(self, device_id=device_id, name=name)
        ValvePositionerBase.__init__(self, valve, device_id=self.id, name=self.name)

    @property
    def state(self) -> HamiltonValvePositionerState:
        """Gets the current state
        """

        return HamiltonValvePositionerState(id=self.id,
                           name=self.name,
                           idle=self.idle,
                           initialized=self.initialized,
                           reserved=self.reserved,
                           error=self.error,
                           digital_outputs=self.digital_outputs,
                           valve_state=self.valve.state)

    async def initialize(self) -> None:
        await super().initialize()
        await self.move_valve(0)

    async def get_info(self) -> dict:
        """Gets information about valve positioner

        Returns:
            dict: information dictionary
        """
        info = await super().get_info()
        controls = {'move_valve': {'type': 'select',
                                   'text': 'Move Valve: ',
                                   'options': [str(i) for i in range(self.valve.n_positions + 1)],
                                   'current': str(self.valve.position)}
                   }
        info['controls'] = info['controls'] | controls
        return info
    
    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'move_valve':
            newposition = data['index']
            await self.run_until_idle(self.move_valve(int(newposition)))

    async def move_valve(self, position):
        self.idle = True
        return await super().move_valve(position)

@dataclass
class HamiltonSyringePumpState(HamiltonDeviceState, SyringePumpValvePositionerState):
    ...

class HamiltonSyringePump(HamiltonValvePositioner, SyringePumpValvePositioner):
    """Hamilton syringe pump device. Includes both a syringe motor and a built-in valve positioner.
    """

    def __init__(self,
                 serial_instance: HamiltonSerial,
                 address: str,
                 valve: SyringeValveBase,
                 syringe_volume: float = 5000,
                 high_resolution = False,
                 id: str = None,
                 name: str = None,
                 ) -> None:
        HamiltonValvePositioner.__init__(self, serial_instance, address, valve, id, name)
        SyringePumpValvePositioner.__init__(self, valve, syringe_volume, self.id, self.name)

        # Syringe poll delay
        self.syringe_poll_delay = 0.2

        # default high resolution mode is False
        self._high_resolution = high_resolution

        # min and max V speed codes
        self.minV, self.maxV = 2, 10000

        # save syringe speed code. Note speed is saved as a speed code
        self._speed = self.minV

        # allow custom max flow rate
        self.max_aspirate_flow_rate = self._max_flow_rate()
        self.max_dispense_flow_rate = self._max_flow_rate()
        self.min_flow_rate = self._min_flow_rate()

    @property
    def syringe_speed(self) -> float:
        return self._flow_rate(self._speed)

    @property
    def syringe_state(self) -> SyringeState:
        """Gets the current state of the syringe

        Returns:
            SyringeState: state of the syringe
        """

        position = self.syringe_position / self._get_max_position() * self.syringe_volume / 1000.
        return SyringeState(position,
                            self.syringe_volume / 1000,
                            self.syringe_speed * 60 / 1000,
                            volume_units='ml',
                            time_units='min')

    @property
    def state(self) -> HamiltonSyringePumpState:
        """Gets the current state
        """

        return HamiltonSyringePumpState(id=self.id,
                           name=self.name,
                           idle=self.idle,
                           initialized=self.initialized,
                           reserved=self.reserved,
                           error=self.error,
                           digital_outputs=self.digital_outputs,
                           valve_state=self.valve.state,
                           syringe_state=self.syringe_state)

    async def initialize(self) -> None:
        await super().initialize()
        await self.run_until_idle(self.set_high_resolution(self._high_resolution))
        await self.run_until_idle(self.update_syringe_status())

    async def is_initialized(self) -> bool:
        """Query device to get initialization state

        Returns:
            bool: True if device is initialized, else False
        """

        status = await self.get_syringe_status()
        syringe_initialized = True if status[-1]=='0' else False
        valve_initialized = await super().is_initialized()

        return (valve_initialized & syringe_initialized)

    async def get_info(self) -> dict:
        """Gets information about valve positioner

        Returns:
            dict: information dictionary
        """
        info = await super().get_info()

        controls = {'soft_reset' : {'type': 'button',
                                    'text': 'Soft Reset'},
                    'home_syringe': {'type': 'button',
                                     'text': 'Home Syringe'},
                    'load_syringe': {'type': 'button',
                                     'text': 'Move to load syringe position'},
                    'set_speed': {'type': 'textbox',
                                  'text': 'Syringe speed (mL / min)'},
                    'aspirate': {'type': 'textbox',
                                 'text': 'Aspirate volume (mL): '},
                    'dispense': {'type': 'textbox',
                                 'text': 'Dispense volume (mL): '},
                    'smart_dispense': {'type': 'textbox',
                                        'text': 'Smart dispense volume (mL): '}
                                 }
        info['controls'] = info['controls'] | controls
        return info
    
    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'home_syringe':
            # homes the syringe
            await self.run_syringe_until_idle(self.home())
        elif command == 'load_syringe':
            # moves the syringe to the load position (half full stroke).
            # NOTE: Does not move any valves, so valves must be in safe position
            await self.run_syringe_until_idle(self.move_absolute(int(self._get_max_position() / 2)))
        elif command == 'aspirate':
            # aspirates given volume
            # NOTE: Does not move any valves, so valves must be in safe position
            await self.run_syringe_until_idle(self.aspirate(float(data['value']) * 1000, self._flow_rate(self._speed)))
        elif command == 'dispense':
            # dispenses given volume
            # NOTE: Does not move any valves, so valves must be in safe position
            await self.run_syringe_until_idle(self.dispense(float(data['value']) * 1000, self._flow_rate(self._speed)))
        elif command == 'smart_dispense':
            # smart dispenses given volume
            await self.smart_dispense(float(data['value']) * 1000, self._flow_rate(self._speed))
        elif command == 'set_speed':
            # set the speed. Do not use run_until_idle because on-the-fly changes are permitted
            await self.run_async(self.set_speed(float(data['value']) * 1000 / 60))
        elif command == 'soft_reset':
            await self.soft_reset()

    async def soft_reset(self) -> str:
        """Soft reset ('z') to use after syringe overloads
        """

        response, error = await self.query('zR')
        if error.error is not None:
            self.logger.error(f'{self}: Soft reset error {error}')

        await self.update_syringe_status()

        return error

    async def get_syringe_status(self) -> str:
        """Gets full status string of device

        Returns:
            str: six-bit binary string with status
        """
        
        response, error = await self.query('?10000')

        return format(int(response), '06b')

    async def set_high_resolution(self, high_resolution: bool) -> None:
        """Turns high resolution mode on or off

        Args:
            high_resolution (bool): turn high resolution on (True) or off (False)
        """
        
        response, error = await self.query(f'N{int(high_resolution)}R')
        if error.error is not None:
            self.logger.error(f'{self}: Error setting resolution: {error}')
        else:
            self._high_resolution = high_resolution

    def _full_stroke(self) -> int:
        """Calculates syringe stroke (# half steps for full volume) based on resolution mode

        Returns:
            float: stroke in half steps
        """

        return 48000 if self._high_resolution else 6000
    
    def _get_max_position(self) -> int:
        """Calculates the maximum position in half steps

        Returns:
            int: max position in half steps
        """

        return self._full_stroke() // 2

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
            self.logger.warning(f'{self}: Warning: clipping desired flow rate {desired_flow_rate} to lowest possible value {self._min_flow_rate()}')
            return self.minV
        elif desired_flow_rate > self._max_flow_rate():
            self.logger.warning(f'{self}: Warning: clipping desired flow rate {desired_flow_rate} to highest possible value {self._max_flow_rate()}')
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

        return float(V * self.syringe_volume) / 6000.
    
    def _stroke_length(self, desired_volume: float) -> int:
        """Calculates stroke length in steps

        Args:
            desired_volume (float): aspirate or dispense volume in uL

        Returns:
            int: stroke length in number of motor steps
        """

        return round(desired_volume * (self._get_max_position()) / self.syringe_volume)

    def _volume_from_stroke_length(self, strokes: int) -> float:
        """Calculates volume from a stroke length (inverts _stroke_length)

        Args:
            strokes (int): number of motor steps

        Returns:
            float: volume corresponding to stroke length
        """

        return strokes * (self.syringe_volume / (self._get_max_position()))

    async def update_syringe_status(self) -> int:
        """Reads absolute position of syringe

        Returns:
            int: absolute position of syringe in steps
        """

        response, error = await self.query('?')
        
        self.syringe_position = int(response)

        if error.error is not None:
            self.logger.error(f'{self}: Error in update_syringe_status: {error}')
        await self.update_status()

        return error

    
    async def get_speed(self) -> str:
        """Reads current speed code of syringe
        
        Returns:
            int: speed code of syringe in steps/second"""

        response, error = await self.query('?2')

        self._speed = int(response)
        
        if error.error is not None:
            self.logger.error(f'{self}: Error in get_speed: {error}')

        return error


    async def set_speed(self, flow_rate: float) -> str:
        """Sets syringe speed to a specified flow rate

        Args:
            flow_rate (float): flow rate in uL / s
        """

        V = self._speed_code(flow_rate)
        self.logger.debug(f'Speed: {V}')
        response, error = await self.query(f'V{V}R')
        await self.run_async(self.get_speed())

        if error.error is not None:
            self.logger.error(f'{self}: Syringe move error {error}')

        return error

    async def poll_syringe_until_idle(self) -> None:
        """Polls device until idle

        Returns:
            str: error string
        """

        timer = PollTimer(self.syringe_poll_delay, self.address_code)

        while (not self.idle):
            # run update_status and start the poll_delay timer
            await asyncio.gather(self.update_syringe_status(), timer.cycle(), self.trigger_update())

            # wait until poll_delay timer has ended before asking for new status.
            await timer.wait_until_set()

        await self.trigger_update()

    async def run_syringe_until_idle(self, cmd: asyncio.Future) -> None:
        """
        Sends from serial connection and waits until idle
        """

        self.idle = False
        await cmd
        await self.poll_syringe_until_idle()
    
    async def aspirate(self, volume: float, flow_rate: float) -> None:
        """Aspirate (Pick-up)

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        await self.update_syringe_status()
        syringe_position = self.syringe_position
        stroke_length = self._stroke_length(volume)
        max_position = self._get_max_position()
        self.logger.debug(f'Stroke length: {stroke_length} out of full stroke {self._get_max_position()}')

        if max_position < (stroke_length + syringe_position):
            self.logger.error(f'{self}: Invalid syringe move from current position {syringe_position} with stroke length {stroke_length} and maximum position {max_position}')
            
            # TODO: this is a hack to clear the response queue...need to fix this
            #await self.update_status()
        else:
            await self.run_until_idle(self.set_speed(flow_rate))
            response, error = await self.query(f'P{stroke_length}R')
            if error.error is not None:
                self.logger.error(f'{self}: Syringe move error {error}')

    async def dispense(self, volume: float, flow_rate: float) -> None:
        """Dispense

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        await self.update_syringe_status()
        syringe_position = self.syringe_position
        stroke_length = self._stroke_length(volume)
        self.logger.debug(f'Stroke length: {stroke_length} out of full stroke {self._get_max_position()}')

        if (syringe_position - stroke_length) < 0:
            self.logger.error(f'{self}: Invalid syringe move from current position {syringe_position} with stroke length {stroke_length} and minimum position 0')
            #await self.update_status()
        else:
            await self.run_until_idle(self.set_speed(flow_rate))
            response, error = await self.query(f'D{stroke_length}R')
            if error.error is not None:
                self.logger.error(f'{self}: Syringe move error {error}')

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

        # check that aspiration and dispense positions are defined
        if (not hasattr(self.valve, 'aspirate_position')) | (not hasattr(self.valve, 'dispense_position')):
            self.logger.error(f'{self.name}: valve must have aspirate_position and dispense_position defined to use smart_dispense')
            return 0
        if (self.valve.aspirate_position is None) | (self.valve.dispense_position is None):
            self.logger.error(f'{self.name}: aspirate_position and dispense_position must be set to use smart_dispense')
            return 0
        
        # convert speeds to V factors
        aspirate_flow_rate = self.max_aspirate_flow_rate
        V_aspirate = self._speed_code(aspirate_flow_rate)
        V_dispense = self._speed_code(dispense_flow_rate)

        # calculate total volume in steps
        total_steps = self._stroke_length(volume)
        self.logger.debug(f'{self.name}: smart dispense requested {total_steps} steps')
        if total_steps <= 0:
            self.logger.warning(f'{self.name}: volume is not positive, smart_dispense terminating')
            return 0

        # calculate max number of steps
        full_stroke = self._get_max_position()

        # update current syringe position (usually zero)
        await self.update_syringe_status()
        syringe_position = self.syringe_position
        current_position = copy.copy(syringe_position)
        self.logger.debug(f'{self.name}: smart dispense, syringe at {current_position}')

        # calculate number of aspirate/dispense operations and volume per operation
        # if there is already enough volume in the syringe, just do a single dispense
        total_steps_dispensed = 0
        if current_position >= total_steps:
            # switch valve and dispense
            self.logger.debug(f'{self.name}: smart dispense dispensing {total_steps} at V {V_dispense}')
            await self.run_until_idle(self.move_valve(self.valve.dispense_position))
            await self.run_until_idle(self.set_speed(dispense_flow_rate))
            try:
                await self.run_syringe_until_idle(self.move_absolute(current_position - total_steps, interrupt_index))
            except asyncio.CancelledError:
                await self.run_until_idle(self.stop())
                await self.update_syringe_status()

            total_steps_dispensed += current_position - self.syringe_position
            
        else:
            # number of full_volume loops plus remainder
            #print(total_steps, full_stroke)
            stroke_steps = [full_stroke] * (total_steps // full_stroke) + [total_steps % full_stroke]
            for stroke in stroke_steps:
                if stroke > 0:
                    self.logger.debug(f'{self.name}: smart dispense aspirating {stroke - current_position} at V {V_aspirate}')
                    # switch valve and aspirate
                    await self.run_until_idle(self.move_valve(self.valve.aspirate_position))
                    await self.run_until_idle(self.set_speed(aspirate_flow_rate))
                    await self.run_syringe_until_idle(self.move_absolute(stroke))
                    position_after_aspirate = copy.copy(self.syringe_position)

                    # switch valve and dispense; run_syringe_until_idle updates self.syringe_position
                    self.logger.debug(f'{self.name}: smart dispense dispensing all at V {V_dispense}')
                    await self.run_until_idle(self.move_valve(self.valve.dispense_position))
                    await self.run_until_idle(self.set_speed(dispense_flow_rate))
                    try:
                        await self.run_syringe_until_idle(self.move_absolute(0, interrupt_index))
                    except asyncio.CancelledError:
                        await self.run_until_idle(self.stop())
                        await self.update_syringe_status()

                    position_change = position_after_aspirate - self.syringe_position
                    total_steps_dispensed += position_change

                    if (stroke == position_change):
                        # update current position and go to next step
                        current_position = copy.copy(self.syringe_position)
                    else:
                        # stop! do not do continued strokes because the interrupt was triggered
                        break

        return self._volume_from_stroke_length(total_steps_dispensed)

    async def move_absolute(self, position: int, interrupt_index: int | None = None) -> None:
        """Low-level method for moving the syringe to an absolute position

        Args:
            position (int): syringe position in steps
            interrupt_index (int | None, optional): index of condition to interrupt syringe movement.
                See Hamilton PSD manual for codes. Defaults to None.

        """

        interrupt_string = '' if interrupt_index is None else f'i{interrupt_index}'

        response, error = await self.query(interrupt_string + f'A{position}R')
        if error.error is not None:
            self.logger.error(f'{self}: Syringe move error {error} for move to position {position} with V {self._speed}')

    async def home(self) -> None:
        """Homes syringe using maximum flow rate
        """

        await self.set_speed(self.max_dispense_flow_rate)
        await self.move_absolute(0)

class SimulatedHamiltonSyringePump(SimulatedHamiltonValvePositioner, SyringePumpValvePositioner):
    """Hamilton syringe pump device. Includes both a syringe motor and a built-in valve positioner.
    """

    def __init__(self,
                 valve: SyringeValveBase,
                 syringe_volume: float = 5000,
                 high_resolution = False,
                 id: str = None,
                 name: str = None,
                 ) -> None:
        SimulatedHamiltonValvePositioner.__init__(self, valve, device_id=id, name=name)
        SyringePumpValvePositioner.__init__(self, valve, syringe_volume, device_id=self.id, name=self.name)

        # Syringe poll delay
        self.syringe_poll_delay = 0.2

        # default high resolution mode is False
        self._high_resolution = high_resolution

        # min and max V speed codes
        minV, maxV = 2, 10000

        # allow custom max flow rate
        self.max_aspirate_flow_rate = self._flow_rate(maxV)
        self.max_dispense_flow_rate = self._flow_rate(maxV)
        self.min_flow_rate = self._flow_rate(minV)
        self._speed = self.min_flow_rate

    @property
    def syringe_speed(self) -> float:
        return self._speed

    @property
    def state(self) -> HamiltonSyringePumpState:
        """Gets the current state
        """

        return HamiltonSyringePumpState(id=self.id,
                           name=self.name,
                           idle=self.idle,
                           initialized=self.initialized,
                           reserved=self.reserved,
                           error=self.error,
                           digital_outputs=self.digital_outputs,
                           valve_state=self.valve.state,
                           syringe_state=self.syringe_state)

    async def get_info(self) -> dict:
        """Gets information about valve positioner

        Returns:
            dict: information dictionary
        """
        info = await super().get_info()

        controls = {'soft_reset' : {'type': 'button',
                                    'text': 'Soft Reset'},
                    'home_syringe': {'type': 'button',
                                     'text': 'Home Syringe'},
                    'load_syringe': {'type': 'button',
                                     'text': 'Move to load syringe position'},
                    'set_speed': {'type': 'textbox',
                                  'text': 'Syringe speed (mL / min)'},
                    'aspirate': {'type': 'textbox',
                                 'text': 'Aspirate volume (mL): '},
                    'dispense': {'type': 'textbox',
                                 'text': 'Dispense volume (mL): '},
                    'smart_dispense': {'type': 'textbox',
                                        'text': 'Smart dispense volume (mL): '}
                                 }
        info['controls'] = info['controls'] | controls
        return info
    
    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

        if command == 'home_syringe':
            # homes the syringe
            await self.run_syringe_until_idle(self.home())
        elif command == 'set_speed':
            await self.run_until_idle(self.set_speed(float(data['value']) * 1000 / 60))
        elif command == 'aspirate':
            # aspirates given volume
            # NOTE: Does not move any valves, so valves must be in safe position
            await self.run_syringe_until_idle(self.aspirate(float(data['value']) * 1000, self._speed))
        elif command == 'dispense':
            # dispenses given volume
            # NOTE: Does not move any valves, so valves must be in safe position
            await self.run_syringe_until_idle(self.dispense(float(data['value']) * 1000, self._speed))
        elif command == 'smart_dispense':
            # smart dispenses given volume
            await self.smart_dispense(float(data['value']) * 1000, self._speed)

    def _flow_rate(self, V: int) -> float:
        """Calculates actual flow rate from speed code parameter (V)

        Args:
            V (float): speed code in half-steps / second

        Returns:
            float: flow rate in uL / s
        """

        return float(V * self.syringe_volume) / 6000.

    async def poll_syringe_until_idle(self) -> None:
        """Polls device until idle

        Returns:
            str: error string
        """

        timer = PollTimer(self.syringe_poll_delay, self.name)

        while (not self.idle):
            # run update_status and start the poll_delay timer
            await asyncio.gather(self.update_syringe_status(), timer.cycle(), self.trigger_update())

            # wait until poll_delay timer has ended before asking for new status.
            await timer.wait_until_set()

        await self.trigger_update()

    async def run_syringe_until_idle(self, cmd: asyncio.Future) -> None:
        """
        Sends from serial connection and waits until idle
        """

        self.idle = False
        await asyncio.gather(cmd, self.poll_syringe_until_idle())

    async def set_speed(self, flow_rate):
        self.idle = True
        return await super().set_speed(flow_rate)

    async def aspirate(self, volume: float, flow_rate: float) -> None:
        """Aspirate (Pick-up)

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        syringe_position = self.syringe_position
        stroke_length = volume
        max_position = self.syringe_volume
        self.logger.debug(f'Stroke length: {stroke_length} out of full stroke {self.syringe_volume}')

        if max_position < (stroke_length + syringe_position):
            self.logger.error(f'{self}: Invalid syringe move from current position {syringe_position} with stroke length {stroke_length} and maximum position {max_position}')
            
            # TODO: this is a hack to clear the response queue...need to fix this
            #await self.update_status()
        else:
            #await self.set_speed(flow_rate)
            self.idle = False
            total_time = volume / flow_rate
            n_steps = total_time // (self.poll_delay / 2.0)
            for _ in range(int(n_steps)):
                await asyncio.sleep(total_time / n_steps)
                self.syringe_position += volume / n_steps

        self.idle = True

    async def dispense(self, volume: float, flow_rate: float) -> None:
        """Dispense

        Args:
            volume (float): volume in uL
            flow_rate (float): flow rate in uL / s
        """

        syringe_position = self.syringe_position
        stroke_length = volume
        self.logger.debug(f'Stroke length: {stroke_length} out of full stroke {self.syringe_volume}')

        if (syringe_position - stroke_length) < 0:
            self.logger.error(f'{self}: Invalid syringe move from current position {syringe_position} with stroke length {stroke_length} and minimum position 0')
            #await self.update_status()
        else:
            #await self.set_speed(flow_rate)
            self.idle = False
            total_time = volume / flow_rate
            n_steps = total_time // (self.poll_delay / 2.0)
            for _ in range(int(n_steps)):
                await asyncio.sleep(total_time / n_steps)
                self.syringe_position -= volume / n_steps

        self.idle = True

    async def move_absolute(self, position, interrupt_index = None):
        self.idle = False
        volume = position - self.syringe_position
        total_time = abs(volume / self._speed)
        self.logger.debug(f'total time {total_time} for volume {volume} at speed {self._speed   }')
        n_steps = total_time // (self.poll_delay / 2.0) + 1
        for _ in range(int(n_steps)):
            await asyncio.sleep(total_time / n_steps)
            self.syringe_position += volume / n_steps
        
        self.idle = True

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

        # check that aspiration and dispense positions are defined
        if (not hasattr(self.valve, 'aspirate_position')) | (not hasattr(self.valve, 'dispense_position')):
            self.logger.error(f'{self.name}: valve must have aspirate_position and dispense_position defined to use smart_dispense')
            return 0
        if (self.valve.aspirate_position is None) | (self.valve.dispense_position is None):
            self.logger.error(f'{self.name}: aspirate_position and dispense_position must be set to use smart_dispense')
            return 0
        
        # convert speeds to V factors
        aspirate_flow_rate = self.max_aspirate_flow_rate
        V_aspirate = aspirate_flow_rate
        V_dispense = dispense_flow_rate

        # calculate total volume in steps
        total_steps = volume
        full_stroke = self.syringe_volume

        # update current syringe position (usually zero)
        syringe_position = self.syringe_position
        current_position = copy.copy(syringe_position)

        # calculate number of aspirate/dispense operations and volume per operation
        # if there is already enough volume in the syringe, just do a single dispense
        total_steps_dispensed = 0
        if current_position >= total_steps:
            # switch valve and dispense
            await self.run_until_idle(self.move_valve(self.valve.dispense_position))
            await self.run_until_idle(self.set_speed(dispense_flow_rate))
            await self.run_syringe_until_idle(self.move_absolute(current_position - total_steps, interrupt_index))
            total_steps_dispensed += current_position - self.syringe_position
        else:
            # number of full_volume loops plus remainder
            #print(total_steps, full_stroke)
            stroke_steps = [full_stroke] * int(total_steps // full_stroke) + [total_steps % full_stroke]
            for stroke in stroke_steps:
                if stroke > 0:
                    self.logger.debug(f'{self.name}: smart dispense aspirating {stroke - current_position} at V {V_aspirate}')
                    # switch valve and aspirate
                    await self.run_until_idle(self.move_valve(self.valve.aspirate_position))
                    await self.run_until_idle(self.set_speed(aspirate_flow_rate))
                    await self.run_syringe_until_idle(self.move_absolute(stroke))
                    position_after_aspirate = copy.copy(self.syringe_position)
                    self.logger.debug(f'position after aspirate: {position_after_aspirate}')

                    # switch valve and dispense; run_syringe_until_idle updates self.syringe_position
                    self.logger.debug(f'{self.name}: smart dispense dispensing all at V {V_dispense}')
                    await self.run_until_idle(self.move_valve(self.valve.dispense_position))
                    await self.run_until_idle(self.set_speed(dispense_flow_rate))
                    await self.run_syringe_until_idle(self.move_absolute(0, interrupt_index))
                    position_change = position_after_aspirate - self.syringe_position
                    total_steps_dispensed += position_change

                    if (stroke == position_change):
                        # update current position and go to next step
                        current_position = copy.copy(self.syringe_position)
                    else:
                        # stop! do not do continued strokes because the interrupt was triggered
                        break

        self.logger.debug(f'{self.name}: smart dispense complete')

        return total_steps_dispensed

class SmoothFlowSyringePump(HamiltonSyringePump):

    def __init__(self, serial_instance: HamiltonSerial, address: str, valve: SyringeValveBase, syringe_volume: float = 5000, name=None) -> None:
        super().__init__(serial_instance, address, valve, syringe_volume, True, name)

        # note that these are now "u" for the smooth flow pump
        self.minV, self.maxV = 400, 816000
        self._speed = 400

    def _full_stroke(self) -> int:
        """Calculates syringe stroke (# steps for full volume)

        Returns:
            float: stroke in steps
        """

        return 192000 if self._high_resolution else 24000
    
    def _get_max_position(self) -> int:
        """Calculates the maximum position in half steps

        Returns:
            int: max position in half steps
        """

        return self._full_stroke()
        
    def _speed_code(self, desired_flow_rate: float) -> int:
        """Calculates speed code (parameter u, see SF PSD/4 manual Appendix H) based on desired
            flow rate and syringe parameters

        Args:
            desired_flow_rate (float): desired flow rate in uL / s

        Returns:
            int: u (steps per minute)
        """

        #calcV = float(desired_flow_rate * 6000) / self.syringe_volume

        if desired_flow_rate < self._min_flow_rate():
            self.logger.warning(f'{self}: Warning: clipping desired flow rate {desired_flow_rate} to lowest possible value {self._min_flow_rate()}')
            return self.minV
        elif desired_flow_rate > self._max_flow_rate():
            self.logger.warning(f'{self}: Warning: clipping desired flow rate {desired_flow_rate} to highest possible value {self._max_flow_rate()}')
            return self.maxV
        else:
            return round(float(desired_flow_rate * 60 * 192000) / self.syringe_volume)

    def _flow_rate(self, V: int) -> float:
        """Calculates actual flow rate from speed code parameter (V)

        Args:
            V (float): speed code in steps / minute ("u" in the smooth flow user manual)

        Returns:
            float: flow rate in uL / s
        """

        return float(V * self.syringe_volume) / 192000. / 60.

    async def set_speed(self, flow_rate: float) -> str:
        """Sets syringe speed to a specified flow rate

        Args:
            flow_rate (float): flow rate in uL / s
        """

        V = self._speed_code(flow_rate)
        self.logger.info(f'Speed: {V}')
        response, error = await self.query(f'u{V}R')
        await self.run_async(self.get_speed())

        if error.error is not None:
            self.logger.error(f'{self}: Syringe move error {error}')

        return error

if __name__ == '__main__':

    logging.basicConfig(
                        format='%(asctime)s.%(msecs)03d %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)

    from valve import SyringeLValve
    from webview import run_socket_app


    async def main():
        sp = SimulatedHamiltonSyringePump(SyringeLValve(4, name='syringe_LValve'), 5000, False, name='syringe_pump')
        app = sp.create_web_app()
        runner = await run_socket_app(app, 'localhost', 5013)
        await sp.initialize()
        #await sp.smart_dispense(200, 10 * 1000 / 60)
        try:
            await asyncio.Event().wait()
        finally:
            logging.info('Cleaning up...')
            asyncio.gather(
                           runner.cleanup())

    asyncio.run(main(), debug=True)

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

class SimulatedSensoronHamiltonDevice(BubbleSensorBase):

    def __init__(self, device: SimulatedHamiltonBase, id: str | None = None, name: str = '') -> None:
        super().__init__(id, name)
        self.device = device

    async def initialize(self) -> None:
        ...

    async def read(self) -> bool:
        """Read bubble sensor

        Returns:
            bool: true if liquid in line; false if air
        """

        return random.choice([True, False])

class SyringePumpwithBubbleSensor(HamiltonSyringePump):
    """DEPRECATED
        Syringe pump with one or more bubble sensors driven by digital outputs and addressing digital inputs of the same index.
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