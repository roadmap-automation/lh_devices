from typing import Tuple, List, Coroutine
import pyvisa
from threading import Event, Lock
from queue import Queue, Empty, Full
from copy import copy
import time
import asyncio
import logging
from enum import Enum

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import convolve
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from connections import Port

plt.set_loglevel('warning')

from HamiltonComm import HamiltonSerial
from HamiltonDevice import HamiltonSyringePump, PollTimer, HamiltonValvePositioner
from assemblies import AssemblyBase, Mode
from valve import SyringeLValve, SyringeValveBase, DistributionValve
from webview import run_socket_app

from labjack import ljm

class YValve(DistributionValve):

    def __init__(self, position: int = 0, ports: List[Port] = [], name=None) -> None:
        super().__init__(3, position, ports, name)
        self.hamilton_valve_code = 0

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
            logging.warning(f'{self}: Warning: clipping desired flow rate {desired_flow_rate} to lowest possible value {self._min_flow_rate()}')
            return self.minV
        elif desired_flow_rate > self._max_flow_rate():
            logging.warning(f'{self}: Warning: clipping desired flow rate {desired_flow_rate} to highest possible value {self._max_flow_rate()}')
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
        logging.info(f'Speed: {V}')
        response, error = await self.query(f'u{V}R')
        await self.run_async(self.get_speed())

        if error:
            logging.error(f'{self}: Syringe move error {error}')

        return error

class SyringePumpRamp(SmoothFlowSyringePump):

    def __init__(self, serial_instance: HamiltonSerial, address: str, valve: SyringeValveBase, syringe_volume: float = 5000, name=None) -> None:
        super().__init__(serial_instance, address, valve, syringe_volume, name)

    async def ramp_speed(self, volume: float, flow_rates: List[float], delay: float = 0, reverse=True) -> None:
        """Ramp speed using program of flow rates such that same time is spent
            on each flow rate

        Args:
            volume (float): total volume in uL
            flow_rates: List[float]: flow rates in uL / s
            delay (float, Optional): time to delay before starting ramp. Default 0 seconds, must
                                        be greater than 5 ms.
        """

        if reverse:
            #volume /= 2.0
            flow_rates = flow_rates + flow_rates[::-1]
        
        delay *= 1000 # now in ms
        delay_list = [30000] * (delay // 30000) + [delay % 30000]

        stroke_length = self._stroke_length(volume)
        step_time = volume / sum(flow_rates)

        itime = time.time()
        data = []

        # set up timer
        timer = PollTimer(step_time, f'Ramp PollTimer at address {self.address_code}')

        # start move and timer simultaneously
        #delay_cmd = ''.join([f'M{round(d)}' if d >= 5 else '' for d in delay_list])
        await asyncio.sleep(delay / 1000)
        await self.query(f'u{self._speed_code(flow_rates[0])}D{stroke_length}R')
        await timer.cycle()
        cur_fr = flow_rates[0]
        data.append((time.time() - itime, self._flow_rate(self._speed_code(flow_rates[0]))))
        #if error:
        #    logging.error(f'{self}: Syringe move error {error}')
        for fr in flow_rates[1:]:
            # wait until poll_delay timer has ended before setting new flow rate
            await timer.wait_until_set()

            if fr != cur_fr:
            # set new speed and start the poll_delay timer
                await asyncio.gather(self.query(f'u{self._speed_code(fr)}'), timer.cycle())
                cur_fr = fr
            else:
                await timer.cycle()
            data.append((time.time() - itime, self._flow_rate(self._speed_code(fr))))
#
#            response, error = await self.query(f'V{self._speed_code(fr)}')
            #if error:
            #    logging.error(f'{self}: Syringe async velocity set error {error}')

        return data

class USBDriverBase:

    def __init__(self, timeout = 0.05, name='USBDriver') -> None:

        self.name = name
        self.timeout = timeout
        self.stop_event = Event()
        self.active_futures: List[asyncio.Future] = []

        self.lock: Lock = Lock()
        self.inqueue: asyncio.Queue = asyncio.Queue()
        self.outqueue: asyncio.Queue = asyncio.Queue(1)

    async def start(self, loop: asyncio.AbstractEventLoop | None = None):

        logging.info(f'Starting {self.name} thread...')

        if loop is None:
            loop = asyncio.get_event_loop()

        return await asyncio.to_thread(self.run, loop=asyncio.get_event_loop())

    def stop(self):
        self.stop_event.set()
        for future in self.active_futures:
            future.cancel()

        self.active_futures = []
    
    def clear(self):
        self.stop_event.clear()

    def open(self):
        pass

    def close(self):
        pass

    def run(self, loop: asyncio.AbstractEventLoop):
        """Synchronous code to interact with the instrument"""

        # open instrument resource
        self.open()
        self.clear()
        while not self.stop_event.is_set():

            cmd = None

            # TODO: figure out how to do this with a timeout (to be responsive to stop signal)
            future = asyncio.run_coroutine_threadsafe(self.inqueue.get(), loop)
            self.active_futures.append(future)
            cmd = future.result()
            self.active_futures.pop(self.active_futures.index(future))
            
            if cmd is not None:

                logging.info('%s => %s', self.name, cmd)

                # write value to instrument

                # read value from instrument
                response = None

                # write response to outqueue (blocks until value is read)
                future = asyncio.run_coroutine_threadsafe(self.outqueue.put(response), loop)
                self.active_futures.append(future)
                future.result()
                self.active_futures.pop(self.active_futures.index(future))
            
        self.close()

    async def write(self, cmd: str) -> None:
        """Writes command to queue"""
        #cmd.replace(' ', '\\s')
        task = asyncio.create_task(self.inqueue.put(cmd))
        self.active_futures.append(task)
        try:
            await task
        except asyncio.CancelledError:
            print(f'Stopping {self.name} inqueue.put...')
        self.active_futures.pop(self.active_futures.index(task))

    async def query(self, cmd: str) -> str:
        """Helper function for performing queries"""
        await self.write(cmd)
        res = None
        task = asyncio.create_task(self.outqueue.get())
        self.active_futures.append(task)
        try:
            await task
            res = task.result()
        except asyncio.CancelledError:
            print(f'Stopping {self.name} outqueue.get...')
        self.active_futures.pop(self.active_futures.index(task))

        return res

class KeithleyDriver(USBDriverBase):

    def __init__(self, timeout=0.05, name='KeithleyDriver') -> None:
        super().__init__(timeout, name)

        self.rm = None
        self.instr = None

    def open(self, model='2450'):
        
        rm = pyvisa.ResourceManager()
        res_id = next(res for res in rm.list_resources() if model in res)
        logging.info('Connecting to %s', res_id)

        instr = rm.open_resource(res_id)

        self.rm = rm
        self.instr = instr

    def close(self):

        self.instr.close()
        self.rm.close()

    def run(self, loop: asyncio.AbstractEventLoop):
        """Synchronous code to interact with the Keithley"""

        # open instrument resource
        self.open()
        self.clear()
        while not self.stop_event.is_set():

            cmd = None

            # TODO: figure out how to do this with a timeout (to be responsive to stop signal)
            future = asyncio.run_coroutine_threadsafe(self.inqueue.get(), loop)
            self.active_futures.append(future)
            cmd = future.result()
            self.active_futures.pop(self.active_futures.index(future))
            
            if cmd is not None:

                logging.debug('%s => %s', self.name, cmd)

                # write value to instrument
                #with self.lock:
                self.instr.write(cmd)
                
                # if command is a query command
                if '?' in cmd:
                    #with self.lock:
                    response: str = self.instr.read()

                    logging.debug('%s <= %s', self.name, response[:-1])

                    # write response to outqueue (blocks until value is read)
                    future = asyncio.run_coroutine_threadsafe(self.outqueue.put(response), loop)
                    self.active_futures.append(future)
                    future.result()
                    self.active_futures.pop(self.active_futures.index(future))
            
        self.close()

    async def setup_source_current_measure_voltage(self, current: float = 0, voltage_limit: float = 0.02, time: float|None = None, additional_commands: List[str] = []):
        if time is None:
            time = 100000
        setup_commands = [
                        ':SOUR:FUNC CURR',
                        f':SOUR:CURR {current}',
                        f':SOUR:CURR:VLIM {voltage_limit:0.3f}',
                        ':SENS:FUNC "VOLT"',
                        ':VOLT:RSEN 0',
                        ':VOLT:NPLC 2',
                        ':VOLT:RANG:AUTO ON'] + \
                           additional_commands + \
                        [':TRIG:LOAD "Empty"',
                         f':TRIG:LOAD "DurationLoop", {time:0.3f}']

        for cmd in setup_commands:
            await self.write(cmd)

    async def setup_source_voltage_measure_current(self, voltage: float = 0, time: float|None = None, additional_commands: List[str] = []):
        if time is None:
            time = 100000
        setup_commands = [
                            #'*RST',
                            ':SOUR:FUNC VOLT',
                            f':SOUR:VOLT {voltage}',
                            ':SENS:FUNC "CURR"',
                            ':CURR:RSEN 0',
                            ':CURR:NPLC 2',
                            ':CURR:RANG:AUTO ON'] + \
                        additional_commands + \
                        [':TRIG:LOAD "Empty"',
                         f':TRIG:LOAD "DurationLoop", {time:0.3f}']
        
        for cmd in setup_commands:
            await self.write(cmd)

class CommandType(Enum):
    READ = 'read'
    WRITE = 'write'

class LabJackDriver(USBDriverBase):

    def __init__(self, timeout=0.05, name='LabJackDriver') -> None:
        super().__init__(timeout, name)

        self.stop_stream: Event = Event()
        self.instr = None

        self.channel_map = {'AIN0': 0,
                            'AIN1': 1,
                            'AIN2': 2,
                            'AIN3': 3}
        
    def open(self, model='T7'):
        
        handle = ljm.openS("ANY","ANY","ANY")
        result = ljm.eReadName(handle, "PRODUCT_ID")
        logging.info('Connecting to %s', result)

        #print(ljm.eReadName(lj.instr, 'AIN3_RANGE'))
        #ljm.eWriteName(handle, 'AIN3_RESOLUTION_INDEX', 7)
        #print(ljm.eReadName(lj.instr, 'AIN3_RESOLUTION_INDEX'))

        self.instr = handle

    def close(self):

        ljm.close(self.instr)
        self.instr = None

    def run(self, loop: asyncio.AbstractEventLoop):
        """Synchronous code to interact with the Labjack via call-and-response"""

        # open instrument resource
        if self.instr is None:
            self.open()

        self.clear()
        while not self.stop_event.is_set():

            cmd = None

            # TODO: figure out how to do this with a timeout (to be responsive to stop signal)
            future = asyncio.run_coroutine_threadsafe(self.inqueue.get(), loop)
            self.active_futures.append(future)
            cmd: Tuple[CommandType, str, str | None] | None = future.result()
            self.active_futures.pop(self.active_futures.index(future))
            
            if cmd is not None:

                name, value, command_type = cmd

                logging.debug('%s => %s', self.name, cmd)

                if command_type == CommandType.WRITE:
                    # write value to instrument
                    #with self.lock:
                    ljm.eWriteName(self.instr, name, value)
                
                # if command is a query command
                else:
                    #with self.lock:
                    response = ljm.eReadName(self.instr, name)

                    logging.debug('%s <= %s', self.name, response)

                    # write response to outqueue (blocks until value is read)
                    future = asyncio.run_coroutine_threadsafe(self.outqueue.put(response), loop)
                    self.active_futures.append(future)
                    future.result()
                    self.active_futures.pop(self.active_futures.index(future))
            
        self.close()

    async def query(self, cmd: str, value: str | None = None, command_type: CommandType = CommandType.WRITE) -> str:
        #await self.write()
        return await super().query((cmd, value, command_type))

    def stream(self, ScansPerSecond: float = 1,
                     addresses: List[int] = [6],
                     ScanRate: float = 500) -> Tuple[float, np.ndarray]:
        """Synchronous code to interact with the Labjack via streaming

        Args:
            ScansPerSecond (int, optional): Scans per second. Defaults to 1.
            addresses (List[int], optional): integer addresses to read (see Modbus Map). Defaults to [6] ('AIN3').
            ScanRate (float, optional): Samples per second. Defaults to 500.

        Returns:
            list: all data returned from stream
        """
        if self.instr is None:
            self.open()

        self.stop_stream.clear()
        
        all_data = []

        def _stream_callback(handle):
            data, deviceScanBacklog, ljmScanBacklog = ljm.eStreamRead(handle)
            #print(deviceScanBacklog, ljmScanBacklog)
            data = np.array(data)
            data = data.reshape((len(addresses), -1), order='F')
            all_data.append(data)

        ljm.eWriteName(self.instr, 'STREAM_SETTLING_US', 10)
        actual_scanrate = ljm.eStreamStart(self.instr, round(ScanRate / ScansPerSecond), len(addresses), addresses, ScanRate)
        ljm.setStreamCallback(self.instr, _stream_callback)

        self.stop_stream.wait()
        _stream_callback(self.instr)
        ljm.eStreamStop(self.instr)

        all_data = np.concatenate(all_data, axis=1)

        return actual_scanrate, all_data    

class PressureSensorwithInAmp(LabJackDriver):

    def __init__(self, channel: str = 'AIN2', gain=201, sensor_voltage = 2.5, timeout=0.05, name='LabJackDriver') -> None:
        super().__init__(timeout, name)

        self.channel = channel
        self.gain = gain
        self.sensor_voltage = sensor_voltage

    def open(self, model='T7') -> None:
        super().open(model)

        ljm.eWriteName(self.instr, self.channel + '_RANGE', 1)

    def stream(self, ScansPerSecond: float = 1, ScanRate: float = 500) -> Tuple[float, np.ndarray]:
        sample_rate, data = super().stream(ScansPerSecond, [self.channel_map[self.channel] * 2], ScanRate)
        data /= self.gain

        return sample_rate, data
    
class PressureSensorDifferential(LabJackDriver):

    def __init__(self, channel: str = 'AIN0', sensor_voltage: float = 5.0, timeout=0.05, name='LabJackDriver') -> None:
        super().__init__(timeout, name)

        self.channel = channel
        self.sensor_voltage = sensor_voltage
        self.gain = 1

    def open(self, model='T7'):
        super().open(model)

        # set up differential measurement
        negative_channel = self.channel_map[self.channel] + 1
        ljm.eWriteName(self.instr, self.channel + '_RANGE', 0.01)
        ljm.eWriteName(self.instr, self.channel + '_NEGATIVE_CH', negative_channel)

        # set up output voltage
        #ljm.eWriteName(self.instr, self.dac_channel, self.dac_output)

    def stream(self, ScansPerSecond: float = 1, ScanRate: float = 500) -> Tuple[float, np.ndarray]:
        sample_rate, data = super().stream(ScansPerSecond, [self.channel_map[self.channel] * 2], ScanRate)
        data /= self.gain

        return sample_rate, data

class PressureSensorDifferentialDouble(LabJackDriver):

    def __init__(self, channel_high: str = 'AIN0', channel_low: str='AIN2', sensor_voltage: float = 5.0, timeout=0.05, name='LabJackDriver') -> None:
        super().__init__(timeout, name)

        self.channel_high = channel_high
        self.channel_low = channel_low
        self.sensor_voltage = sensor_voltage
        self.gain = 1

    def open(self, model='T7'):
        super().open(model)

        # set up differential measurement
        for channel in [self.channel_high, self.channel_low]:
            negative_channel = self.channel_map[channel] + 1
            ljm.eWriteName(self.instr, channel + '_RANGE', 0.01)
            ljm.eWriteName(self.instr, channel + '_NEGATIVE_CH', negative_channel)

        # set up output voltage
        #ljm.eWriteName(self.instr, self.dac_channel, self.dac_output)

    def stream(self, ScansPerSecond: float = 1, ScanRate: float = 500) -> Tuple[float, np.ndarray]:
        sample_rate, data = super().stream(ScansPerSecond, [self.channel_map[self.channel_high] * 2, self.channel_map[self.channel_low] * 2], ScanRate * 2)
        data /= self.gain

        return sample_rate, data

class PollTask:

    def __init__(self, async_function: Coroutine, *args, **kwargs) -> None:
        self.fxn = async_function
        self.args = args
        self.kwargs = kwargs

    def func(self):

        return self.fxn(*self.args, **self.kwargs)

class StreamPot:

    def __init__(self, smu: KeithleyDriver, syringepump: SyringePumpRamp, psensor: LabJackDriver):
        self.smu = smu
        self.syringepump = syringepump
        self.psensor = psensor
        self.trigger: asyncio.Event = asyncio.Event()
        self._stop_polling: asyncio.Event = asyncio.Event()

        # TODO: Pressure sensor (create Labjack driver)
        # TODO: Syringe pump loop system

    async def start_polling(self, duration: float, sample_rate: float, tasks: List[PollTask] = []):
            # set up timer
            timer = PollTimer(sample_rate, 'P/V PollTimer')
            ntasks = len(tasks)
            data = []
            itime = time.time()
            curtime = copy(itime)

            while (curtime < itime + duration) & (not self._stop_polling.is_set()):

                poll_tasks = [t.func() for t in tasks] + [timer.cycle()]
                # set new speed and start the poll_delay timer
                responses = await asyncio.gather(*poll_tasks)
                
                #logging.info(speed_code_response)
                
                # get results and time stamp them
                curtime = time.time()
                data.append((time.time() - itime, *responses[:-1]))

                # wait until poll_delay timer has ended before setting new flow rate
                await timer.wait_until_set()
        
            return data

    async def measure_iv(self,
                         maxV: float = 0.001,
                         npts: int = 5,
                         time_per_point: float = 0.2) -> Tuple[float,
                                                               float,
                                                               np.ndarray,
                                                               np.ndarray]:

        # protect against V/mV confusion
        assert maxV < 0.1

        # run setup commands
        setup_commands = [
                          ':SOUR:FUNC VOLT',
                          ':SENS:FUNC "CURR"',
                          #':CURR:RANG:AUTO ON',
                        f':SOUR:SWE:VOLT:LIN -{maxV}, {maxV}, {npts}, -1',
                        ':INIT',
                        ]

        for cmd in setup_commands:
            await self.smu.write(cmd)

        # start polling pressure sensor
        #poll_task = asyncio.create_task(self.start_polling(1e9, 0.01, [PollTask(self.psensor.query, 'AIN2', None, CommandType.READ)]))
        sample_rate = 500
        stream = asyncio.create_task(asyncio.to_thread(self.psensor.stream, 1, sample_rate))

        # monitor output until correct number of points have been collected
        pointcount = 0
        status = 'RUNNING'
        while 'RUNNING' in status:
            await asyncio.sleep(time_per_point)
            status = await self.smu.query(':TRIG:STAT?')

        # stop pressure sensor
        self.psensor.stop_stream.set()


        #while pointcount < npts:
        #    await asyncio.sleep(time_per_point)
        
        #    print(await(self.smu.query(':TRIG:STAT?')))

        pointcount = int(await self.smu.query(':TRAC:ACT? "defbuffer1"'))
        # read all data        
        data = await self.smu.query(f':TRAC:DATA? 1, {pointcount}, "defbuffer1", SOUR, READ')

        # get pressure data result
        actual_sample_rate, pdata = await stream
        t = np.arange(pdata.shape[1]) / actual_sample_rate
        pressure_data = t, np.array(pdata[1]) - np.array(pdata[0])

        #plt.figure()
        #plt.plot(poll_data[0], poll_data[1])
    
        # format data into numpy arrays
        data = np.fromstring(data, sep=',')
        V, I = data.reshape((2, len(data) // 2), order='F')

        # Calculate resistance and voltage offset (in mV)
        p, cov = np.polyfit(I, V, 1, full=False, cov=True)
        R, b = p
        dR, db = np.sqrt(np.diag(cov))
        dV = b * 1e3
        ddV = db * 1e3

        return R, dR, dV, ddV, V, I, pdata

    async def measure_streaming_potential(self,
                        current: float = 0,
                        min_flow_rate: float = 0.1,
                        max_flow_rate: float = 1.0,
                        volume: float = 0.2,
                        baseline_duration: float = 2.0,
                        repeats: int = 1) -> Tuple[float,
                                                               float,
                                                               np.ndarray,
                                                               np.ndarray]:

        """Measures streaming potential using an up/down linear ramp in flow rate

        Returns:
            current (float, Optional): Clamping current in A. Default 0.
            min_flow_rate (float, Optional): starting flow rate. Default 0.1 mL/min.
            max_flow_rate (float, Optional): maximum flow rate. Default 1.0 mL/min.
            volume (float, Optional): total volume for run. Default 0.2 mL.
            baseline_duration (float, Optional): idle time before and after measurement. Default 2.0 s.
        """

        # 1. Calculate flow rate program for syringe pump

        MIN_STEP_TIME = 0.1 # no more than 10 points per second

        sp = self.syringepump

        vol = volume * 1000 # convert volume from mL to uL

        #await sp.move_valve(sp.valve.dispense_position)
        #await sp.home()
        await sp.move_valve(sp.valve.aspirate_position)
        await sp.run_until_idle(sp.aspirate(vol, sp.max_aspirate_flow_rate))
        await sp.move_valve(sp.valve.dispense_position)
        # allow system to settle after valve moves
        await asyncio.sleep(10)

        # calculate expected time
        actual_max_flow_rate = sp._flow_rate(sp._speed_code(max_flow_rate * 1000 / 60)) * 60 / 1000
        actual_min_flow_rate = sp._flow_rate(sp._speed_code(min_flow_rate * 1000 / 60)) * 60 / 1000
        expected_time = volume / (0.5 * (actual_max_flow_rate + actual_min_flow_rate)) * 60
        nsteps = round(expected_time / MIN_STEP_TIME / 2)

        flow_rates = np.linspace(actual_min_flow_rate, actual_max_flow_rate, nsteps, endpoint=True) / 60 * 1000

        # 2. set up the Keithley

        # run setup commands
        if True:
            await self.smu.setup_source_current_measure_voltage(current, time=expected_time + 2 * baseline_duration)

        else:
            await self.smu.setup_source_voltage_measure_current(current, time=expected_time + 2 * baseline_duration)

        # 3. Trigger syringe pump and measurement devices
        init_time = time.time()

        sample_rate = 500
        stream = asyncio.create_task(asyncio.to_thread(self.psensor.stream, 1, sample_rate))
        speed_data, _ = await asyncio.gather(sp.ramp_speed(vol, flow_rates.tolist(), delay=baseline_duration, reverse=True),
                                               self.smu.write(':INIT'))

        #print(poll_data, speed_data)

        # gets current syringe speed
        #response, error = await sp.query('?2')

        # 4. Wait until syringe pump is done
        #await sp.poll_until_idle()
        await asyncio.sleep(baseline_duration)
        print(f'Ramp time expected: {expected_time + 2.0 * baseline_duration}\n\tand elapsed: {time.time() - init_time}')

        # 5. Get measurement result
        # TODO: Figure out how to get status and wait until it's done measuring.
        #status = 'RUNNING'
        #while 'RUNNING' in status:
        #    await asyncio.sleep(0.1)
        #    status = await self.smu.query(':TRIG:STAT?')

        # stop pressure sensor and measurement
        self.psensor.stop_stream.set()
        await self.smu.write(':ABOR')
        await self.smu.write(':OUTP 0')

        # Get number of available points
        pointcount = int(await self.smu.query(':TRAC:ACT? "defbuffer1"'))

        # read all data        
        data = await self.smu.query(f':TRAC:DATA? 1, {pointcount}, "defbuffer1", READ, REL')

        # get pressure data result
        actual_sample_rate, pdata = await stream
        tp = np.arange(pdata.shape[1]) / actual_sample_rate
        pressure_data = tp, pdata[0] - pdata[1]

        # format data into numpy arrays
        data = np.fromstring(data, sep=',')
        V, t = data.reshape((2, pointcount), order='F')

        return V, t, pressure_data, speed_data, expected_time
    
    async def exchange(self, volume: float, flow_rate: float, baseline_duration: float = 10) -> None:

        sp = self.syringepump

        # set up voltage measurement
        await self.smu.setup_source_current_measure_voltage(0.0, time=None)

        # calculate aspirate time
        aspirate_time = volume / sp.max_aspirate_flow_rate

        async def aspirate():
            await sp.move_valve(sp.valve.aspirate_position)
            await sp.run_until_idle(sp.aspirate(volume, sp.max_aspirate_flow_rate))
            await sp.move_valve(sp.valve.dispense_position)

        # start aspirating immediately (simultaneously with waiting for baseline)
        aspirate_task = asyncio.create_task(aspirate())
        await aspirate_task

        # if baseline is longer than aspiration time, allow aspiration to proceed before triggering data collection
        if aspirate_time > baseline_duration:
            await asyncio.sleep(aspirate_time - baseline_duration)

        # start psensor stream
        sample_rate = 500
        stream = asyncio.create_task(asyncio.to_thread(self.psensor.stream, 1, sample_rate))

        # trigger measurement and wait baseline duration
        await asyncio.gather(asyncio.sleep(baseline_duration),
                             self.smu.write(':INIT'),
                             aspirate_task)

        # start the syringepump
        await sp.run_until_idle(sp.dispense(volume, flow_rate))

        # get a new baseline
        await asyncio.sleep(baseline_duration)

        # stop the measurements
        self.psensor.stop_stream.set()
        await self.smu.write(':ABOR')
        await self.smu.write(':OUTP 0')

        # collect the measurements
        actual_sample_rate, pdata = await stream
        tp = np.arange(pdata.shape[1]) / actual_sample_rate
        pressure_data = tp, np.array(pdata[1]) - np.array(pdata[0])

        # Get number of available points and read all data
        pointcount = int(await self.smu.query(':TRAC:ACT? "defbuffer1"'))
        data = await self.smu.query(f':TRAC:DATA? 1, {pointcount}, "defbuffer1", READ, REL')
        data = np.fromstring(data, sep=',')
        V, t = data.reshape((2, pointcount), order='F')

        return (t, V), (tp, pdata)
    
def response_function(yr, mag, sigma, dt):
    #print(sigma)
    xwdw = np.arange(max(10*sigma, 10))
    #wdw = np.exp(-0.5 * (xwdw - max(xwdw) / 2 - 2*sigma) ** 2 / sigma ** 2)
    wdw = np.exp(-(xwdw - max(xwdw) / 2 - dt) / sigma)
    wdw[xwdw < max(xwdw) / 2 - dt] = 0
    #plt.plot(wdw)
    return convolve(yr*mag, wdw / np.trapz(wdw, xwdw), 'same')


def analyze_streampot(t, V, baseline, baseline_pad = 0):
    """Subtract baseline from signal.
    """
    init_baseline = t <= baseline
    fit_init_baseline = t <= baseline - baseline_pad
    fit_final_baseline = t >= t[-1] - baseline + baseline_pad
    final_baseline = t >= t[-1] - baseline
    inner = (~init_baseline) & (~final_baseline)

    baseline_p = np.polyfit(t[fit_init_baseline], V[fit_init_baseline], 1)

    # subtract initial baseline
    mfVi = V - np.polyval(baseline_p, t)

    if False:
        plt.plot(V)
        plt.plot(V[fit_init_baseline])
        plt.plot(mfVi)
        plt.plot(np.polyval(baseline_p, t))
        plt.show()

    final_baseline_offset = np.average(mfVi[fit_final_baseline])
    global_offset = np.zeros_like(t)
    global_offset[final_baseline] = final_baseline_offset
    global_offset[inner] = np.linspace(0, final_baseline_offset, sum(inner) + 2, endpoint=True)[1:-1]

    mfVo = mfVi - global_offset

    print(f'Mean: {np.average(mfVo[inner])}, SD: {np.std(mfVo[inner])}, SE: {np.std(mfVo[inner]) / np.sqrt(sum(inner))}')

    model0 = np.zeros_like(t)
    model0[inner] = 1.0

    fs = 1./np.average(np.diff(t))
    #print(fs)

    #res, pcov = curve_fit(lambda x, mag, sig, dt: response_function(model0, mag, sig, dt), t, mfVo, p0=(np.min(mfVo[inner]), 100/fs, 0.0), bounds=([-np.inf, 1./fs, -np.inf], [np.inf, np.inf, np.inf]))
    fitfunc = lambda x, mag, sig: response_function(model0, mag, sig, 0)
    res, pcov = curve_fit(fitfunc, t, mfVo, p0=(np.min(mfVo[inner]), 100/fs), bounds=([-np.inf, 1./fs], [np.inf, np.inf]))
    perr = np.sqrt(np.diag(pcov))
    print(res, perr)

    print(f'Response time (s): {res[1]/fs} +/- {perr[1]/fs}\nSignal: {res[0]} +/- {perr[0]}')

    return mfVo, fitfunc(t, *res), res, perr

async def main():

    ser = HamiltonSerial(port='COM6', baudrate=38400)
    sp = SyringePumpRamp(ser, '0', SyringeLValve(4, name='syringe_LValve'), 1000, name='Smooth Flow Syringe Pump')
    sp.max_dispense_flow_rate = 4. / 60 * 1000.
    sp.max_aspirate_flow_rate = 4. / 60 * 1000.

    await sp.run_until_idle(sp.initialize())

    k = KeithleyDriver(timeout=0.05)
    thread_task = asyncio.create_task(k.start())
    #await asyncio.sleep(0.1)

    lj = PressureSensorDifferentialDouble(channel_high='AIN0', channel_low='AIN2', sensor_voltage=5.0)
    #lj = PressureSensorwithInAmp(channel='AIN2', gain=201)
    await asyncio.to_thread(lj.open)
    #thread_task2 = asyncio.create_task(lj.start())
    #await asyncio.sleep(0.1) 

    streampot = StreamPot(k, sp, lj)
    #R, dV, V, I = await sp.measure_iv()
    baseline = 30
    flow_rate = 0.01
    volume = flow_rate * 2.5

    if False:
        print(f'Actual speed: {sp._flow_rate(sp._speed_code(flow_rate / 60 * 1000)) / 1000 * 60:0.3f} mL/min')

        volume = flow_rate * 1000 / 10
        expected_length = 2.0 * baseline + volume / (flow_rate / 60 * 1000)
        #volume = 1.0 * 1000
        e_data, p_data = await streampot.exchange(volume, flow_rate / 60 * 1000, baseline)
        plt.figure()
        plt.plot(e_data[0], e_data[1])
        plt.figure()
        mv2pa = 0.2584e-3 * lj.sensor_voltage / 6894.75
        pa = (p_data[1][0]) / mv2pa
        tp = p_data[0]

        baseline_crit = (tp>1) & (tp < 2)
        crit = (tp > 11) & (tp < 12)
        pa_sub = np.average(pa[crit]) - np.average(pa[baseline_crit])
        print(f'Average pressure (Pa): {pa_sub} +/- {np.std(pa[crit])}')

        #np.savetxt('viscosity_data.csv', np.stack((tp, pa), axis=-1), delimiter=',')

        tcrit = tp < expected_length
        y_cond, y_model, _, _ = analyze_streampot(tp[tcrit], pa[tcrit], baseline=baseline, baseline_pad=baseline/2)
        #plt.plot(tp, pa)
        plt.plot(tp[tcrit], y_cond)
        plt.plot(tp[tcrit], y_model)
        plt.show()
    

        return
    else:
        
        expected_length = 2.0 * baseline + volume / (flow_rate / 60)
        logging.info('Expected time (s): %0.3f', expected_length)

        V, t, pressure_data, speed_data, expected_length = await streampot.measure_streaming_potential(min_flow_rate=flow_rate,
                                                    max_flow_rate=flow_rate,
                                                    volume=volume,
                                                    baseline_duration=baseline)
        logging.info('Actual expected time (s): %0.3f', expected_length)

        expected_length += 2.0 * baseline

        Rs = []
        dRs = []
        V0s = []
        dV0s = []
        for i in range(1):
            print(f'Measuring IV curve {i}')
            R, dR, V0, dV0, viv, iiv, _ = await streampot.measure_iv(maxV=0.001)
            Rs.append(R)
            dRs.append(dR)
            V0s.append(V0)
            dV0s.append(dV0)

            plt.plot(viv, iiv, 'o-')

        plt.show()

        print(f'R: Average: {np.average(Rs)}, SD: {np.std(Rs)}')    
        print(f'dR: Average: {np.average(dRs)}, SD: {np.std(dRs)}')    
        print(f'V0: Average: {np.average(V0s)}, SD: {np.std(V0s)}')
        print(f'dV0: Average: {np.average(dV0s)}, SD: {np.std(dV0s)}')

        #print(f'R (ohm): {R}, dV (mV): {dV}')
        #plt.plot(t, V*1e6, '-', alpha=0.3)
        #plt.plot(t, median_filter(V, 21, mode='reflect')*1e6, '-', alpha=0.7)
        V = V[t < expected_length]
        t = t[t < expected_length]
        mf_order = 21
        fig, ax = plt.subplots(1, 2, figsize=(14, 8))
        axleft, axright = ax
        V *= 1e6
        axleft.plot(t, V, alpha=0.3)
        axleft.plot(t, median_filter(V, mf_order, mode='reflect'), alpha=0.7)
        axright.plot(t, analyze_streampot(t, V, baseline=baseline)[0], alpha=0.3)
        print('==== Electrical ====')
        y_cond, y_model, _, _ = analyze_streampot(t, median_filter(V, mf_order, mode='reflect'), baseline=baseline)
        axright.plot(t, y_cond, alpha=0.5)
        axright.plot(t, y_model, alpha=0.5)
        #axright.plot(t, median_filter(analyze_streampot(t, V, baseline=baseline)[0], 21, mode='reflect'), alpha=0.5)
        #np.savetxt('streampot_data_lowsalt.csv', np.stack((t, V), axis=-1), delimiter=',')
        axleft.axvline(baseline, color='k', linestyle='--')
        axleft.axvline(max(t) - baseline, color='k', linestyle='--')

        axright.axvline(baseline, color='k', linestyle='--')
        axright.axvline(max(t) - baseline, color='k', linestyle='--')

        speed_data = np.array(speed_data).T
    #    plt.figure()
    #    plt.plot(speed_data[0] + baseline, speed_data[1])

        tp, mv = pressure_data
        tcrit = tp < expected_length
        mv = mv[tcrit]
        tp = tp[tcrit]
        mv2pa = 0.2584e-3 * lj.sensor_voltage / 6894.75
        pa = mv / mv2pa
        print('==== Pressure ====')
        y_cond, y_model, _, _ = analyze_streampot(tp, pa, baseline=baseline)
        fig, ax = plt.subplots(1, 2, figsize=(14, 8))
        axleft, axright = ax
        axleft.plot(tp, pa, alpha=0.5)
        axright.plot(tp, y_cond, alpha=0.5)
        axright.plot(tp, y_model, alpha=0.5)

        #plt.figure()
        #plt.plot(t, V / np.interp(t, poll_data[0], y_cond))
        plt.show()

async def viscosity():

    ser = HamiltonSerial(port='COM6', baudrate=38400)
    sp = SyringePumpRamp(ser, '3', SyringeLValve(4, name='syringe_LValve'), 250, False, name='syringe_pump')
    sp.max_dispense_flow_rate = 5. / 60 * 1000.
    sp.max_aspirate_flow_rate = 5. / 60 * 1000.

    await sp.run_until_idle(sp.initialize())

    k = KeithleyDriver(timeout=0.05)
    thread_task = asyncio.create_task(k.start())
    #await asyncio.sleep(0.1)

    lj = PressureSensorDifferential(channel='AIN0', sensor_voltage=5.0)
    #lj = PressureSensorwithInAmp(channel='AIN2', gain=201)
    await asyncio.to_thread(lj.open)
    #thread_task2 = asyncio.create_task(lj.start())
    #await asyncio.sleep(0.1)

    streampot = StreamPot(k, sp, lj)
    #R, dV, V, I = await sp.measure_iv()
    baseline = 10
    flow_rate = 0.2 / 1
    print(f'Actual speed: {sp._flow_rate(sp._speed_code(flow_rate / 60 * 1000)) / 1000 * 60:0.3f} mL/min')

    volume = flow_rate * 1000 / 60 * 20
    expected_length = 2.0 * baseline + volume / (flow_rate / 60 * 1000)
    #volume = 1.0 * 1000
    e_data, p_data = await streampot.exchange(volume, flow_rate / 60 * 1000, baseline)
    #plt.figure()
    #plt.plot(e_data[0], e_data[1])
    plt.figure()
    mv2pa = 0.2584e-3 * lj.sensor_voltage / 6894.75
    pa = (p_data[1][0]) / mv2pa
    tp = p_data[0]

    baseline_crit = (tp>1) & (tp < 2)
    crit = (tp > 11) & (tp < 12)
    pa_sub = np.average(pa[crit]) - np.average(pa[baseline_crit])
    print(f'Average pressure (Pa): {pa_sub} +/- {np.std(pa[crit])}')

    #np.savetxt('viscosity_data.csv', np.stack((tp, pa), axis=-1), delimiter=',')

    tcrit = tp < expected_length
    y_cond, y_model, _, _ = analyze_streampot(tp[tcrit], pa[tcrit], baseline=baseline, baseline_pad=baseline/2)
    #plt.plot(tp, pa)
    plt.plot(tp[tcrit], y_cond)
    plt.plot(tp[tcrit], y_model)
    plt.show()
    
async def labjack():

    lj = LabJackDriver()
    lj.open()
    res = []
    init_time = time.time()
    for _ in range(100):
        res.append((time.time() - init_time, ljm.eReadName(lj.instr, "AIN2")))
    
    res = np.array(res).T
    print(np.average(res[1]), np.std(res[1]))
    plt.plot(res[0], res[1])
    plt.show()
    lj.close()

async def labjack_stream():

    lj = PressureSensorDifferential(channel='AIN1', negative_channel='AIN0', dac_channel='DAC0', dac_output=5)
    await asyncio.to_thread(lj.open)
    sample_rate = 500
    stream = asyncio.create_task(asyncio.to_thread(lj.stream, 1, sample_rate))
    await asyncio.sleep(5)
    lj.stop_stream.set()
    actual_sample_rate, data = await stream
    print(actual_sample_rate)
    t = np.arange(int(data.shape[1])) / actual_sample_rate

    plt.plot(t, data[0])
    plt.show()

async def ivcurve():

    k = KeithleyDriver(timeout=0.05)
    thread_task = asyncio.create_task(k.start())
    #await asyncio.sleep(0.1)

    lj = PressureSensorDifferential(channel='AIN0', sensor_voltage=5.0)
    #lj = PressureSensorwithInAmp(channel='AIN2', gain=201)
    await asyncio.to_thread(lj.open)
    #thread_task2 = asyncio.create_task(lj.start())
    #await asyncio.sleep(0.1)

    streampot = StreamPot(k, None, lj)
    #R, dV, V, I = await sp.measure_iv()

    R, dR, V0, dV0, viv, iiv, _ = await streampot.measure_iv(maxV=0.001)
    print(R, dR, V0, dV0)
    #plt.plot(viv, iiv, 'o-')
    #plt.show()

async def exchange_with_iv():
    ser = HamiltonSerial(port='COM6', baudrate=38400)
    sp = SyringePumpRamp(ser, '0', SyringeLValve(4, name='syringe_LValve'), 1000, name='Smooth Flow Syringe Pump')
    sp.max_dispense_flow_rate = 4. / 60 * 1000.
    sp.max_aspirate_flow_rate = 4. / 60 * 1000.

    k = KeithleyDriver(timeout=0.05)
    thread_task = asyncio.create_task(k.start())
    #await asyncio.sleep(0.1)

    lj = PressureSensorDifferentialDouble(channel_high='AIN0', channel_low='AIN2', sensor_voltage=5.0)
    #lj = PressureSensorwithInAmp(channel='AIN2', gain=201)
    await asyncio.to_thread(lj.open)
    #thread_task2 = asyncio.create_task(lj.start())
    #await asyncio.sleep(0.1)

    streampot = StreamPot(k, sp, lj)
    #R, dV, V, I = await sp.measure_iv()

    await sp.run_until_idle(sp.initialize())
    init_time = time.time()
    volume = sp.syringe_volume * 0.1
    flow_rate = 0.2 / 60 * 1000
    expected_time = volume / flow_rate
    print(expected_time)
    task = asyncio.create_task(sp.smart_dispense(volume, flow_rate))

    mv2pa = 0.2584e-3 * lj.sensor_voltage / 6894.75
    while (time.time() - init_time) < expected_time:
        R, dR, V0, dV0, viv, iiv, pdata = await streampot.measure_iv(maxV=0.001)
        try:
            print(time.time() - init_time, R, dR, V0, dV0, np.average(pdata[1][0] / mv2pa))
        except:
            print(time.time() - init_time, R, dR, V0, dV0)

    await task


async def sp_test():

    ser = HamiltonSerial(port='COM6', baudrate=38400)
    sp = SyringePumpRamp(ser, '0', SyringeLValve(4, name='syringe_LValve'), 1000, name='Smooth Flow Syringe Pump')
    mvp = HamiltonValvePositioner(ser, '1', YValve(name='MVP Y Valve'), name='Switching Y Valve')
    sp.max_dispense_flow_rate = 4. / 60 * 1000.
    sp.max_aspirate_flow_rate = 4. / 60 * 1000.

    sp_system = AssemblyBase([sp, mvp], name='Streaming Potential Setup')
    #sp_system.modes = {'Load': Mode({sp: 3, mvp:3}),
    #                   'Measure': Mode({sp: 4, mvp: 0}),
    #                   'Flush': Mode({sp: 4, mvp: 1}),
    #                   'Manual flush': Mode({sp: 0, mvp: 2})}

    sp_system.modes = {'Load': Mode({sp: 3, mvp: 2}),
                       'Measure': Mode({sp: 4, mvp: 1}),
                       'Flush': Mode({sp: 4, mvp: 1}),
                       'Manual flush': Mode({sp: 0, mvp: 3})}


    await sp_system.initialize()
    app = sp_system.create_web_app(template='roadmap.html')
    runner = await run_socket_app(app, 'localhost', 5003)
    #await sp.run_until_idle(sp.smart_dispense(200, 0.5 / 60 * 1000))
    #await sp.move_valve(3)
    #for _ in range(20):
    #    await sp.smart_dispense(sp.syringe_volume, sp.max_aspirate_flow_rate)

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()

    #return
    load = True
    if load:
        #await sp.query('TR')
        #await sp.move_valve(sp.valve.dispense_position)
        #await sp.get_syringe_position()
        #print(sp.syringe_position)
        #await sp.run_until_idle(sp.move_absolute(0, sp.max_dispense_flow_rate))
        await sp.move_valve(3)
        #await sp.move_valve(sp.valve.dispense_position)
        #await sp.run_until_idle(sp.home())
        #await sp.smart_dispense(sp.syringe_volume*1, 0.01 / 60 * 1000)
    else:
        vol = 200
        #await sp.move_valve(sp.valve.dispense_position)
        #await sp.home()
        await sp.move_valve(sp.valve.aspirate_position)
        await sp.run_until_idle(sp.aspirate(vol, sp.max_aspirate_flow_rate))
        await sp.move_valve(sp.valve.dispense_position)
        flow_rates = np.linspace(0.1, 2.0, 21, endpoint=True) / 60 * 1000
        #flow_rates = np.array([1.0, 1.0]) / 60 * 1000
        #await sp.run_until_idle(sp.ramp_speed(vol, flow_rates.tolist()))
        expected_time = vol / sum(flow_rates) * len(flow_rates)
        init_time = time.time()
        await sp.ramp_speed(vol, flow_rates.tolist(), reverse=True)
        #await asyncio.sleep(10)

        # gets current syringe speed
        #response, error = await sp.query('?2')
        await sp.poll_until_idle()
        print(f'Ramp time expected: {expected_time}\n\tand elapsed: {time.time() - init_time}')
        #await sp.ramp_speed(vol, flow_rates.tolist())
        #await sp.run_until_idle(sp.ramp_speed(vol, 0.5 / 60 * 1000))
    
    #res = []
    #init_time = time.time()
    #for _ in range(100):
    #    res.append((time.time() - init_time, ljm.eReadName(lj.instr, "AIN2")))
   # 
   # res = np.array(res).T
    #print(np.average(res[1]), np.std(res[1]))
    #plt.plot(res[0], res[1])
    #plt.show()
    #lj.close()

if __name__=='__main__':
    logging.basicConfig(
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO)
    #mlog = logging.getLogger('matplotlib')
    #mlog.setLevel('WARNING')
    asyncio.run(sp_test(), debug=True)