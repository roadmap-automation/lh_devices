from typing import Tuple, List
import pyvisa
from threading import Event
from queue import Queue, Empty, Full
import time
import asyncio
import logging
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt

from HamiltonComm import HamiltonSerial
from HamiltonDevice import HamiltonSyringePump, PollTimer
from valve import SyringeLValve, SyringeValveBase

class KeithleyDriver:

    def __init__(self, timeout = 0.05, name='KeithleyDriver') -> None:

        self.name = name
        self.rm = None
        self.instr = None
        self.timeout = timeout
        self.inqueue: Queue = Queue()
        self.outqueue: Queue = Queue(1)
        self._stop_event = Event()
        self.stopped = False
        self.lock: asyncio.Lock = asyncio.Lock()

    def stop(self):
        self.stopped = True
    
    def clear(self):
        self.stopped = False

    def open(self, model='2450'):
        
        rm = pyvisa.ResourceManager()
        res_id = next(res for res in rm.list_resources() if model in res)
        print(res_id)

        instr = rm.open_resource(res_id)

        self.rm = rm
        self.instr = instr
        print(instr.timeout)

    def close(self):

        self.instr.close()
        self.rm.close()

    def run(self):
        """Synchronous code to interact with the Keithley"""

        # open instrument resource
        self.open()
        self.clear()
        while not self.stopped:

            cmd = None

            # get queue
            while (not self.stopped) & (cmd is None):
                time.sleep(self.timeout)
                #print('Waiting on queue...')
                try:
                    cmd = self.inqueue.get_nowait()
                except Empty:
                    pass
            
            if cmd is not None:

                logging.info('%s => %s', self.name, cmd)

                # write value to instrument
                self.instr.write(cmd)
                
                # if command is a query command
                if '?' in cmd:
                    response: str = self.instr.read()

                    logging.info('%s <= %s', self.name, response[:-1])

                    # write response to outqueue (blocks until value is read)
                    success = False
                    while (not self.stopped) & (not success):
                        try:
                            self.outqueue.put_nowait(response)
                            success = True
                        except Full:
                            pass

                        time.sleep(self.timeout)
            
        self.close()

    async def monitor_queue(self) -> str:

        res = None
        success = False
        while (not self.stopped) & (not success):
            await asyncio.sleep(self.timeout)
            try:
                res = self.outqueue.get_nowait()
                success = True
            except Empty:
                pass
        
        return res

    def write(self, cmd: str) -> None:
        """Writes command to queue"""
        #cmd.replace(' ', '\\s')
        self.inqueue.put(cmd)

    async def query(self, cmd: str) -> str:
        """Helper function for performing queries"""
        self.write(cmd)
        res = await self.monitor_queue()

        return res

class StreamPot:

    def __init__(self, driver: KeithleyDriver = KeithleyDriver()):
        self.driver = driver

        # TODO: Pressure sensor (create Labjack driver)
        # TODO: Syringe pump loop system

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
        setup_commands = [':SOUR:FUNC VOLT',
                        f':SOUR:SWE:VOLT:LIN -{maxV}, {maxV}, {npts}, {time_per_point}',
                        ':INIT',
                        ]

        for cmd in setup_commands:
            self.driver.write(cmd)

        # monitor output until correct number of points have been collected
        pointcount = 0
        while pointcount < npts:
            await asyncio.sleep(time_per_point)
            pointcount = int(await self.driver.query(':TRAC:ACT? "defbuffer1"'))

        # read all data        
        data = await self.driver.query(f':TRAC:DATA? 1, {npts}, "defbuffer1", SOUR, READ')
    
        # format data into numpy arrays
        data = np.fromstring(data, sep=',')
        V, I = data.reshape((2, len(data) // 2), order='F')

        # Calculate resistance and voltage offset (in mV)
        R, b = np.polyfit(I, V, 1)
        dV = b * 1e3

        return R, dV, V, I

    async def measure_streaming_potential(self,
                        current: float = 0,
                        time: float = 30,
                        time_per_point: float = 0.1) -> Tuple[float,
                                                               float,
                                                               np.ndarray,
                                                               np.ndarray]:

        # run setup commands
        setup_commands = [':SOUR:FUNC CURR',
                          f':SOUR:CURR {current}',
                          ':SENS:FUNC "VOLT"'
                          ':VOLT:RSEN 0',
                          ':VOLT:RANG:AUTO ON',
                          f':TRIG:LOAD "DurationLoop", {time}, {time_per_point}',
                          ':INIT',
                        ]

        for cmd in setup_commands:
            self.driver.write(cmd)

        await asyncio.sleep(time + 10 * time_per_point)

        # Get number of available points
        pointcount = int(await self.driver.query(':TRAC:ACT? "defbuffer1"'))

        # read all data        
        data = await self.driver.query(f':TRAC:DATA? 1, {pointcount}, "defbuffer1", READ, REL')
    
        # format data into numpy arrays
        data = np.fromstring(data, sep=',')
        V, t = data.reshape((2, pointcount), order='F')

        return V, t

class SyringePumpRamp(HamiltonSyringePump):

    def __init__(self, serial_instance: HamiltonSerial, address: str, valve: SyringeValveBase, syringe_volume: float = 5000, high_resolution=False, name=None) -> None:
        super().__init__(serial_instance, address, valve, syringe_volume, high_resolution, name)

    async def ramp_speed(self, volume: float, flow_rates: List[float], reverse=True) -> None:
        """Ramp speed using program of flow rates such that same time is spent
            on each flow rate

        Args:
            volume (float): total volume in uL
            flow_rates: List[float]: flow rates in uL / s
        """

        if reverse:
            #volume /= 2.0
            flow_rates = flow_rates + flow_rates[::-1]

        stroke_length = self._stroke_length(volume)
        step_time = volume / sum(flow_rates)

        # set up timer
        timer = PollTimer(step_time, self.address_code)

        # start move and timer simultaneously
        await asyncio.gather(self.query(f'V{self._speed_code(flow_rates[0])}D{stroke_length}R'), timer.cycle())
        #if error:
        #    logging.error(f'{self}: Syringe move error {error}')

        for fr in flow_rates[1:]:
            # wait until poll_delay timer has ended before setting new flow rate
            await timer.wait_until_set()

            # set new speed and start the poll_delay timer
            await asyncio.gather(self.query(f'V{self._speed_code(fr)}'), timer.cycle())
#
#            response, error = await self.query(f'V{self._speed_code(fr)}')
            #if error:
            #    logging.error(f'{self}: Syringe async velocity set error {error}')

async def main():

    k = KeithleyDriver(timeout=0.05)
    thread_task = asyncio.create_task(asyncio.to_thread(k.run))
    await asyncio.sleep(0.1)
    sp = StreamPot(k)
    #R, dV, V, I = await sp.measure_iv()
    V, t = await sp.measure_streaming_potential(time=5)

    k.stop()
    await thread_task

    #print(f'R (ohm): {R}, dV (mV): {dV}')
    plt.plot(t, V, 'o-', alpha=0.7)
    plt.show()

async def sp_test():

    ser = HamiltonSerial(port='COM6', baudrate=38400)
    sp = SyringePumpRamp(ser, '3', SyringeLValve(4, name='syringe_LValve'), 5000, False, name='syringe_pump')
    sp.max_dispense_flow_rate = 10. / 60 * 1000.
    sp.max_aspirate_flow_rate = 10. / 60 * 1000.

    await sp.run_until_idle(sp.initialize())
    #await sp.move_valve(3)
    #for _ in range(3):
    #    await sp.smart_dispense(5000, sp.max_aspirate_flow_rate)
    load = False
    if load:
        await sp.move_valve(3)
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


if __name__=='__main__':
    logging.basicConfig(
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO)
    asyncio.run(sp_test(), debug=True)