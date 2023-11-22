import asyncio
import logging
from typing import List
from nidaqmx import Task
from nidaqmx.constants import Signal
from nidaqmx._task_modules.channels import Channel

from HamiltonDevice import HamiltonSyringePumpInterrupt

class BubbleController(Task):

    def connect(self, channel_name: str, device: HamiltonSyringePumpInterrupt) -> Channel:
        # NOTE: Use falling edge only
        channel = self.di_channels.add_di_chan(channel_name)
        channel.di_dig_fltr_min_pulse_width = 2e-4
        channel.di_dig_fltr_enable = True

        def stop_device(task_handle: Task, signal_type: Signal, callback_data) -> None:
            logging.debug('Interrupting device %s', device.name)
            device.interrupt.set()
            return 0
        
        self.register_signal_event(Signal.CHANGE_DETECTION_EVENT, stop_device)

    def simconnect(self, channel_name: str) -> Channel:
        # NOTE: Use falling edge only
        channel = self.di_channels.add_di_chan(channel_name)
        channel.di_dig_fltr_min_pulse_width = 2e-4
        channel.di_dig_fltr_enable = True

        def stop_device(task_handle: Task, signal_type: Signal, callback_data) -> None:
            logging.debug('Interrupting...')
            #device.interrupt.set()
            return 0
        
        self.timing.cfg_change_detection_timing(channel.name, '')
        self.register_signal_event(Signal.CHANGE_DETECTION_EVENT, stop_device)

async def main():

    # NOTE: This is broken. Can only have one task at a time.

    bc = BubbleController()
    bc.simconnect('Dev1/port3/line0')
    bc.start()
    bc2 = BubbleController()
    bc2.simconnect('Dev1/port0/line1')
    bc2.start()
    await asyncio.sleep(30)
    bc.stop()
    bc2.stop()


async def mainold() -> None:

    with Task('bubble detector') as task:
        ch30 = task.di_channels.add_di_chan('Dev1/port3/line0')
        
        res: List[float] = task.read(1)
        logging.debug(res)
        task.timing.cfg_change_detection_timing('Dev1/port3/line0', 'Dev1/port3/line0')
        ch30.di_dig_fltr_min_pulse_width = 2e-4
        ch30.di_dig_fltr_enable = True
        
        def print_cde(task_handle: Task, signal_type: Signal, callback_data) -> None:
            logging.debug(f'Change detected!!!, task = {task_handle}, signal_type = {signal_type}, data = {callback_data}')
            logging.debug(task.read(1))
            return 0
        
        task.register_signal_event(Signal.CHANGE_DETECTION_EVENT, print_cde)
        task.start()
    
        await asyncio.sleep(100)

if __name__=='__main__':

    logging.basicConfig(
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.DEBUG)

    asyncio.run(main(), debug=True)