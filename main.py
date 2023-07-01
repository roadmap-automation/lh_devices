import asyncio
import aioconsole

from HamiltonComm import HamiltonSerial
from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump
from valve import LoopFlowValve, SyringeYValve
from assemblies import AssemblyTest

class AsyncKeyboard:

    def __init__(self, serial_instance: HamiltonSerial, test_device: HamiltonBase, stop_event: asyncio.Event) -> None:
        self.serial = serial_instance
        self.console_queue: asyncio.Queue = asyncio.Queue()
        self.dev = test_device
        self.stopped = False
        self.stop_event = stop_event
    
    def stop(self) -> None:
        self.stop_event.set()
        print('stopping keyboard input...')
        self.stopped = True
        
    async def initialize(self) -> None:

        await asyncio.gather(self.get_input(), self.send_sp_command())

    async def get_input(self) -> None:
        while not self.stopped:
            inp = await aioconsole.ainput()
            if inp.strip() == 'stop':
                self.stop()
            else:
                await self.console_queue.put(inp.strip())

    async def send_command(self) -> None:
        while not self.stopped:
            cmd = await self.console_queue.get()
            #response = cmd
            response = await self.dev.run_until_idle(cmd)
            if response is not None:
                print(response)

    async def send_mvp_command(self) -> None:
        while not self.stopped:
            cmd = await self.console_queue.get()
            await self.dev.move_valve(int(cmd))
            #if response is not None:
            #    print(response)

    async def send_sp_command(self) -> None:
        while not self.stopped:
            cmd = await self.console_queue.get()
            if cmd.startswith('?'):
                response = await self.dev.run(self.dev.query(cmd))
            elif cmd.startswith('d'):
                volume, flow_rate = cmd[1:].split('f')
                response = await self.dev.run_until_idle(self.dev.dispense(float(volume), float(flow_rate)))
            elif cmd.startswith('a'):
                volume, flow_rate = cmd[1:].split('f')
                response = await self.dev.run_until_idle(self.dev.aspirate(float(volume), float(flow_rate)))
            elif cmd.startswith('r'):
                resolution = int(cmd[1:])
                response = await self.dev.run_until_idle(self.dev.set_high_resolution(bool(resolution)))
                print(self.dev._high_resolution)
            else:
                response = await self.dev.run_until_idle(self.dev.query(cmd))
            if response is not None:
                print(response)

            #if response is not None:
            #    print(response)

class Launcher:
    """Launches async devices and monitors event for closing"""

    def __init__(self, tasks: asyncio.Future, stop_event: asyncio.Event) -> None:
        self.tasks = [asyncio.create_task(task) for task in tasks]
        self.stop_event = stop_event
        
    async def run(self):

        try:
            await asyncio.gather(*(self.tasks + [self.on_stop()]))
        except asyncio.CancelledError:
            print('Exiting launcher...')

    async def on_stop(self):
        """Stops stuff"""
        await self.stop_event.wait()
        for task in self.tasks:
            task.cancel()

        print('Cleaning up...')
        await asyncio.sleep(2)

async def main():
    
    ser = HamiltonSerial(port='COM5', baudrate=38400)
    mvp = HamiltonValvePositioner(ser, '1', LoopFlowValve(8))
    sp = HamiltonSyringePump(ser, '0', SyringeYValve(), 5000, False)
    #mvp = HamiltonValvePositioner(ser, '0', DistributionValve(8))
    #mvp = HamiltonBase(ser, '0')
    #await mvp.initialize()
    stop_event = asyncio.Event()
    ak = AsyncKeyboard(ser, sp, stop_event)
    at = AssemblyTest(mvp, sp)
    #launch = Launcher([ser.initialize(), mvp.initialize(), sp.initialize(), ak.initialize()], stop_event)
    launch = Launcher([ser.initialize(), at.initialize(), ak.initialize()], stop_event)
    await launch.run()

    #await asyncio.gather(ser.initialize(), sp.initialize(), ak.initialize())

if __name__ == '__main__':
    asyncio.run(main(), debug=True)

