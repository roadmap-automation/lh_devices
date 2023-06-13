import time
import asyncio
import aioserial
import aioconsole
from HamiltonComm import HamiltonSerial
from HamiltonDevice import HamiltonBase, HamiltonValvePositioner, HamiltonSyringePump
from valve import LoopFlowValve, SyringeYValve

class AsyncKeyboard:

    def __init__(self, serial_instance: HamiltonSerial, test_device: HamiltonBase) -> None:
        self.serial = serial_instance
        self.console_queue: asyncio.Queue = asyncio.Queue()
        self.dev = test_device
        
    async def initialize(self) -> None:

        await asyncio.gather(self.get_input(), self.send_sp_command())

    async def get_input(self) -> None:
        while True:
            inp = await aioconsole.ainput()
            await self.console_queue.put(inp.strip())

    async def send_command(self) -> None:
        while True:
            cmd = await self.console_queue.get()
            #response = cmd
            response = await self.dev.run_until_idle(cmd)
            if response is not None:
                print(response)

    async def send_mvp_command(self) -> None:
        while True:
            cmd = await self.console_queue.get()
            await self.dev.move_valve(int(cmd))
            #if response is not None:
            #    print(response)

    async def send_sp_command(self) -> None:
        while True:
            cmd = await self.console_queue.get()
            if cmd.startswith('?'):
                response = await self.dev.query(cmd)
            elif cmd.startswith('d'):
                volume, flow_rate = cmd[1:].split('f')
                response = await self.dev.dispense(float(volume), float(flow_rate))
            elif cmd.startswith('a'):
                volume, flow_rate = cmd[1:].split('f')
                response = await self.dev.aspirate(float(volume), float(flow_rate))
            elif cmd.startswith('r'):
                resolution = int(cmd[1:])
                response = await self.dev.set_high_resolution(bool(resolution))
                print(self.dev._high_resolution)
            else:
                response = await self.dev.run_until_idle(cmd)
            if response is not None:
                print(response)

            #if response is not None:
            #    print(response)

async def main():
    
    ser = HamiltonSerial(port='COM5', baudrate=9600)
    #mvp = HamiltonValvePositioner(ser, '0', LoopFlowValve(2))
    sp = HamiltonSyringePump(ser, '0', SyringeYValve(), 5000, False)
    #mvp = HamiltonValvePositioner(ser, '0', DistributionValve(8))
    #mvp = HamiltonBase(ser, '0')
    #await mvp.initialize()
    ak = AsyncKeyboard(ser, sp)

    await asyncio.gather(ser.initialize(), sp.initialize(), ak.initialize())

if __name__ == '__main__':
    asyncio.run(main(), debug=True)

