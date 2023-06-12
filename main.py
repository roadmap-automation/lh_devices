import time
import asyncio
import aioserial
import aioconsole
from hamilton import HamiltonSerial, HamiltonBase, printcodes
from hamiltonvalve import HamiltonSerial, HamiltonValvePositioner, LoopFlowValve

class AsyncKeyboard:

    def __init__(self, serial_instance: HamiltonSerial, test_device: HamiltonBase) -> None:
        self.serial = serial_instance
        self.console_queue: asyncio.Queue = asyncio.Queue()
        self.dev = test_device
        
    async def initialize(self) -> None:

        await asyncio.gather(self.get_input(), self.send_mvp_command())

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
            await self.dev.move(int(cmd))
            #if response is not None:
            #    print(response)

async def main():
    
    ser = HamiltonSerial(port='COM5', baudrate=9600)
    mvp = HamiltonValvePositioner(ser, '0', LoopFlowValve(2))
    #mvp = HamiltonValvePositioner(ser, '0', DistributionValve(8))
    #mvp = HamiltonBase(ser, '0')
    #await mvp.initialize()
    ak = AsyncKeyboard(ser, mvp)

    await asyncio.gather(ser.initialize(), mvp.initialize(), ak.initialize())

if __name__ == '__main__':
    asyncio.run(main(), debug=True)

