import asyncio
import aioserial
import aioconsole
from hamilton import HamiltonSerial, HamiltonBase

class AsyncKeyboard:

    def __init__(self, serial_instance: HamiltonSerial, test_device: HamiltonBase) -> None:
        self.serial = serial_instance
        self.console_queue: asyncio.Queue = asyncio.Queue()
        self.dev = test_device
        self.async_tasks = [self.get_input(), self.send_command()]

    async def get_input(self) -> None:
        while True:
            inp = await aioconsole.ainput()
            await self.console_queue.put(inp.strip())

    async def send_command(self) -> None:
        while True:
            cmd = await self.console_queue.get()
            #response = cmd
            response = await self.dev.send_until_idle(cmd)
            if response is not None:
                print(response)

async def main():
    
    ser = HamiltonSerial(port='COM5', baudrate=9600)
    mvp = HamiltonBase(ser, 1)
    ak = AsyncKeyboard(ser, mvp)

    tasks = ser.async_tasks + ak.async_tasks

    await asyncio.gather(*tasks)

if __name__ == '__main__':
    asyncio.run(main(), debug=True)

