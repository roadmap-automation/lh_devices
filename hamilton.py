import asyncio
import aioserial
from typing import List, Union

class HamiltonSerial(aioserial.AioSerial):
    """
    Asynchronous serial port interactor for Hamilton devices. Implements an async read/write sequence using Queues,
    with a delay (Hamilton-recommended 100 ms) between IO operations on different devices
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.read_queue: asyncio.Queue = asyncio.Queue()
        self.write_queue: asyncio.Queue = asyncio.Queue()
        self.ioblocked: asyncio.Condition = asyncio.Condition()
        self.async_tasks: List[asyncio.Task] = [self.reader_async(),
                                                self.writer_async(),
                                                self.query_timer(),
                                                ]
        
        # delay time between I/O operations (Hamilton default = 100 ms)
        self.delay = 0.1

    async def query_timer(self):
        """
        Waits for write_started event to be triggered, then waits a delay and allows I/O again.
        """
        while True:
            async with self.ioblocked:
                await self.ioblocked.wait()
                await asyncio.sleep(self.delay)
                self.ioblocked.notify()
    
    async def query(self, address:int, cmd: str) -> str:
        """User interface for async read/write query operations"""

        data = {'address': address, 'cmd': cmd}
        await self.write_queue.put(data)
        return await self.read_queue.get()
    
    async def reader_async(self) -> None:
        """
        Async reader. Should be started as part of asyncio loop
        """
        while self.is_open:
            data: bytes = await self.read_until_async((chr(3) + chr(13) + chr(10)).encode())
            await self.read_queue.put(data.decode())

    async def writer_async(self) -> None:
        """
        Async writer. Should be started as part of asyncio loop. Monitors the write_queue.
        Sends data to serial connection queues. Uses ioblocked Condition to
        ensure write operations are separated by a minimum delay time
        """

        while self.is_open:
            wdata = await self.write_queue.get()

            # format data using address
            data = f"/{wdata['address']}{wdata['cmd']}\r".encode()

            async with self.ioblocked:

                # write data
                await self.write_async(data)

                # start query timer
                self.ioblocked.notify()

class HamiltonBase:
    """Base class for Hamilton multi-valve positioner (MVP) and syringe pump (PSD) devices.

        Requires:
        serial_instance -- HamiltonSerial instance for communication
        address -- single character string from '0' to 'F' corresponding to the physical
                    address switch position on the device. Automatically converted to the 
                    correct address code.
     
       """

    def __init__(self, serial_instance: HamiltonSerial, address: str) -> None:
        
        self.serial = serial_instance
        self.idle = True
        self.busy_code = '@'
        self.idle_code = '`'
        self.address = address
        self.address_code = chr(int(address, base=16) + int('31', base=16))

    async def query(self, cmd: str) -> str:
        """
        Wraps self.serial.query with a trimmed response
        """

        response = await self.serial.query(self.address_code, cmd)
        
        return response[2:-3]

    async def send_until_idle(self, cmd: str) -> str:
        """
        Sends from serial connection and waits until idle
        """

        self.idle = False
        response = await self.query(cmd)
        error = self.parse_status_byte(response)
        while (not self.idle) & (not error):
            error = await self.update_status()

        return error

    async def update_status(self) -> Union[str, None]:
        """
        Polls the status of the device using 'Q'
        """

        response = await self.query('Q')
        error = self.parse_status_byte(response)

        return error

    def parse_status_byte(self, c: str) -> Union[str, None]:
        """
        Parses status byte
        """

        error = None
        match c:
            case self.busy_code:
                self.idle = False
            case self.idle_code:
                self.idle = True
            case 'b':
                error = 'Bad command'
            case _ :
                error = f'Error code: {c}'

        if error:
            self.idle = True

        return error

