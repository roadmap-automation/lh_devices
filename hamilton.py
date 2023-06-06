import asyncio
import aioserial
from dataclasses import dataclass
from typing import List, Union

def printcodes(p: str) -> None:
    print(p, [hex(ord(b)) for b in p])

def checksum(msg: str) -> str:
    """
    Calculates checksum of message as the character representing the bitwise xor operation over
    all bytes in the message.
    """
    msg_binary = ''.join(format(ord(b), '08b') for b in msg)
    chksum_binary = ''.join(str(sum(int(bt) for bt in msg_binary[i::8]) % 2) for i in range(8))

    return chr(int(chksum_binary, base=2))

@dataclass
class HamiltonMessage:
    """
    Message for Hamilton communication protocols
    """

    address: str
    cmd: str
    sequence_number: int
    repeat: str = '0'

    def standard_encode(self):
        """
        Encodes message into Hamilton message format for use with standard communication protocol
        """

        sequence_byte = chr(int('0011' + self.repeat + format(self.sequence_number, '03b'), base=2))
        data = chr(2) + self.address + sequence_byte + self.cmd + chr(3)
        data += checksum(data)
        #printcodes(data)
        return data.encode()
    
    def terminal_encode(self):
        """
        Encodes message into Hamilton message format for use with terminal communication protocol
        """
        
        return f'/{self.address}{self.cmd}\r'.encode()

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
        self.max_retries = 3
        self.sequence_number = 1

    async def query_timer(self):
        """
        Waits for write_started event to be triggered, then waits a delay and allows I/O again.
        """
        while True:
            async with self.ioblocked:
                await self.ioblocked.wait()
                await asyncio.sleep(self.delay)
                self.ioblocked.notify()
    
    async def query(self, address: str, cmd: str) -> str:
        """User interface for async read/write query operations"""

        return_value = {'value': None}
        async def get_value(rv):
            rv['value'] = await self.read_queue.get()

        trial = 0
        data = HamiltonMessage(address, cmd, self.sequence_number)
        await self.write_queue.put(data)

        while trial < self.max_retries:
            try:
                await asyncio.wait_for(get_value(return_value), timeout=self.delay*3)
                trial = self.max_retries
            except asyncio.TimeoutError:
                trial += 1
                data.repeat = '1'
                print(f'Trial {trial}... {data}')
                await self.write_queue.put(data)
        
        # increment sequence number, cycling between 1 and 7
        self.sequence_number = max(1, (self.sequence_number + 1) % 8) 
        
        return return_value['value']
    
    async def reader_async(self) -> None:
        """
        Async reader. Should be started as part of asyncio loop
        """
        while self.is_open:
            data: bytes = await self.read_until_async(chr(3).encode())
            data = data[1:].decode()
            data_chksum = ord(checksum(data))
            chksum: bytes = await self.read_async(1)
            recv_chksum = ord(chksum.decode())
            if recv_chksum == data_chksum:
                await self.read_queue.put(data)
            else:
                print(f'Received checksum {recv_chksum}, calculated checksum {data_chksum}, for data {data}')

    async def writer_async(self) -> None:
        """
        Async writer. Should be started as part of asyncio loop. Monitors the write_queue.
        Sends data to serial connection queues. Uses ioblocked Condition to
        ensure write operations are separated by a minimum delay time
        """

        while self.is_open:
            wdata: HamiltonMessage = await self.write_queue.get()

            async with self.ioblocked:

                # write data
                await self.write_async(wdata.standard_encode())

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
        
        return response[2:-1]

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

