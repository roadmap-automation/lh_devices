import asyncio
import aioserial
import datetime
from dataclasses import dataclass
from typing import List, Union, Tuple

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
        #print(data)
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
        self.wait_timeout = 1.0
        self.max_retries = 3
        self.sequence_number = 1

    async def query_timer(self):
        """
        Waits for write_started event to be triggered, then waits a delay and allows I/O again.
        """
        while True:
            async with self.ioblocked:
                #print(datetime.datetime.now().isoformat() + ': Waiting on ioblocked')
                await self.ioblocked.wait()
                #print(datetime.datetime.now().isoformat() + ': Sleeping on ioblocked')
                await asyncio.sleep(self.delay)
                #print(datetime.datetime.now().isoformat() + ': query_timer notifying on ioblocked')
                self.ioblocked.notify()
    
    async def query(self, address: str, cmd: str) -> str:
        """User interface for async read/write query operations"""

        # set up return value container
        return_value = {'value': None}
        async def get_value(rv):
            rv['value'] = await self.read_queue.get()

        # write out message
        trial = 0
        data = HamiltonMessage(address, cmd, self.sequence_number)
        #print(datetime.datetime.now().isoformat() + ': Writing to write queue')
        await self.write_queue.put(data)

        # wait for results to come in, with timeout
        while trial < self.max_retries:
            try:
                await asyncio.wait_for(get_value(return_value), timeout=self.wait_timeout)
                break
            except asyncio.TimeoutError:
                # if not successful try again up to max_retries, changing repeat bit
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
            # read data
            data: bytes = await self.read_until_async(chr(3).encode())
            #printcodes(data.decode('latin-1'))

            # throw away first byte (always ASCII 255)
            data = data[1:].decode()

            # calculate checksum
            data_chksum = ord(checksum(data))

            # read checksum byte
            chksum: bytes = await self.read_async(1)
            recv_chksum = ord(chksum.decode())

            # compare checksums; if they match, put in response queue
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
            #print(datetime.datetime.now().isoformat() + ': writer_async waiting for write data')
            wdata: HamiltonMessage = await self.write_queue.get()

            async with self.ioblocked:

                # write data
                #print(wdata)
                #printcodes(wdata.standard_encode().decode())
                #print(datetime.datetime.now().isoformat() + ': writer_async writing')
                await self.write_async(wdata.standard_encode())

                # start query timer
                #print(datetime.datetime.now().isoformat() + ': writer_async notify ioblocked')
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
        
        if response:
            return response[2:-1]
        else:
            return None

    async def run_until_idle(self, cmd: str) -> Tuple[str, str]:
        """
        Sends from serial connection and waits until idle
        """

        self.idle = False
        response = await self.query(cmd + 'R')
        if response is not None:

            status_byte = response[0]
            if len(response) > 1:
                response = response[1:]

            error = self.parse_status_byte(status_byte)
            while (not self.idle) & (not error):
                error = await self.update_status()

            return response, error
        else:
            return None, None

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

