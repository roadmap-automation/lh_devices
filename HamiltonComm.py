import asyncio
import aioserial
from dataclasses import dataclass
import datetime

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
        
        # delay time between I/O operations (Hamilton default = 100 ms)
        self.delay = 0.1
        self.wait_timeout = 0.5
        self.max_retries = 3
        self.sequence_number = 1

    async def initialize(self) -> None:

        try:
            await asyncio.gather(self.reader_async(), self.writer_async())
        except asyncio.CancelledError:
            print('Closing serial connection...')
            self.close()
            
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
                print(datetime.datetime.now().isoformat() + ': <= ' + data)
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

            #async with self.ioblocked:

            # write data
            print(datetime.datetime.now().isoformat() + ': => ' + repr(wdata))
            #printcodes(wdata.standard_encode().decode())
            #print(datetime.datetime.now().isoformat() + ': writer_async writing')
            await self.write_async(wdata.standard_encode())
            await asyncio.sleep(self.delay)
            # start query timer
            #print(datetime.datetime.now().isoformat() + ': writer_async notify ioblocked')
    #            self.ioblocked.notify()
