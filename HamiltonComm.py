from typing import List
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
    Asynchronous serial port interactor for Hamilton devices.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.write_lock: asyncio.Lock = asyncio.Lock()
        
        self.wait_timeout = 0.5
        self.max_retries = 3
        self.sequence_number = 1

    async def get_value(self, rv: dict):
        rv['value'] = await self.reader_async()

    async def _write_message(self, data: HamiltonMessage, response_queue: asyncio.Queue):
        """Basic write/read operation

        Args:
            data (HamiltonMessage): message to send
            response_queue (asyncio.Queue): queue in which to insert response
        """

        return_value = {'value': None}
        print(f'{datetime.datetime.now().isoformat()}: {self.port} => {repr(data)}')
        await self.write_async(data.standard_encode())

        # wait for results to come in, with timeout
        trial = 0
        while trial < self.max_retries:
            try:
                await asyncio.wait_for(self.get_value(return_value), timeout=self.wait_timeout)

                # break out of while if return_value is not None, else try again (finally)
                if return_value['value']:
                    #print(f'Return value: {return_value["value"]}')
                    break
            except asyncio.TimeoutError:
                print('Timed out!')
                pass

            # if not successful try again up to max_retries, changing repeat bit
            trial += 1
            data.repeat = '1'
            #print(f'Trial {trial}... {data}')
            print(f'{datetime.datetime.now().isoformat()}: {self.port} => {repr(data)}')
            await self.write_async(data.standard_encode())
        
        # increment sequence number, cycling between 1 and 7
        self.sequence_number = max(1, (self.sequence_number + 1) % 8) 

        # put return value in response queue
        await response_queue.put(return_value['value'])

    async def query(self, address: str, cmd: str, response_queue: asyncio.Queue) -> None:
        """User interface for single async read/write query operation

        Args:
            address (str): device address (single character)
            cmd (str): command string
            response_queue (asyncio.Queue): queue in which to put response
        """

        # write out message
        await self.batch_query([address], [cmd], [response_queue])

    async def batch_query(self, addresses: List[str], cmds: List[str], response_queues: List[asyncio.Queue]) -> None:
        """Batch processing of multiple query operations

        Args:
            addresses (List[str]): List of address characters, one for each command
            cmds (List[str]): List of command strings
            response_queues (List[asyncio.Queue]): List of response queues corresponding to each
                command.
        """

        # lock writing until all batch messages are written
        async with self.write_lock:

            # send each message
            for address, cmd, response_queue in zip(addresses, cmds, response_queues):
                data = HamiltonMessage(address, cmd, self.sequence_number)
                await self._write_message(data, response_queue)

    async def reader_async(self) -> str | None:
        """
        Async reader.
        """
        while self.is_open:
            # read data
            data: bytes = await self.read_until_async(chr(3).encode())
            #printcodes(data.decode('latin-1'))

            # throw away first byte (always ASCII 255)
            data = data[1:].decode()
            print(f'{datetime.datetime.now().isoformat()}: {self.port} <= {data}')

            # calculate checksum
            data_chksum = ord(checksum(data))

            # read checksum byte
            chksum: bytes = await self.read_async(1)
            recv_chksum = ord(chksum.decode())

            # compare checksums; if they match, put in response queue
            if recv_chksum == data_chksum:
                return data
            else:
                print(f'Received checksum {recv_chksum}, calculated checksum {data_chksum}, for data {data}')
                return None
