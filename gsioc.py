from typing import Dict
from enum import Enum
from dataclasses import dataclass
import datetime
import asyncio
import aioserial

""" Configuration using com0com and hub4com:

Note: Gilson 506C is connected to physical port COM1.

com0com 2.2.2.0:
>install 1 EmuBR=yes,EmuOverrun=yes PortName=COM12,EmuBR=yes,EmuOverrun=yes
>install 2 EmuBR=yes,EmuOverrun=yes PortName=COM13,EmuBR=yes,EmuOverrun=yes
>list
       CNCA2 PortName=-,EmuBR=yes,EmuOverrun=yes
       CNCB2 PortName=COM13,EmuBR=yes,EmuOverrun=yes
       CNCA1 PortName=-,EmuBR=yes,EmuOverrun=yes
       CNCB1 PortName=COM12,EmuBR=yes,EmuOverrun=yes

hub4com 2.1.0.0:
command prompt> hub4com --baud=19200 --parity=e --octs=off --route=1,2:0 --route=0:1,2 \\.\CNCA1 \\.\CNCA2 \\.\COM1

This routes all traffic from CNCA1 (<->COM12) back and forth to CNCA2 (<-> virtual COM13) and COM1 (physical port)

"""

interrupt = False
class GSIOCCommandType(Enum):
    BUFFERED = 'buffered'
    IMMEDIATE = 'immediate'

@dataclass
class GSIOCMessage:
    messagetype: str
    data: str

class GSIOC(aioserial.AioSerial):
    """
    Virtual GSIOC object.
    """

    def __init__(self, gsioc_address: int, port: str='COM13', baudrate: int=19200, parity=aioserial.PARITY_EVEN, gsioc_name: str='Virtual GSIOC'):
        """
        Connects to a virtual COM port.

        Inputs:
        gsioc_address -- address of GSIOC virtual device
        port -- virtual COM port identifier
        baudrate -- baud rate of virtual COM port. GSIOC must be 19200, 9600, or 4800
        parity -- parity of virtual COM port. Must be even for GSIOC specification.

        8 bits, 1 stop bit are defaults and part of the GSIOC spec.
        """
        super().__init__(port=port, baudrate=baudrate, parity=parity)

        self.address = gsioc_address    # between 1 and 63
        self.interrupt: bool = False
        self.connected: bool = False
        self.gsioc_name = gsioc_name
        self.message_queue: asyncio.Queue = asyncio.Queue(1)
        self.response_queue: asyncio.Queue = asyncio.Queue(1)

        #signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal, frame):
        """
        Handles stop signals for a clean exit
        """
        self.interrupt = True

    async def listen(self) -> None:
        """
        Starts GSIOC listener. Only ends when an interrupt signal is received.
        """

        print('Starting GSIOC listener... Ctrl+C to exit.')
        self.interrupt = False

        # infinite loop to wait for a command
        while not self.interrupt:

            #print('Waiting for connection...')

            await self.wait_for_connection()

            if self.connected: # address received
                #print('Connection established, waiting for command')
                # waits for a command
                cmd = await self.wait_for_command()

                if cmd:

                    print(f'{datetime.datetime.now().isoformat()}: {self.port} (GSIOC) <= {cmd}')

                    # process ID request immediately
                    if cmd == '%':
                        await self.send(self.gsioc_name)
                    
                    else:
                        # parses received command into a GSIOCMessage
                        data = await self.parse_command(cmd)

                        # put message in listener queue
                        await self.message_queue.put(data)

                        if data.messagetype == GSIOCCommandType.IMMEDIATE:
                            #print('Waiting for response to immediate command')
                            response: str = await self.response_queue.get()
                            await self.send(response)

            #print('Connection reset...')

        # close serial port before exiting when interrupt is received
        self.close()

    async def wait_for_connection(self):
        """
        Waits for an address byte (connection request). Also handles byte 255, which
        disconnects all GSIOC listeners, by sending an active break signal.
        """

        comm = await self.read1()

        # if address matches, echo the address
        if comm == self.address + 128:
            #print(f'address matches, writing {chr(self.address + 128).encode()}')
            await self.write1(chr(self.address + 128))
            self.connected = True
        
        # if result is byte 255, send a break character
        # TODO: check if this should be self.ser.send_break
        elif comm == 255:
            #print('sending break')
            await self.write1(chr(10))
            self.connected = False

    async def wait_for_command(self):
        """
        Waits for a command from the GSIOC master. 
        """

        # collect single bytes. Timeout is set for responsiveness to keyboard interrupts.
        while self.connected & (not self.interrupt):
            comm = await self.read1()

            # buffered command...read until line feed is sent
            if comm == 10:
                #print(f'got LF, starting buffered command read')
                await self.write1(chr(comm))
                msg = ''
                while comm != 13:
                    # get a character
                    comm = await self.read1()

                    # acknowledge character
                    await self.write1(chr(comm))

                    msg += chr(comm)

                #print(f'got CR, end of message: {msg}')

                return msg
            
            # immediate command...read a single character
            else:
                return chr(comm)

    async def parse_command(self, cmd: str | None) -> GSIOCMessage:
        """
        Parses various command bytes.

        cmd -- the ascii code number of the command byte
        """

        if cmd.endswith(chr(13)):
            # buffered command
            cmd = cmd[:-1]

            # TODO: do stuff
            #print(f'Buffered command received: {cmd}')
            return GSIOCMessage(GSIOCCommandType.BUFFERED, cmd)

            #await self.send(f'You sent me buffered command {cmd}')
        else:
            # immediate (single-character) command
            return GSIOCMessage(GSIOCCommandType.IMMEDIATE, cmd)

    async def read1(self):
        """
        Reads a single byte and handles logging
        """
        comm = await self.read_async()
        if len(comm):
            #print(comm)
            #print(int(comm.hex(), base=16))#, len(comm), [ord(c) for c in comm])
            return int(comm.hex(), base=16)
        else:
            return None

    async def read_ack(self):
        """
        Waits for acknowledgement of a sent byte from the master
        """
        ret = await self.read1()
        if ret != 6:
            print(f'Warning: wrong acknowledgement character received: {ret}')

    async def write1(self, char: str):
        """
        Writes a single character to the serial port. Latin-1 encoding is critical.
        """
        await self.write_async(char.encode(encoding='latin-1'))

    async def send(self, msg: str):
        """
        Sends a message to the serial port, one character at a time
        """
        for char in msg[:-1]:
            await self.write1(char)
            await self.read_ack()

        # last character gets sent with high bit (add ASCII 128)
        await self.write1(chr(ord(msg[-1]) + 128))

        print(f'{datetime.datetime.now().isoformat()}: {self.port} (GSIOC) => {msg}')

class GSIOCTestDevice:

    def __init__(self, gsioc: GSIOC) -> None:
        self.gsioc = gsioc
        self.gsioc_command_queue: asyncio.Queue = asyncio.Queue()
        self.idle = True

    async def monitor_gsioc(self) -> None:
        """Monitor GSIOC queue
        """

        while True:
            data: GSIOCMessage = await self.gsioc.message_queue.get()

            response = await self.handle_gsioc(data)
            if data.messagetype == GSIOCCommandType.IMMEDIATE:
                await self.gsioc.response_queue.put(response)

    async def handle_gsioc(self, data: GSIOCMessage) -> str | None:
        """Handles GSIOC messages. Base version only handles idle requests

        Args:
            data (GSIOCMessage): GSIOC Message to be parsed / handled

        Returns:
            str: response (only for GSIOC immediate commands)
        """

        if data.data == 'Q':
            # busy query
            return 'idle' if self.idle else 'busy'
        
        elif data.data.startswith('sleep: '):
            sleeptime = float(data.data.split('sleep: ', 1)[1])
            await self.gsioc_command_queue.put(asyncio.create_task(self.sleep(sleeptime)))
        
        else:
            return 'error: unknown command'
        
        return None

    async def sleep(self, sleeptime: float) -> None:
        """Sleeps a time sleeptime and updates idle flag"""

        self.idle = False
        print(f'Sleeping {sleeptime} s...')
        await asyncio.sleep(sleeptime)
        self.idle = True

    async def gsioc_actions(self) -> None:
        """Monitors GSIOC command queue and performs actions asynchronously
        """

        while True:
            task: asyncio.Future = self.gsioc_command_queue.get()
            await task

async def main():

    virtual_port = 'COM13'
    baud_rate = 19200
    read_timeout = 0.1
    write_timeout = 0.1
    gsioc_address = 62

#    signal.signal(signal.SIGINT, signal_handler)

    gsioc = GSIOC(gsioc_address, port=virtual_port, baudrate=baud_rate, parity=aioserial.PARITY_EVEN)
    gtd = GSIOCTestDevice(gsioc)
    await asyncio.gather(gsioc.listen(), gtd.monitor_gsioc(), gtd.gsioc_actions())


if __name__ == '__main__':

    asyncio.run(main(), debug=True)