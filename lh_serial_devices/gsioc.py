import logging
from enum import Enum
from dataclasses import dataclass

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
        self.client_lock: asyncio.Lock = asyncio.Lock()
        self.gsioc_name = gsioc_name
        self.message_queue: asyncio.Queue = asyncio.Queue(1)
        self.response_queue: asyncio.Queue = asyncio.Queue(1)

    async def listen(self) -> None:
        """
        Starts GSIOC listener. Only ends when an interrupt signal is received.
        """

        logging.info('Starting GSIOC listener... Ctrl+C to exit.')

        if not self.is_open:
            self.open()

        # initialize recovery data
        recovery_data: dict | None = None

        # infinite loop to wait for a command. Break by cancelling the task
        try:
            while True:

                logging.debug('GSIOC: Waiting for connection...')

                await self.wait_for_connection()

                if self.connected: # address received
                    logging.debug('GSIOC: Connection established, waiting for command')
                    # waits for a command
                    cmd = await self.wait_for_command()

                    if cmd:

                        logging.debug(f'{self.port} (GSIOC) <= {cmd}')

                        # process ID request immediately
                        if cmd == '%':
                            await self.send(self.gsioc_name)
                        
                        else:

                            # parses received command into a GSIOCMessage
                            data = await self.parse_command(cmd)
                            
                            # if previous send failed
                            if recovery_data is not None:
                                # if we're handling a repeated command, use recovery data, otherwise proceed as normal
                                if recovery_data['command'] == cmd:
                                    logging.debug(f"GSIOC: Using recovery data for repeat command {recovery_data['command']} with response {recovery_data['response']}")
                                    response = recovery_data['response']                                
                                else:
                                    recovery_data = None
                            
                            # normal requests OR if previous send failed and it is not a repeat command
                            if recovery_data is None:

                                # put message in listener queue if a client is connected; otherwise, send an error for immediate commands
                                if self.client_lock.locked():
                                    await self.message_queue.put(data)

                                if data.messagetype == GSIOCCommandType.IMMEDIATE:
                                    if self.client_lock.locked():
                                        logging.debug('GSIOC: Waiting for response to immediate command')
                                        response: str = await self.response_queue.get()
                                    else:
                                        response: str = f'GSIOC error: no client connected on address {self.address}'

                            # attempt to send and save recovery data if not
                            if data.messagetype == GSIOCCommandType.IMMEDIATE:
                                if await self.send(response):
                                    recovery_data = None
                                else:
                                    recovery_data = {'command': cmd,
                                                    'response': response}


                logging.debug('GSIOC: Connection reset...')
        except asyncio.CancelledError:
            logging.info('Closing GSIOC connection...')
            #await self.write1(chr(10))
        except Exception:
            raise

        finally:

            # close serial port before exiting when interrupt is received
            self.message_queue.empty()
            self.response_queue.empty()
            self.close()

    async def wait_for_connection(self):
        """
        Waits for an address byte (connection request). Also handles byte 255, which
        disconnects all GSIOC listeners, by sending an active break signal.
        """

        comm = await self.read1()

        # if address matches, echo the address
        if comm == self.address + 128:
            await self.write1(chr(self.address + 128))
            logging.debug(f'GSIOC: address matches, writing {chr(self.address + 128).encode()}')
            self.connected = True
        
        # if result is byte 255, send a break character
        # TODO: check if this should be self.ser.send_break
        elif comm == 255:
            logging.debug('GSIOC: sending break')
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
                logging.debug(f'GSIOC: got LF, starting buffered command read')
                await self.write1(chr(comm))
                msg = ''
                while comm != 13:
                    # get a character
                    comm = await self.read1()

                    # acknowledge character
                    await self.write1(chr(comm))

                    msg += chr(comm)

                logging.debug(f'GSIOC: got CR, end of message: {msg}')

                return msg
            
            # address repeated (lagging communications); resend my own address and wait again
            elif comm == self.address + 128:
                await self.write1(chr(self.address + 128))

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
            logging.debug(f'GSIOC: Buffered command received: {cmd}')
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
            logging.debug(f'GSIOC: received character {int(comm.hex(), base=16)}') #, len(comm), [ord(c) for c in comm])
            return int(comm.hex(), base=16)
        else:
            return None

    async def read_ack(self) -> bool:
        """
        Waits for acknowledgement of a sent byte from the master
        """
        ret = await self.read1()
        if ret != 6:
            logging.warning(f'GSIOC: wrong acknowledgement character received: {ret}; attempting to repair comms')
            #await self.repair_comms()
            return False

        return True

    async def repair_comms(self):
        """
        Attempts to repair comms
        """

        # TODO: try sending break signal, close and re-open serial port, etc.

        self.reset_input_buffer()
        await asyncio.sleep(0)

    async def write1(self, char: str):
        """
        Writes a single character to the serial port. Latin-1 encoding is critical.
        """
        await self.write_async(char.encode(encoding='latin-1'))

    async def send(self, msg: str) -> dict | None:
        """
        Sends a message to the serial port, one character at a time. 

        Returns True if successful, False if read acknowledgement error
        """
        for char in msg[:-1]:
            logging.debug(f'GSIOC: sending character {char}')
            await self.write1(char)
            if not await self.read_ack():
                logging.warning(f'GSIOC: Terminating send of message {msg} due to wrong acknowledgement character, waiting for new connection')
                return False

        # last character gets sent with high bit (add ASCII 128)
        await self.write1(chr(ord(msg[-1]) + 128))

        logging.debug(f'{self.port} (GSIOC) => {msg}')

        return True

async def main():

    virtual_port = 'COM13'
    baud_rate = 19200
    read_timeout = 0.1
    write_timeout = 0.1
    gsioc_address = 62

#    signal.signal(signal.SIGINT, signal_handler)

    gsioc = GSIOC(gsioc_address, port=virtual_port, baudrate=baud_rate, parity=aioserial.PARITY_EVEN)
    #gtd = GSIOCTimer(gsioc)
    #await gtd.initialize()
    #await asyncio.gather(gsioc.listen(), gtd.monitor_gsioc(), gtd.gsioc_actions())


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO)

    asyncio.run(main(), debug=True)