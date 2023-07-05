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

class GSIOC(aioserial.AioSerial):
    """
    Virtual GSIOC object.
    """

    def __init__(self, gsioc_address: int, port: str='COM13', baudrate: int=19200, parity=aioserial.PARITY_EVEN):
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

        #signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal, frame):
        """
        Handles stop signals for a clean exit
        """
        self.interrupt = True

    async def listen(self):
        """
        Starts GSIOC listener. Only ends when an interrupt signal is received.
        """

        print('Starting GSIOC listener... Ctrl+C to exit.')
        self.interrupt = False

        # infinite loop to wait for a command
        while not self.interrupt:

            await self.wait_for_connection()

            if self.connected: # address received
                # waits for a command
                cmd = await self.wait_for_command()

                # parses received command
                await self.parse_command(cmd)

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
            print(f'address matches, writing {chr(self.address + 128).encode()}')
            await self.write1(chr(self.address + 128))
            self.connected = True
        
        # if result is byte 255, send a break character
        # TODO: check if this should be self.ser.send_break
        elif comm == 255:
            print('sending break')
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
                print(f'got LF, starting buffered command read')
                msg = ''
                while comm != 13:
                    # get a character
                    comm = await self.read1()

                    # acknowledge character
                    await self.write1(chr(comm))

                    msg += chr(comm)

                print(f'got CR, end of message: {msg}')

                return msg[-1] if len(msg) else None
            
            # immediate command...read a single character
            else:
                return chr(comm)

    async def parse_command(self, cmd: str | None):
        """
        Parses various command bytes.

        cmd -- the ascii code number of the command byte
        """
        if cmd is not None:
            if cmd.endswith(chr(13)):
                # buffered command
                cmd = cmd[-1]

                # TODO: do stuff
                print(f'Buffered command received: {cmd}')
            else:
                # immediate (single-character) command
                if cmd == '%':
                    # identification request
                    await self.send('Virtual GSIOC')
                else:
                    # all other commands
                    print(f'Immediate command received: {cmd}')
                    await self.send(f'You sent me immediate command {cmd}')

    async def read1(self):
        """
        Reads a single byte and handles logging
        """
        comm = await self.read_async()
        if len(comm):
            print(comm)
            print(int(comm.hex(), base=16))#, len(comm), [ord(c) for c in comm])
            return int(comm.hex(), base=16)
        else:
            return None

    async def read_ack(self):
        """
        Waits for acknowledgement of a sent byte from the master
        """
        ret = await self.read1()
        if ret == 6:
            print('acknowledgement received')
        else:
            print(f'wrong acknowledgement character received: {ret}')

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

async def main():

    virtual_port = 'COM13'
    baud_rate = 19200
    read_timeout = 0.1
    write_timeout = 0.1
    gsioc_address = 62

#    signal.signal(signal.SIGINT, signal_handler)

    gsioc = GSIOC(gsioc_address, port=virtual_port, baudrate=baud_rate, parity=aioserial.PARITY_EVEN)
    await asyncio.gather(gsioc.listen())


if __name__ == '__main__':

    asyncio.run(main(), debug=True)