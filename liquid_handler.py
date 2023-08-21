import asyncio
import logging
import json
from functools import wraps
from assemblies import AssemblyBasewithGSIOC
from gsioc import GSIOCMessage, GSIOCCommandType

def log_method(func):
    """Decorator for logging start and end time of methods"""
    @wraps(func)
    async def inner(self, *args, **kwargs):
        logging.info(f'{self.name}: starting method {func.__name__}')
        result = await func(self, *args, **kwargs)
        logging.info(f'{self.name}: end of method {func.__name__}')
        return result
    return inner

class SimLiquidHandler:
    """Simulated liquid handler device"""

    def __init__(self, assembly: AssemblyBasewithGSIOC, name='SimLiquidHandler'):
        self.assembly = assembly
        self.name = name

    @log_method
    async def test_log(self, msg = 'hello') -> None:
        print(msg)
        return 5

    @log_method
    async def LoopInject(self,
                         pump_volume: float = 0,
                         pump_flow_rate: float = 1,
                         air_gap_plus_extra_volume: float = 0,
                         tag_name: str = '',
                         sleep_time: float = 0,
                         record_time: float = 0
                         ) -> None:
        """Simulated LoopInject method on liquid handler

        Args:
            pump_volume (float, optional): Volume of sample to inject (uL). Defaults to 0.
            pump_flow_rate (float, optional): Injection flow rate (mL / min). Defaults to 1.
            air_gap_plus_extra_volume (float, optional): Volume of air gap and extra volume (uL). Defaults to 0.
            tag_name (str, optional): Name of recording tag. Defaults to ''.
            sleep_time (float, optional): Equilibration time before recording (seconds). Defaults to 0.
            record_time (float, optional): Recording time (seconds). Defaults to 0.
        """

        aspirate_flow_rate = 2.0
        inject_flow_rate = 2.0

        init_dict = {'method': 'LoopInject',
                     'kwargs': {'pump_volume': pump_volume,
                                'pump_flow_rate': pump_flow_rate,
                                'air_gap_plus_extra_volume': air_gap_plus_extra_volume,
                                'tag_name': tag_name,
                                'sleep_time': sleep_time,
                                'record_time': record_time}}
        init_message = GSIOCMessage(GSIOCCommandType.BUFFERED, json.dumps(init_dict))

        logging.info(f'{self.name}: sending initialization message {init_message}')

        # initialize the remote method
        await self.assembly.handle_gsioc(init_message)

        logging.info(f'{self.name}: waiting for wait status')

        # wait for a waiting status (simulates WaitUntilWaiting method in LH)
        await self.assembly.waiting.wait()

        logging.info(f'{self.name}: sending dead volume request')
        response = await self.assembly.handle_gsioc(GSIOCMessage(GSIOCCommandType.IMMEDIATE, 'V'))

        # get the dead volume
        dead_volume = float(response)
        logging.info(f'{self.name}: got dead volume {dead_volume}, waiting for wait status')

        # start aspiration
        time_aspirate = 60 * ((pump_volume + air_gap_plus_extra_volume * 2 + dead_volume) / 1000) / (aspirate_flow_rate)
        logging.info(f'{self.name}: aspirating for {time_aspirate} seconds')
        await asyncio.sleep(time_aspirate)
        
        logging.info(f'{self.name}: waiting for wait status')
        # wait for a waiting status (simulates WaitUntilWaiting method in LH)
        await self.assembly.waiting.wait()
        
        logging.info(f'{self.name}: sending trigger')
        # send trigger for injection
        response = await self.assembly.handle_gsioc(GSIOCMessage(GSIOCCommandType.IMMEDIATE, 'T'))

        # simulate injection
        time_dispense = 60 * ((pump_volume + air_gap_plus_extra_volume * 2 + dead_volume) / 1000) / (inject_flow_rate)
        logging.info(f'{self.name}: dispensing for {time_aspirate} seconds')
        await asyncio.sleep(time_dispense)


        # wait for a waiting status (simulates WaitUntilWaiting method in LH)
        await self.assembly.waiting.wait()

        # send trigger that injection is complete
        response = await self.assembly.handle_gsioc(GSIOCMessage(GSIOCCommandType.IMMEDIATE, 'T'))

if __name__ == '__main__':

    logging.basicConfig(
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)

    assy = AssemblyBasewithGSIOC([], 'testassembly')
    lh = SimLiquidHandler(assy)

    async def main():
        result = await lh.test_log('ablkasdf')
        print(result)

    asyncio.run(main())




