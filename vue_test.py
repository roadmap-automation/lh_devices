import asyncio
from uuid import uuid4
from typing import Any, Coroutine
from aiohttp.web_app import Application as Application
from webview import WebNodeBase, run_socket_app
from valve import SyringeYValve, ValveBase
from HamiltonDevice import HamiltonSyringePump
from HamiltonComm import HamiltonSerial

class VueTest(WebNodeBase):

    def __init__(self, valve: ValveBase) -> None:
        super().__init__()
        self.id = str(uuid4())
        self.idle = False
        self.valve = valve
    
    def create_web_app(self, template: str) -> Application:
        return super().create_web_app(template)
    
    async def get_info(self) -> Coroutine[Any, Any, dict]:
        d = await super().get_info()

        d.update({'idle': self.idle,
                  'valve': self.valve.get_info()})

        return d

    async def event_handler(self, command: str, data: dict) -> None:
        await super().event_handler(command, data)
    
        if command=='move_valve':
            self.valve.move(int(data['position']))

async def main():

    import functools

    async def monkey_patch_event_handler(self: HamiltonSyringePump, command: str, data: dict) -> None:
    
        if command=='toggle_idle':
            self.idle = not self.idle
        
        elif command=='move_valve':
            self.valve.position = data['index']

        else:
            await HamiltonSyringePump.event_handler(self, command, data)

        await self.trigger_update()

    #vuetest = VueTest(valve=SyringeYValve(1))
    vuetest = HamiltonSyringePump(HamiltonSerial(), '0', SyringeYValve(1), 5000, name='Syringe Pump')
    vuetest.event_handler = functools.partial(monkey_patch_event_handler, vuetest)
    vuetest.idle = True
    app = vuetest.create_web_app('index.html')
    runner = await run_socket_app(app, 'localhost', 5020)
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        await runner.cleanup()

if __name__=='__main__':

    import logging
    logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

    asyncio.run(main(), debug=True)