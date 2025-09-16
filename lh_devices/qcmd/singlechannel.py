import asyncio
import logging
import datetime
import pathlib

from lh_manager.liquid_handler.bedlayout import LHBedLayout, Rack
from .multichannel import MultiChannelAssembly, QCMDMeasurementDevice, QCMDMeasurementChannel, QCMDAcceptTransfer
from ..layout import LayoutPlugin
from ..notify import notifier
from ..webview import run_socket_app

LOG_PATH = pathlib.Path(__file__).parent.parent.parent / 'logs'
HISTORY_PATH = pathlib.Path(__file__).parent.parent.parent / 'history'
NOTIFICATION_CONFIG_PATH = pathlib.Path(__file__).parent.parent.parent / 'notification_settings.json'

class QCMDSingleMeasurementDevice(MultiChannelAssembly, LayoutPlugin):
    """QCMD recording device for a single QCMD instrument"""

    def __init__(self,
                 qcmd_address: str = 'localhost',
                 qcmd_port: int = 5011,
                 qcmd_id: str | None = None,
                 database_path: str | None = None,
                 layout_path: str | None = None,
                 name='Single Channel QCMD Measurement Device') -> None:

        if qcmd_id is not None:
            channel = QCMDMeasurementChannel(
                QCMDMeasurementDevice(f'http://{qcmd_address}:{qcmd_port}/QCMD/id/{qcmd_id}/',
                                      name=f'QCMD Measurement Device, Serial Number {qcmd_id}'),
                name=f'QCMD Measurement Channel'
            )
        else:
            channel = QCMDMeasurementChannel(
                QCMDMeasurementDevice(f'http://{qcmd_address}:{qcmd_port}/QCMD/0/',
                                      name=f'QCMD Measurement Device'),
                name=f'QCMD Measurement Device'
            )

        super().__init__(channels=[channel],
                         assemblies=[],
                         database_path=database_path,
                         name=name)

        # set up layout        
        LayoutPlugin.__init__(self, self.id, self.name)
        self.layout_path = layout_path

        # attempt to load the layout from log file
        self.load_layout()

        if self.layout is None:
            racks = {
                channel.name: Rack(columns=1,
                                   rows=1,
                                   max_volume=1,
                                   min_volume=0.0,
                                   wells=[channel.well],
                                   style='grid',
                                   height=300,
                                   width=300,
                                   x_translate=0,
                                   y_translate=0,
                                   shape='circle',
                                   editable=False)
            }
            self.layout = LHBedLayout(racks=racks)
        else:
            channel.well = self.layout.racks[channel.name].wells[0]

        # add AcceptTransfer method, which updates the layout
        # trigger a layout update whenever any method runs
        async def trigger_layout_update(result):
            await self.trigger_layout_update()

        channel.methods.update({'QCMDAcceptTransfer': QCMDAcceptTransfer(channel, self.layout)})
        channel.method_callbacks.append(trigger_layout_update)

    def create_web_app(self, template='roadmap.html'):
        app = super().create_web_app(template)

        app.add_routes(LayoutPlugin._get_routes(self))

        return app

    async def event_handler(self, command: str, data: dict) -> None:
        """Handles events from web interface

        Args:
            command (str): command name
            data (dict): any data required by the command
        """

        await super().event_handler(command, data)

    async def get_info(self):
        d = await super().get_info()

        return d

async def qcmd_single_channel_measure():
    # connect to error notifier
    notifier.load_config(NOTIFICATION_CONFIG_PATH)
    notifier.connect()

    measurement_system = QCMDSingleMeasurementDevice('localhost', 5011,
                                                     qcmd_id=None,
                                                     database_path=HISTORY_PATH / 'qcmd.db',
                                                     layout_path=LOG_PATH / 'qcmd_layout.json')
    await measurement_system.initialize()
    app = measurement_system.create_web_app(template='roadmap.html')
    runner = await run_socket_app(app, 'localhost', 5006)
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        logging.info('Closing QCMD Single Channel Measurement Device...')
        await runner.cleanup()

if __name__ == '__main__':
    logging.basicConfig(handlers=[
                            logging.FileHandler(LOG_PATH / (datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '_qcmd_log.txt')),
                            logging.StreamHandler()
                        ],
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    asyncio.run(qcmd_single_channel_measure(), debug=True)