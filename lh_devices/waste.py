"""
Waste plugin used by methods for reporting waste stream contents
"""

from aiohttp import ClientSession, ClientConnectionError
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from lh_manager.waste_manager.waste import WasteItem, Composition

from .logutils import Loggable

WATER = Composition(solvents=[dict(name='H2O',
                                   fraction=1.0)])

@dataclass
class WasteResponse:
    success: bool | None = None
    response: dict = field(default_factory=dict)

class WasteInterfaceBase(Loggable):

    async def submit(self, waste: WasteItem) -> WasteResponse:
        
        return WasteResponse()

    async def submit_water(self, volume: float) -> WasteResponse:
        """Convenience function for water

        Args:
            volume (float): volume of water in mL

        Returns:
            WasteResponse: response
        """

        return await self.submit(WasteItem(composition=WATER,
                                    volume=volume))

class RoadmapWasteInterface(WasteInterfaceBase):

    def __init__(self, url: str | None = None):
        super().__init__()
        url_parts = urlsplit(url)
        self.session = ClientSession(f'{url_parts.scheme}://{url_parts.netloc}')
        self.url_path = url_parts.path
        self.timeout = 3

    async def submit(self, waste: WasteItem) -> WasteResponse:
        """Submits 

        Args:
            waste (WasteItem): waste to submit

        Returns:

            WasteResponse: response
        """

        post_data = waste.model_dump()
        headers = {'Content-Type': 'application/json'}

        self.logger.debug(f'{self.session._base_url}{self.url_path} => {post_data}')

        response = {}

        try:
            async with self.session.post(self.url_path, headers=headers, json=post_data, timeout=self.timeout) as resp:
                response_json = await resp.json()
                self.logger.debug(f'{self.session._base_url}{self.url_path} <= {response_json}')
                response = WasteResponse(success=True, response=response_json)
        except (ConnectionRefusedError, ClientConnectionError):
            self.logger.error(f'request to {self.session._base_url}{self.url_path} failed: connection refused')
            response = WasteResponse(success=False, response={'error': 'connection refused'})
        except TimeoutError:
            self.logger.error(f'request to {self.session._base_url}{self.url_path} failed: timed out')
            response = WasteResponse(success=False, response={'error': f'timed out after {self.timeout} seconds'})

        return response
    

 