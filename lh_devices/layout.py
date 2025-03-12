import asyncio
import logging
import json

from aiohttp import web
from aiohttp.web_app import Application as Application
from pathlib import Path

from lh_manager.liquid_handler.bedlayout import LHBedLayout, Rack, Well

from .webview import WebNodeBase, sio

class LayoutPlugin(WebNodeBase):

    def __init__(self, id: str = '', name: str = ''):

        self.id = id
        self.name = name
        self.layout: LHBedLayout | None = None
        self.layout_path: Path | None = None

    def save_layout(self):
        """Saves the layout to a JSON file
        """

        if self.layout_path is not None:
            if self.layout_path.is_file():
                with open(self.layout_path, 'w') as f:
                    json.dump(self.layout.model_dump(), f, indent=2)         

    async def _get_layout(self, request: web.Request) -> web.Response:
        return web.Response(text=self.layout.model_dump_json() if self.layout is not None else json.dumps(None), status=200)

    async def _get_wells(self, request: web.Request) -> web.Response:
        if self.layout is not None:
            wells = self.layout.get_all_wells()
            wells_dict = [well.model_dump() for well in wells]
            for wd in wells_dict:
                wd['zone'] = None
            return web.Response(text=json.dumps(wells_dict), status=200)
        else:
            return web.Response(text=json.dumps(None), status=200)

    async def _update_well(self, request: web.Request) -> web.Response:
        data = await request.json()
        assert isinstance(data, dict)
        well = Well(**data)
        self.layout.update_well(well)
        await self.trigger_layout_update()
        return web.Response(text=well.model_dump_json(), status=200)
    
    async def _remove_well(self, request: web.Request) -> web.Response:
        data = await request.json()
        assert isinstance(data, dict)
        self.layout.remove_well_definition(data["rack_id"], data["well_number"])
        await self.trigger_layout_update()
        return web.Response(text={"well definition removed": data}, status=200)

    def _get_routes(self) -> web.RouteTableDef:

        routes = web.RouteTableDef()

        @routes.get('/GUI/GetLayout')
        async def get_layout(request: web.Request) -> web.Response:
            return await self._get_layout(request)
        
        @routes.get('/GUI/GetWells')
        async def get_wells(request: web.Request) -> web.Response:
            return await self._get_wells(request)        
       
        @routes.post('/GUI/UpdateWell')
        async def update_well(request: web.Request) -> web.Response:
            return await self._update_well(request)

        @routes.post('/GUI/RemoveWellDefinition')
        async def remove_well(request: web.Request) -> web.Response:
            return await self._remove_well(request)            

        return routes

    def create_web_app(self, template='roadmap.html') -> Application:
        app = super().create_web_app(template=template)

        app.add_routes(self._get_routes())

        return app

    async def trigger_layout_update(self):
        """Emits a socketio event with id"""

        await sio.emit(self.id, 'update_layout')
        self.save_layout()