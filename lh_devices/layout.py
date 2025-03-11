import asyncio
import logging

from aiohttp import web
from aiohttp.web_app import Application as Application

from lh_manager.liquid_handler.bedlayout import LHBedLayout, Rack, Well

from .webview import WebNodeBase

class LayoutPlugin(WebNodeBase):

    def __init__(self, id: str = '', name: str = ''):

        self.id = id
        self.name = name
        self.layout: LHBedLayout | None = None

    async def _get_layout(self, request: web.Request) -> web.Response:
        return web.Response(text=self.layout.model_dump_json(), status=200)

    async def _update_well(self, request: web.Request) -> web.Response:
        data = await request.json()
        assert isinstance(data, dict)
        well = Well(**data)
        self.layout.update_well(well)
        return web.Response(text=well.model_dump_json(), status=200)
    
    async def _remove_well(self, request: web.Request) -> web.Response:
        data = await request.json()
        assert isinstance(data, dict)
        self.layout.remove_well_definition(data["rack_id"], data["well_number"])
        return web.Response(text={"well definition removed": data}, status=200)

    def _get_routes(self) -> web.RouteTableDef:

        routes = web.RouteTableDef()

        @routes.post('/GUI/GetLayout')
        async def get_layout(request: web.Request) -> web.Response:
            return await self._get_layout(request)
       
        @routes.get('/GUI/UpdateWell')
        async def update_well(request: web.Request) -> web.Response:
            return await self._update_well(request)

        @routes.get('/GUI/RemoveWellDefinition')
        async def remove_well(request: web.Request) -> web.Response:
            return await self._remove_well(request)            

        return routes

    def create_web_app(self, template='roadmap.html') -> Application:
        app = super().create_web_app(template=template)
        
        app.add_routes(self._get_routes())

        return app