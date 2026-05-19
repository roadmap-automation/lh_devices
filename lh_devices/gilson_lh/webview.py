"""Gilson Trilution LH 4.0 aiohttp callback endpoints.

Replaces lh_manager/lh_api/endpoints.py (Flask) with aiohttp routes.
Routes are built by get_routes(layout_plugin, lh_interface_inst) and
added to the app in app.py.
"""

import json
import traceback
import logging

from aiohttp import web

from lh_devices.core.bedlayout import Composition
from lh_devices.webview import sio

from .formulation import solve_formulation
from .lhinterface import LHJob, LHJobHistory, InterfaceStatus, LHInterface
from .job import ResultStatus, ValidationStatus

logger = logging.getLogger(__name__)


def _json(data, status: int = 200) -> web.Response:
    return web.Response(text=json.dumps(data), status=status, content_type='application/json')


def get_routes(layout_plugin, lh_iface: LHInterface) -> web.RouteTableDef:
    """Returns route table wired to layout_plugin and lh_iface."""

    routes = web.RouteTableDef()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _emit_lh_update():
        await sio.emit('update_lh_job', {'msg': 'update_lh_job'})

    async def _emit_layout_update():
        await layout_plugin.trigger_layout_update()

    # ------------------------------------------------------------------
    # Management / introspection
    # ------------------------------------------------------------------

    @routes.get('/LH/GetJob/{job_id}')
    async def GetJob(request: web.Request) -> web.Response:
        job_id = request.match_info['job_id']
        with LHJobHistory() as history:
            job = history.get_job_by_uuid(job_id)
        if job is not None:
            return _json({'success': job.model_dump()})
        return _json({'error': f'job {job_id} does not exist'}, 400)

    @routes.get('/LH/GetActiveJob')
    async def GetActiveJob(request: web.Request) -> web.Response:
        job = lh_iface.get_active_job()
        return _json({'active_job': job.model_dump() if job is not None else None})

    @routes.get('/LH/GetState')
    async def GetState(request: web.Request) -> web.Response:
        return _json({
            'active_job': lh_iface._active_job.model_dump() if lh_iface._active_job is not None else None,
            'status': lh_iface.get_status(),
        })

    @routes.get('/LH/GetListofSampleLists')
    async def GetListofSampleLists(request: web.Request) -> web.Response:
        job: LHJob | None = lh_iface.get_active_job()
        sample_list = [] if job is None else [job.get_method_data(listonly=True)]
        return _json({'sampleLists': sample_list})

    @routes.get('/LH/GetSampleList/{sample_list_id}')
    async def GetSampleList(request: web.Request) -> web.Response:
        sample_list_id = request.match_info['sample_list_id']
        job = lh_iface.get_active_job()
        if job is None:
            return _json({'error': 'no active jobs'}, 400)
        if int(sample_list_id) != job.LH_id:
            return _json({'error': f'requested job ID {sample_list_id} does not match active job ID {job.LH_id}'}, 400)
        return _json({'sampleList': job.get_method_data(listonly=False)})

    # ------------------------------------------------------------------
    # Job submission (called by broker_plugin, not Trilution)
    # ------------------------------------------------------------------

    @routes.post('/LH/SubmitJob')
    async def SubmitJob(request: web.Request) -> web.Response:
        if lh_iface.get_status() != InterfaceStatus.UP:
            return _json({'error': 'job rejected, LH interface busy'}, 400)
        data = await request.json()
        try:
            job = LHJob(**data)
        except Exception:
            return _json({'error': 'job rejected, cannot be deserialized'}, 400)
        lh_iface.activate_job(job, layout_plugin.layout)
        await _emit_lh_update()
        return _json({'success': 'job accepted'})

    # ------------------------------------------------------------------
    # Formulation check
    # ------------------------------------------------------------------

    @routes.post('/LH/CheckFormulation')
    async def CheckFormulation(request: web.Request) -> web.Response:
        data = await request.json()
        try:
            target_composition = Composition(**data.get('target_composition', {}))
            target_volume = float(data.get('target_volume', 0.0))
            exact_match = bool(data.get('exact_match', True))
            result = solve_formulation(
                layout=layout_plugin.layout,
                target_composition=target_composition,
                target_volume=target_volume,
                exact_match=exact_match,
            )
            if result['wells']:
                result['wells'] = [w.model_dump() for w in result['wells']]
            return _json(result)
        except Exception as exc:
            return _json({'success': False, 'error': str(exc)}, 400)

    # ------------------------------------------------------------------
    # Trilution callbacks — validation
    # ------------------------------------------------------------------

    @routes.post('/LH/PutSampleListValidation/{sample_list_id}')
    async def PutSampleListValidation(request: web.Request) -> web.Response:
        sample_list_id = request.match_info['sample_list_id']
        data = await request.json()
        job = lh_iface.get_active_job()
        if job is None:
            return _json({'error': 'no active jobs'}, 400)
        if int(sample_list_id) != job.LH_id:
            return _json({'error': f'validation job ID {sample_list_id} does not match active job ID {job.LH_id}'}, 400)

        job.validation = data
        lh_iface.update_job_validation(job, job.get_validation_status()[0])

        error = None
        if job.get_validation_status()[0] != ValidationStatus.SUCCESS:
            error = 'Error in validation. Full message: ' + data['validation']['message']
            lh_iface.has_error = True
            lh_iface.deactivate()
        else:
            lh_iface.has_error = False

        await _emit_lh_update()
        return _json({sample_list_id: job.get_validation_status()[0], 'error': error})

    # ------------------------------------------------------------------
    # Trilution callbacks — results
    # ------------------------------------------------------------------

    @routes.post('/LH/PutSampleData')
    async def PutSampleData(request: web.Request) -> web.Response:
        data = await request.json()
        assert isinstance(data, dict)

        sample_id = int(data['sampleData']['runData'][0]['sampleListID'])
        method_number = int(data['sampleData']['runData'][0]['iteration']) - 1
        method_name = data['sampleData']['runData'][0]['methodName']

        job = lh_iface.get_active_job()
        if job is None:
            return _json({'error': 'no active jobs'}, 400)
        if sample_id != job.LH_id:
            return _json({'error': f'PutSampleData job ID {sample_id} does not match active job ID {job.LH_id}'}, 400)
        if job.LH_method_data['columns'][method_number]['METHODNAME'] != method_name:
            return _json({'error': f'PutSampleData method name mismatch'}, 400)

        job.results.append(data)
        lh_iface.update_job_result(job, method_number, method_name, job.get_result_status())

        error = None
        if job.get_result_status() == ResultStatus.FAIL:
            error = 'Error in results. Full message: ' + repr(data)
            lh_iface.throw_error(error)
        elif job.get_result_status() == ResultStatus.SUCCESS:
            try:
                job.execute_methods(layout_plugin.layout)
            except Exception:
                lh_iface.throw_error(traceback.format_exc())
            await _emit_layout_update()

        await _emit_lh_update()
        return _json({'data': data})

    # ------------------------------------------------------------------
    # Trilution callbacks — error / control
    # ------------------------------------------------------------------

    @routes.post('/LH/ReportError')
    async def ReportError(request: web.Request) -> web.Response:
        data = await request.json()
        lh_iface.throw_error('Error in results. Full message: ' + repr(data))
        await _emit_lh_update()
        return _json({'data': data})

    @routes.post('/LH/ResetErrorState')
    async def ResetErrorState(request: web.Request) -> web.Response:
        lh_iface.has_error = False
        await _emit_lh_update()
        return _json({'success': 'error state reset'})

    @routes.post('/LH/ResubmitActiveJob')
    async def ResubmitActiveJob(request: web.Request) -> web.Response:
        if lh_iface._active_job is None:
            return _json({'error': 'no active job'}, 400)
        lh_iface._active_job.LH_id += 1
        await _emit_lh_update()
        return _json({'success': f'LH_id incremented to {lh_iface._active_job.LH_id}'})

    @routes.post('/LH/Deactivate')
    async def Deactivate(request: web.Request) -> web.Response:
        lh_iface.deactivate()
        await _emit_lh_update()
        return _json({'success': 'deactivated'})

    @routes.post('/LH/PauseResume')
    async def PauseResume(request: web.Request) -> web.Response:
        lh_iface.running = not lh_iface.running
        await _emit_lh_update()
        return _json({'status': lh_iface.get_status()})

    return routes
