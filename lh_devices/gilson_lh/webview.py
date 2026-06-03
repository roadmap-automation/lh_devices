"""Gilson Trilution LH 4.0 aiohttp callback endpoints.

Routes are built by get_routes(lh_iface) and added to the app in app.py.
lh_iface is both the LHInterface and the LayoutPlugin after the refactor.
"""

import json
import logging

from aiohttp import web

from lh_devices.core.bedlayout import Composition

from .formulation import solve_formulation
from .lhinterface import LHJob, LHJobHistory, InterfaceStatus, LHInterface
from .job import ResultStatus, ValidationStatus

logger = logging.getLogger(__name__)


def _json(data, status: int = 200) -> web.Response:
    return web.Response(text=json.dumps(data), status=status, content_type='application/json')


def get_routes(lh_iface: LHInterface) -> web.RouteTableDef:
    """Returns route table wired to lh_iface (which is also the layout plugin)."""

    routes = web.RouteTableDef()

    # ------------------------------------------------------------------
    # Management / introspection
    # ------------------------------------------------------------------

    @routes.get('/LH/GetJob/{job_id}')
    @routes.get('/LH/GetJob/{job_id}/')
    async def GetJob(request: web.Request) -> web.Response:
        job_id = request.match_info['job_id']
        with LHJobHistory() as history:
            job = history.get_job_by_uuid(job_id)
        if job is not None:
            return _json({'success': job.model_dump()})
        return _json({'error': f'job {job_id} does not exist'}, 400)

    @routes.get('/LH/GetActiveJob')
    @routes.get('/LH/GetActiveJob/')
    async def GetActiveJob(request: web.Request) -> web.Response:
        job = lh_iface.get_active_job()
        return _json({'active_job': job.model_dump() if job is not None else None})

    @routes.get('/LH/GetState')
    @routes.get('/LH/GetState/')
    async def GetState(request: web.Request) -> web.Response:
        return _json({
            'active_job': lh_iface._active_job.model_dump() if lh_iface._active_job is not None else None,
            'status': lh_iface.get_status(),
        })

    @routes.get('/LH/GetListofSampleLists')
    @routes.get('/LH/GetListofSampleLists/')
    async def GetListofSampleLists(request: web.Request) -> web.Response:
        job: LHJob | None = lh_iface.get_active_job()
        sample_list = [] if job is None else [job.get_method_data(listonly=True)]
        return _json({'sampleLists': sample_list})

    @routes.get('/LH/GetSampleList/{sample_list_id}')
    @routes.get('/LH/GetSampleList/{sample_list_id}/')
    async def GetSampleList(request: web.Request) -> web.Response:
        sample_list_id = request.match_info['sample_list_id']
        job = lh_iface.get_active_job()
        if job is None:
            return _json({'error': 'no active jobs'}, 400)
        if int(sample_list_id) != job.LH_id:
            return _json({'error': f'requested job ID {sample_list_id} does not match active job ID {job.LH_id}'}, 400)
        return _json({'sampleList': job.get_method_data(listonly=False)})

    # ------------------------------------------------------------------
    # Formulation check
    # ------------------------------------------------------------------

    @routes.post('/LH/CheckFormulation')
    @routes.post('/LH/CheckFormulation/')
    async def CheckFormulation(request: web.Request) -> web.Response:
        data = await request.json()
        try:
            target_composition = Composition(**data.get('target_composition', {}))
            target_volume = float(data.get('target_volume', 0.0))
            exact_match = bool(data.get('exact_match', True))
            result = solve_formulation(
                layout=lh_iface.layout,
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
    @routes.post('/LH/PutSampleListValidation/{sample_list_id}/')
    async def PutSampleListValidation(request: web.Request) -> web.Response:
        sample_list_id = request.match_info['sample_list_id']
        data = await request.json()
        job = lh_iface.get_active_job()
        if job is None:
            return _json({'error': 'no active jobs'}, 400)
        if int(sample_list_id) != job.LH_id:
            return _json({'error': f'validation job ID {sample_list_id} does not match active job ID {job.LH_id}'}, 400)

        job.validation = data
        await lh_iface.update_job_validation(job, job.get_validation_status()[0])

        error = None
        validation_status, _ = job.get_validation_status()
        if validation_status != ValidationStatus.SUCCESS:
            error = 'Error in validation. Full message: ' + data['validation']['message']
            lh_iface.has_error = True
            await lh_iface.deactivate()
            await lh_iface.trigger_update()
        else:
            lh_iface.has_error = False

        return _json({sample_list_id: validation_status, 'error': error})

    # ------------------------------------------------------------------
    # Trilution callbacks — results
    # ------------------------------------------------------------------

    @routes.post('/LH/PutSampleData')
    @routes.post('/LH/PutSampleData/')
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
        result_status = job.get_result_status()
        await lh_iface.update_job_result(job, method_number, method_name, result_status)

        if result_status == ResultStatus.FAIL:
            await lh_iface.throw_error('Error in results. Full message: ' + repr(data))
        # execute_methods is now called by GilsonLHMethod._run_job after done.wait()

        return _json({'data': data})

    # ------------------------------------------------------------------
    # Trilution callbacks — error / control
    # ------------------------------------------------------------------

    @routes.post('/LH/ReportError')
    @routes.post('/LH/ReportError/')
    async def ReportError(request: web.Request) -> web.Response:
        data = await request.json()
        await lh_iface.throw_error('Error in results. Full message: ' + repr(data))
        return _json({'data': data})

    @routes.post('/LH/ResetErrorState')
    @routes.post('/LH/ResetErrorState/')
    async def ResetErrorState(request: web.Request) -> web.Response:
        lh_iface.has_error = False
        await lh_iface.trigger_update()
        return _json({'success': 'error state reset'})

    @routes.post('/LH/ResubmitActiveJob')
    @routes.post('/LH/ResubmitActiveJob/')
    async def ResubmitActiveJob(request: web.Request) -> web.Response:
        if lh_iface._active_job is None:
            return _json({'error': 'no active job'}, 400)
        lh_iface._active_job.LH_id += 1
        await lh_iface.trigger_update()
        return _json({'success': f'LH_id incremented to {lh_iface._active_job.LH_id}'})

    @routes.post('/LH/Deactivate')
    @routes.post('/LH/Deactivate/')
    async def Deactivate(request: web.Request) -> web.Response:
        await lh_iface.deactivate()
        await lh_iface.trigger_update()
        return _json({'success': 'deactivated'})

    @routes.post('/LH/PauseResume')
    @routes.post('/LH/PauseResume/')
    async def PauseResume(request: web.Request) -> web.Response:
        lh_iface.running = not lh_iface.running
        await lh_iface.trigger_update()
        return _json({'status': lh_iface.get_status()})

    return routes
