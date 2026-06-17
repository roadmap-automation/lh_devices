import asyncio
import copy
import json
import logging
import sqlite3
import traceback

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Callable, Tuple

from pydantic import BaseModel

from lh_devices.autocontrolplugin import AutocontrolPlugin
from lh_devices.device import DeviceBase
from lh_devices.layout import LayoutPlugin
from lh_devices.core.bedlayout import LHBedLayout
from lh_devices.waste import WasteInterfaceBase

from .job import JobBase, ResultStatus, ValidationStatus
from .notify import notifier
from .lhmethods import BaseLHMethod
from .app_config import config

DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'

LH_JOB_HISTORY = config.persistent_path / 'lh_jobs.sqlite'


class InterfaceStatus(str, Enum):
    UP = 'up'
    BUSY = 'busy'
    DOWN = 'down'
    ERROR = 'error'


class SampleList(BaseModel):
    """Class representing a sample list in JSON
        serializable format for Gilson Trilution LH Web Service"""
    name: str
    id: str | None
    createdBy: str
    description: str
    createDate: str
    startDate: str
    endDate: str
    columns: List[dict] | None


class LHJob(JobBase):
    """Container for a single liquid handler sample list"""

    LH_id: int | None = None
    LH_methods: List[BaseLHMethod] | None = None
    LH_method_data: dict | None = None

    def get_validation_status(self) -> Tuple[ValidationStatus, dict | None]:
        if not len(self.validation):
            return ValidationStatus.UNVALIDATED, None

        if self.validation['validation']['validationType'] == 'SUCCESS':
            return ValidationStatus.SUCCESS, None
        else:
            return ValidationStatus.FAIL, self.validation

    def get_result_status(self) -> ResultStatus:
        if not len(self.results):
            return ResultStatus.EMPTY

        results = self.get_results()

        if ResultStatus.FAIL in results:
            return ResultStatus.FAIL

        if ResultStatus.INCOMPLETE in results:
            return ResultStatus.INCOMPLETE

        return ResultStatus.SUCCESS

    def get_number_of_methods(self) -> int:
        if self.LH_method_data['columns'] is None:
            return 0
        else:
            return len(self.LH_method_data['columns'])

    def get_results(self) -> List[ResultStatus]:
        # Build a slot→latest-result map so retries override earlier failures.
        # self.results is append-only (history preserved); last entry per slot wins.
        by_slot: dict[int, dict] = {}
        for result in self.results:
            slot = int(result['sampleData']['runData'][0]['iteration']) - 1
            by_slot[slot] = result

        results = []
        for slot in range(self.get_number_of_methods()):
            if slot not in by_slot:
                results.append(ResultStatus.INCOMPLETE)
            else:
                notifs = by_slot[slot]['sampleData']['resultNotifications']['notifications'].values()
                all_ok = all('completed successfully' in n for n in notifs)
                results.append(ResultStatus.SUCCESS if all_ok else ResultStatus.FAIL)
        return results

    def setup_method_data(self, sample_id: str, sample_description: str,
                          lh_methods: List[BaseLHMethod], layout: LHBedLayout) -> None:
        from .resolver import WellResolver
        resolver = WellResolver(layout, sample_id)
        createdDate = datetime.now().strftime(DATE_FORMAT)
        method_list = [m2 for m in lh_methods
                       for m2 in m.render_lh_method(sample_id, sample_description, resolver)]
        # Build canonical field order from first-appearance across all rows (dict preserves insertion order).
        canonical: dict = {}
        for m in method_list:
            canonical.update(dict.fromkeys(m))
        # Rebuild every row with identical key order, filling gaps with None.
        method_list = [{k: m.get(k) for k in canonical} for m in method_list]
        self.LH_method_data = SampleList(
            name=sample_id,
            id='0',  # placeholder; updated to real LH_id in _sync_activate_job
            createdBy='System',
            description=sample_description,
            createDate=str(createdDate),
            startDate=str(createdDate),
            endDate=str(createdDate),
            columns=method_list).model_dump()
        self.LH_methods = lh_methods

    def get_method_data(self, listonly=False) -> dict:
        samplelist = copy.copy(self.LH_method_data)
        samplelist['id'] = str(self.LH_id)
        if listonly:
            samplelist['columns'] = None
        return samplelist

    def execute_methods(self, layout: LHBedLayout) -> None:
        for m in self.LH_methods:
            result = m.execute(layout)


class LHJobHistory:
    table_name = 'lh_job_record'
    table_definition = f"""\
        CREATE TABLE IF NOT EXISTS {table_name}(
            uuid TEXT PRIMARY KEY,
            LH_id INTEGER,
            job JSON,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) WITHOUT ROWID;"""

    def __init__(self, database_path: str = LH_JOB_HISTORY) -> None:
        self.db_path: str = database_path
        self.db: sqlite3.Connection | None = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def open(self) -> None:
        import os
        db_exists = os.path.exists(self.db_path)
        self.db = sqlite3.connect(self.db_path)
        if not db_exists:
            self.db.execute(self.table_definition)

    def close(self) -> None:
        self.db.close()

    def smart_insert(self, job: LHJob) -> None:
        self.db.execute(f"""\
            INSERT INTO {self.table_name}(uuid, LH_id, job) VALUES (?, ?, ?)
            ON CONFLICT(uuid) DO UPDATE SET
              LH_id=excluded.LH_id,
              job=excluded.job;
        """, (job.id, job.LH_id, job.model_dump_json(exclude={'LH_methods'})))
        self.db.commit()

    def get_job_by_uuid(self, uuid: str) -> LHJob | None:
        res = self.db.execute(f"SELECT job FROM {self.table_name} WHERE uuid='{uuid}'")
        results = res.fetchall()
        return None if not len(results) else LHJob(**json.loads(results[0][0]))

    def get_job_by_LH_id(self, LH_id: str) -> LHJob | None:
        res = self.db.execute(f"SELECT job FROM {self.table_name} WHERE LH_id='{LH_id}'")
        results = res.fetchall()
        return None if not len(results) else LHJob(**json.loads(results[0][0]))

    def get_max_LH_id(self) -> int:
        res = self.db.execute(f"SELECT MAX(LH_id) FROM {self.table_name}")
        maxval = res.fetchone()
        return maxval[0]


class LHInterface(AutocontrolPlugin, DeviceBase, LayoutPlugin):
    """Gilson LH interface device.

    Inherits:
        AutocontrolPlugin — channels property, method_callbacks, /SubmitTask routes
        DeviceBase        — idle/state/get_info/event_handler/trigger_update web pattern
        LayoutPlugin      — bed layout management and /GUI/* routes
    """

    def __init__(self,
                 device_id: str = 'lh',
                 name: str = 'Gilson LH Interface',
                 database_path: Path | None = None) -> None:
        # Explicit base inits following the RinseSystem pattern
        DeviceBase.__init__(self, device_id=device_id, name=name)
        AutocontrolPlugin.__init__(self, database_path=database_path, id=self.id, name=self.name)
        LayoutPlugin.__init__(self, id=self.id, name=self.name)

        self._active_job: LHJob | None = None
        self.running: bool = True
        self.has_error: bool = False

        # Async callables: f(job, *args, **kwargs) -> None
        self.validation_callbacks: List[Callable] = []
        self.results_callbacks: List[Callable] = []

        # Set by deactivate() to unblock any waiting _run_job; cleared before each activate_job
        self._job_cancelled: asyncio.Event = asyncio.Event()

        # Waste tracker; replaced with a broker-aware implementation in app.py if needed
        self.waste_tracker = WasteInterfaceBase()

        # Device is always ready; no hardware handshake at startup
        self.initialized = True

        # Register all Gilson LH methods in the lh_devices MethodRunner
        from .lhmethods import (GilsonTransferWithRinse, GilsonMixWithRinse, GilsonInjectWithRinse,
                                 GilsonSleep, GilsonPrime, GilsonQCMDLoadLoop, GilsonQCMDDirectInject,
                                 GilsonDirectInjectPrime, GilsonFormulation, GilsonSoluteFormulation)
        self.methods.update({
            'NCNR_TransferWithRinse': GilsonTransferWithRinse(self),
            'NCNR_MixWithRinse': GilsonMixWithRinse(self),
            'NCNR_InjectWithRinse': GilsonInjectWithRinse(self),
            'NCNR_Sleep': GilsonSleep(self),
            'NCNR_Prime': GilsonPrime(self),
            'ROADMAP_QCMD_LoadLoop': GilsonQCMDLoadLoop(self),
            'ROADMAP_QCMD_DirectInject': GilsonQCMDDirectInject(self),
            'ROADMAP_DirectInjectPrime': GilsonDirectInjectPrime(self),
            'Formulation': GilsonFormulation(self),
            'SoluteFormulation': GilsonSoluteFormulation(self),
        })

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> InterfaceStatus:
        if not self.running:
            return InterfaceStatus.DOWN
        if self.has_error:
            return InterfaceStatus.ERROR
        if self._active_job is not None:
            return InterfaceStatus.BUSY
        return InterfaceStatus.UP

    def get_active_job(self) -> LHJob | None:
        return self._active_job

    # ------------------------------------------------------------------
    # History (async wrapper over blocking SQLite)
    # ------------------------------------------------------------------

    def _sync_update_history(self) -> None:
        if self._active_job is not None:
            with LHJobHistory() as history:
                history.smart_insert(self._active_job)

    async def _async_update_history(self) -> None:
        await asyncio.to_thread(self._sync_update_history)

    # ------------------------------------------------------------------
    # Job management (all async; callers are aiohttp handlers or coroutines)
    # ------------------------------------------------------------------

    def _sync_update_job(self, job: LHJob) -> None:
        if self._active_job is not None:
            if job.id != self._active_job.id:
                raise RuntimeError(f'Received update for job {job.id} but active job is {self._active_job.id}')
        else:
            raise RuntimeError(f'Received update for job {job.id} but no active job exists')
        self._active_job = job

    async def update_job_result(self, job: LHJob, *args, **kwargs) -> None:
        self._sync_update_job(job)
        await self._async_update_history()
        if job.get_result_status() == ResultStatus.SUCCESS:
            self.has_error = False
        await asyncio.gather(*[cb(job, *args, **kwargs) for cb in self.results_callbacks])
        await self.trigger_update()

    async def update_job_validation(self, job: LHJob, *args, **kwargs) -> None:
        self._sync_update_job(job)
        await self._async_update_history()
        await asyncio.gather(*[cb(job, *args, **kwargs) for cb in self.validation_callbacks])
        await self.trigger_update()

    async def throw_error(self, msg: str) -> None:
        self.has_error = True
        logging.error(f'Error in {self.name}\n' + msg)
        notifier.notify(f'Error in {self.name}', msg)
        await self.trigger_update()

    def _sync_activate_job(self, job: LHJob) -> None:
        """Synchronous part of activate_job: ID assignment (method_data pre-built by caller)."""
        with LHJobHistory() as history:
            max_LH_id = history.get_max_LH_id()

        if max_LH_id is None:
            max_LH_id = 0

        job.LH_id = max_LH_id + 1
        if job.LH_method_data is not None:
            job.LH_method_data['id'] = str(job.LH_id)
        self._active_job = job

    async def activate_job(self, job: LHJob) -> None:
        if self.get_status() != InterfaceStatus.UP:
            raise RuntimeError('Attempted to activate job but LHInterface is not idle')

        self._job_cancelled.clear()

        try:
            await asyncio.to_thread(self._sync_activate_job, job)
        except Exception:
            await self.throw_error(traceback.format_exc())
            return

        self.idle = False
        await self._async_update_history()
        await self.trigger_update()

    async def deactivate(self) -> None:
        if self._active_job is not None:
            self._job_cancelled.set()
        self._active_job = None
        self.idle = True
        await self._async_update_history()

    # ------------------------------------------------------------------
    # get_info — merge AutocontrolPlugin (active_methods) + DeviceBase (state/controls)
    # ------------------------------------------------------------------

    async def get_info(self) -> dict:
        d = await AutocontrolPlugin.get_info(self)
        d.update(await DeviceBase.get_info(self))

        # Mirror has_error into DeviceBase.error so roadmap.html shows the red border.
        d['state']['error']['error'] = 'LH error' if self.has_error else None

        status = self.get_status()
        d['lh_status'] = status
        d['active_job'] = self._active_job.model_dump() if self._active_job is not None else None

        display: dict = {'Status': status.value}
        if self._active_job is not None:
            md = self._active_job.LH_method_data or {}
            display['Job ID'] = self._active_job.LH_id
            display['Name'] = md.get('name', '')
            display['Description'] = md.get('description', '') or None
        d['state']['display'] = display
        d['state']['pre'] = (
            {'label': f'Active job (LH_id={self._active_job.LH_id})', 'data': self._active_job.model_dump()}
            if self._active_job is not None else None
        )
        from lh_devices.methods import method_schemas_for_display
        d['method_schemas'] = method_schemas_for_display(self.methods)
        d['controls'] = d['controls'] | {
            'pause_resume': {
                'type': 'button',
                'text': 'Resume' if not self.running else 'Pause',
                'enabled': True,
            },
            'clear_lh_error': {
                'type': 'button',
                'text': 'Clear Error',
                'visible': self.has_error,
                'enabled': self.has_error,
            },
            'resubmit_active_job': {
                'type': 'button',
                'text': 'Resubmit Active Job',
                'visible': self._active_job is not None,
                'enabled': self._active_job is not None,
            },
            'deactivate': {
                'type': 'button',
                'text': 'Clear Active Job',
                'visible': self._active_job is not None,
                'enabled': self._active_job is not None,
            },
        }
        return d

    # ------------------------------------------------------------------
    # event_handler — four UI buttons, delegate remainder to AutocontrolPlugin
    # ------------------------------------------------------------------

    async def event_handler(self, command: str, data: dict) -> None:
        if command == 'pause_resume':
            self.running = not self.running
            await self.trigger_update()
        elif command == 'clear_lh_error':
            if self.has_error:
                self.has_error = False
                await self.trigger_update()
        elif command == 'resubmit_active_job':
            if self._active_job is not None:
                self._active_job.LH_id += 1
                await self.trigger_update()
        elif command == 'deactivate':
            if self._active_job is not None:
                await self.deactivate()
                await self.trigger_update()
        elif command == 'run_method':
            method_name = data.get('method_name')
            method_data = data.get('method_data', {})
            if method_name in self.methods:
                method_data['name'] = method_name
                method_data.setdefault('sample_id', 'GUI')
                asyncio.create_task(self.methods[method_name].start(**method_data))
        else:
            await AutocontrolPlugin.event_handler(self, command, data)

    # ------------------------------------------------------------------
    # create_web_app — combine MethodPlugin routes + layout routes + Trilution routes
    # ------------------------------------------------------------------

    def create_web_app(self, template='roadmap.html'):
        from .webview import get_routes
        from aiohttp import web

        # MethodPlugin.create_web_app adds /SubmitTask, /GetStatus, /GetTaskData
        # and calls WebNodeBase.create_web_app for the base socket.io app
        app = super().create_web_app(template)

        # Layout routes: /GUI/GetLayout, /GUI/GetWells, /GUI/UpdateWell, etc.
        app.add_routes(LayoutPlugin._get_routes(self))

        # Trilution callback routes: /LH/GetState, /LH/PutSampleData, etc.
        app.add_routes(get_routes(self))

        return app


lh_interface = LHInterface()
