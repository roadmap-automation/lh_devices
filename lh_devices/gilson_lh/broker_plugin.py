"""Broker integration for gilson_lh device service.

Inbound (broker → gilson_lh):
  command.gilson_lh.submit_task  — payload: {method_name, sample_id, parameters}

Outbound (gilson_lh → broker):
  task.accepted          — immediately on valid command receipt
  task.completed         — after Trilution job completes; payload includes resolved_composition
  task.failed            — after Trilution job fails
  layout.updated         — after job completes and layout is mutated
  waste.generated        — per-method waste stream

Sequential dispatch:
  prefetch_count=1 + awaiting job completion before acking naturally gates
  the queue. One LHJob runs at a time; the next submit_task is not delivered
  until the current job is done.

Idempotency:
  LHJobHistory is checked for task_id before activating. If already present,
  task.completed is re-published and the message is acked without hardware action.

Subprotocol cleanup:
  Subscribes to protocol_studio.run.step_completed and releases well reservations
  for the sample_id on SUBPROTOCOL_COMPLETED step type.
"""

import asyncio
import logging
from typing import Optional

import aio_pika

from roadmap_broker_client.connection import get_connection
from roadmap_broker_client.consumer import consume
from roadmap_broker_client.envelope import Envelope, build
from roadmap_broker_client.publisher import publish
from roadmap_broker_client.topology import declare_node_queue, declare_event_queue, declare_topology
from roadmap_broker_client.topics import (
    DEVICE_REGISTERED,
    INSTRUMENT_EXCHANGE,
    LAYOUT_UPDATED,
    TASK_ACCEPTED,
    TASK_COMPLETED,
    TASK_FAILED,
    WASTE_GENERATED,
    RUN_STEP_COMPLETED,
    PROTOCOL_EXCHANGE,
)

from lh_devices.core.methods import method_manager
from lh_devices.core.bedlayout import Composition

from .lhinterface import LHInterface, LHJob, LHJobHistory, InterfaceStatus
from .job import ResultStatus
from .reservation import reservation_store

logger = logging.getLogger(__name__)

DEVICE_ID = 'gilson_lh'


class GilsonLHBrokerWorker:
    """Broker consumer/publisher for the Gilson LH service.

    Wire in by calling await worker.start() after layout is loaded.
    """

    def __init__(
        self,
        layout_plugin,
        lh_iface: LHInterface,
        local_port: int = 5001,
        device_id: str = DEVICE_ID,
    ) -> None:
        self.device_id = device_id
        self.layout_plugin = layout_plugin
        self.lh_iface = lh_iface
        self.local_port = local_port
        self._exchange: Optional[aio_pika.abc.AbstractExchange] = None
        self._protocol_exchange: Optional[aio_pika.abc.AbstractExchange] = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        connection = await get_connection()
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        await declare_topology(channel)

        self._exchange = await channel.get_exchange(INSTRUMENT_EXCHANGE)
        self._protocol_exchange = await channel.get_exchange(PROTOCOL_EXCHANGE)

        # Subscribe to gilson_lh command queue on instrument exchange
        cmd_queue = await declare_node_queue(channel, self.device_id, INSTRUMENT_EXCHANGE)
        asyncio.create_task(consume(cmd_queue, self._on_command))

        # Subscribe to protocol step completions for reservation cleanup
        step_queue = await declare_event_queue(
            channel, f'gilson_lh.step_events', PROTOCOL_EXCHANGE, RUN_STEP_COMPLETED,
        )
        asyncio.create_task(consume(step_queue, self._on_step_completed))

        # Announce presence
        await self._emit_device_registered()
        await self._emit_layout_updated()
        logger.info("GilsonLHBrokerWorker [%s] running on port %d.", self.device_id, self.local_port)

    # ------------------------------------------------------------------
    # Inbound: command.gilson_lh.submit_task
    # ------------------------------------------------------------------

    async def _on_command(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        verb = (message.routing_key or '').split('.')[-1]

        if verb == 'submit_task':
            await self._on_submit_task(envelope, message)
        else:
            logger.warning("[%s] Unknown command verb '%s'", self.device_id, verb)

    async def _on_submit_task(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        task_id = str(envelope.task_id)
        payload = envelope.payload
        sample_id = str(envelope.sample_id or payload.get('sample_id', ''))

        # Idempotency: if this task_id is already in job history, re-publish completed
        with LHJobHistory() as history:
            existing = history.get_job_by_uuid(task_id)
        if existing is not None and existing.get_result_status() == ResultStatus.SUCCESS:
            logger.info("[%s] task %s already complete — re-publishing.", self.device_id, task_id)
            await self._publish_completed(existing, envelope)
            return

        method_name = payload.get('method_name', '')
        parameters = payload.get('parameters', {})

        # Resolve method and build LH method list
        method_cls = method_manager.get_method_by_name(method_name)
        if method_cls is None:
            logger.error("[%s] Unknown method: %s", self.device_id, method_name)
            await self._emit(TASK_FAILED, envelope, {'error': f'Unknown method: {method_name}'})
            return

        layout = self.layout_plugin.layout
        if layout is None:
            logger.error("[%s] Layout not loaded", self.device_id)
            await self._emit(TASK_FAILED, envelope, {'error': 'Layout not loaded'})
            return

        method = method_cls(**parameters)
        # explode() resolves Formulation → list of transfer/mix methods
        flat_methods = method.explode(layout)

        method_list = [
            {
                'sample_name': sample_id,
                'sample_description': '',
                'method_name': m.method_name,
                'method_data': m.model_dump(exclude={'status', 'tasks', 'id'}),
            }
            for m in flat_methods
        ]

        job = LHJob(id=task_id, method_data={'method_list': method_list})

        await self._emit(TASK_ACCEPTED, envelope, {})

        # Gate: wait for interface to be idle (sequential dispatch)
        while self.lh_iface.get_status() != InterfaceStatus.UP:
            await asyncio.sleep(0.5)

        # Set up completion gate
        done = asyncio.Event()
        resolved_composition: list[Composition] = []

        def _on_result(completed_job: LHJob, *args, **kwargs) -> None:
            if completed_job.id != task_id:
                return
            # Collect resolved composition from layout mutations
            try:
                carrier = layout.carrier_well
                if carrier is not None:
                    resolved_composition.append(carrier.composition)
            except Exception:
                pass
            asyncio.ensure_future(self._on_job_done(completed_job, envelope, resolved_composition, done))

        self.lh_iface.results_callbacks.append(_on_result)

        try:
            self.lh_iface.activate_job(job, layout)
        except RuntimeError as exc:
            logger.error("[%s] activate_job failed: %s", self.device_id, exc)
            self.lh_iface.results_callbacks.remove(_on_result)
            await self._emit(TASK_FAILED, envelope, {'error': str(exc)})
            return

        # Wait for Trilution to call back and complete the job
        await done.wait()
        self.lh_iface.results_callbacks.remove(_on_result)

    async def _on_job_done(
        self,
        job: LHJob,
        envelope: Envelope,
        resolved_composition: list,
        done: asyncio.Event,
    ) -> None:
        status = job.get_result_status()

        if status == ResultStatus.SUCCESS:
            await self._publish_completed(job, envelope, resolved_composition)
            # Publish waste for each method
            for m in (job.LH_methods or []):
                try:
                    waste = m.waste(self.layout_plugin.layout)
                    await self._emit_waste(waste)
                except Exception:
                    pass
            await self._emit_layout_updated()
        else:
            await self._emit(TASK_FAILED, envelope, {'error': f'Job result: {status}'})

        done.set()

    # ------------------------------------------------------------------
    # Inbound: protocol_studio.run.step_completed (reservation cleanup)
    # ------------------------------------------------------------------

    async def _on_step_completed(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        payload = envelope.payload or {}
        if payload.get('step_type') == 'SUBPROTOCOL_COMPLETED':
            sample_id = str(envelope.sample_id or payload.get('sample_id', ''))
            if sample_id:
                reservation_store.release_sample(sample_id)

    # ------------------------------------------------------------------
    # Outbound helpers
    # ------------------------------------------------------------------

    async def _publish_completed(
        self,
        job: LHJob,
        envelope: Envelope,
        resolved_composition: list | None = None,
    ) -> None:
        payload: dict = {'method_name': job.method_data.get('method_list', [{}])[0].get('method_name', '')}
        if resolved_composition:
            payload['resolved_composition'] = resolved_composition[0].model_dump()

        msg = build(
            device_id=self.device_id,
            routing_key=TASK_COMPLETED,
            task_id=envelope.task_id,
            sample_id=envelope.sample_id,
            assigned_channel=envelope.assigned_channel,
            execution_policy=envelope.execution_policy or 'irreversible',
            payload=payload,
        )
        await publish(self._exchange, TASK_COMPLETED, msg)

    async def _emit_device_registered(self) -> None:
        if self._exchange is None:
            return
        msg = build(
            device_id=self.device_id,
            routing_key=DEVICE_REGISTERED,
            payload={
                "device_id": self.device_id,
                "display_name": "Gilson 271 Liquid Handler",
                "device_type": "lh",
                "num_channels": 1,
                "allow_sample_mixing": True,
                "address": f"http://localhost:{self.local_port}",
            },
        )
        await publish(self._exchange, DEVICE_REGISTERED, msg)
        logger.info("[%s] device.registered published.", self.device_id)

    async def _emit_layout_updated(self) -> None:
        if self._exchange is None:
            return
        msg = build(
            device_id=self.device_id,
            routing_key=LAYOUT_UPDATED,
            payload={
                'device_name': self.device_id,
                'retrieval_uri': f'http://localhost:{self.local_port}',
            },
        )
        await publish(self._exchange, LAYOUT_UPDATED, msg)

    async def _emit_waste(self, waste) -> None:
        if self._exchange is None:
            return
        try:
            payload = waste.model_dump()
        except AttributeError:
            return
        msg = build(device_id=self.device_id, routing_key=WASTE_GENERATED, payload=payload)
        await publish(self._exchange, WASTE_GENERATED, msg)

    async def _emit(self, routing_key: str, envelope: Envelope, extra: dict) -> None:
        if self._exchange is None:
            return
        msg = build(
            device_id=self.device_id,
            routing_key=routing_key,
            task_id=envelope.task_id,
            sample_id=envelope.sample_id,
            assigned_channel=envelope.assigned_channel,
            execution_policy=envelope.execution_policy or 'irreversible',
            payload=extra,
        )
        await publish(self._exchange, routing_key, msg)
