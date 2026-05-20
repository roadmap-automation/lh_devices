"""Broker integration for gilson_lh device service.

Inbound (broker → gilson_lh):
  command.gilson_lh.submit_task  — payload: {method_name, sample_id, parameters}

Outbound (gilson_lh → broker):
  task.accepted          — immediately on valid command receipt
  task.completed         — after Trilution job completes
  task.failed            — after Trilution job fails
  layout.updated         — after job completes
  waste.generated        — per-method waste stream

Sequential dispatch:
  prefetch_count=1 + await method.start() naturally gates the queue.
  One job runs at a time; the next submit_task is not delivered until done.

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

from .lhinterface import LHInterface, LHJobHistory
from .job import ResultStatus
from .reservation import reservation_store

logger = logging.getLogger(__name__)

DEVICE_ID = 'lh'


class GilsonLHBrokerWorker:
    """Broker consumer/publisher for the Gilson LH service.

    Wire in by calling await worker.start() after layout is loaded.
    """

    def __init__(
        self,
        lh_iface: LHInterface,
        local_port: int = 5001,
        device_id: str = DEVICE_ID,
    ) -> None:
        self.device_id = device_id
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
        method_name = payload.get('method_name', '')
        parameters = {**payload.get('parameters', {}), 'sample_id': sample_id, 'task_id': task_id}

        # Idempotency: if this task_id is already in job history, re-publish completed
        with LHJobHistory() as history:
            existing = history.get_job_by_uuid(task_id)
        if existing is not None and existing.get_result_status() == ResultStatus.SUCCESS:
            logger.info("[%s] task %s already complete — re-publishing.", self.device_id, task_id)
            await self._emit(TASK_COMPLETED, envelope, {})
            return

        if method_name not in self.lh_iface.methods:
            logger.error("[%s] Unknown method: %s", self.device_id, method_name)
            await self._emit(TASK_FAILED, envelope, {'error': f'Unknown method: {method_name}'})
            return

        await self._emit(TASK_ACCEPTED, envelope, {})

        method = self.lh_iface.methods[method_name]
        result = await method.start(**parameters)  # blocks; message un-acked until complete

        if result.result.get('error'):
            await self._emit(TASK_FAILED, envelope, {'error': result.result['error']})
        else:
            payload_out: dict = {}
            rc = result.result.get('resolved_composition')
            if rc:
                payload_out['resolved_composition'] = rc
            for waste_data in result.result.get('waste', []):
                await self._emit_waste_raw(waste_data)
            await self._emit_layout_updated()
            await self._emit(TASK_COMPLETED, envelope, payload_out)

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

    async def _emit_waste_raw(self, payload: dict) -> None:
        if self._exchange is None:
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
