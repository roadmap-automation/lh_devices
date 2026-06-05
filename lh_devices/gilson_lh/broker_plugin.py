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
    DEVICE_ANNOUNCE_REQUEST,
    DEVICE_REGISTERED,
    INSTRUMENT_EXCHANGE,
    LAYOUT_UPDATED,
    PROTOCOL_EXCHANGE,
    SUBPROTOCOL_COMPLETED,
    SUBPROTOCOL_FAILED,
    TASK_ACCEPTED,
    TASK_COMPLETED,
    TASK_FAILED,
    WASTE_GENERATED,
    composition_transfer_key,
    gsioc_dead_volume_key,
    gsioc_trigger_key,
)

from ..gilson.gsioc import GSIOC, GSIOCCommandType, GSIOCMessage
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
        gsioc: Optional[GSIOC] = None,
    ) -> None:
        self.device_id = device_id
        self.lh_iface = lh_iface
        self.local_port = local_port
        self._exchange: Optional[aio_pika.abc.AbstractExchange] = None
        self._protocol_exchange: Optional[aio_pika.abc.AbstractExchange] = None
        self._amqp_channel: Optional[aio_pika.abc.AbstractChannel] = None
        self._methods_schema: dict[str, dict] = {}  # built in start()

        # GSIOC serial ↔ broker translation
        self._gsioc: Optional[GSIOC] = gsioc
        self._current_gsioc_task_id: Optional[str] = None
        self._dead_volume_event: asyncio.Event = asyncio.Event()
        self._dead_volume_value: str = ''
        # Separate channel for GSIOC dead-volume subscriptions.
        # Must not share self._amqp_channel: that channel has prefetch_count=1
        # and holds the submit_task message unacked for the entire Trilution job,
        # which would block delivery of the dead-volume message to any consumer
        # on the same channel.
        self._gsioc_amqp_channel: Optional[aio_pika.abc.AbstractChannel] = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        connection = await get_connection()
        channel = await connection.channel()
        self._amqp_channel = channel
        await channel.set_qos(prefetch_count=1)
        await declare_topology(channel)

        self._exchange = await channel.get_exchange(INSTRUMENT_EXCHANGE)
        self._protocol_exchange = await channel.get_exchange(PROTOCOL_EXCHANGE)

        # Dedicated channel for GSIOC dead-volume subscriptions — no QoS limit
        # so delivery is never blocked by the unacked submit_task message above.
        gsioc_channel = await connection.channel()
        self._gsioc_amqp_channel = gsioc_channel

        # Build method schemas from the LH interface's registered methods.
        for name, method_instance in self.lh_iface.methods.items():
            get_schema = getattr(type(method_instance), 'get_pydantic_schema', None)
            if get_schema is not None:
                try:
                    self._methods_schema[name] = get_schema()
                except Exception:
                    logger.debug("Could not build schema for lh method '%s'", name, exc_info=True)

        # Subscribe to gilson_lh command queue on instrument exchange
        cmd_queue = await declare_node_queue(channel, self.device_id, INSTRUMENT_EXCHANGE)
        asyncio.create_task(consume(cmd_queue, self._on_command))

        # Subscribe to subprotocol end events for well reservation cleanup.
        # Both COMPLETED and FAILED trigger release so wells are never leaked.
        subprotocol_queue = await channel.declare_queue(
            f'{self.device_id}.subprotocol_events',
            durable=False,
            auto_delete=True,
            arguments={'x-dead-letter-exchange': 'exchange.dead_letter'},
        )
        for rk in (SUBPROTOCOL_COMPLETED, SUBPROTOCOL_FAILED):
            await subprotocol_queue.bind(self._protocol_exchange, routing_key=rk)
        asyncio.create_task(consume(subprotocol_queue, self._on_subprotocol_end))

        # Subscribe to re-announce requests so lh_manager can trigger re-registration.
        announce_queue = await channel.declare_queue(
            f"{self.device_id}.announce_request",
            durable=False,
            auto_delete=True,
        )
        await announce_queue.bind(self._exchange, routing_key=DEVICE_ANNOUNCE_REQUEST)
        asyncio.create_task(consume(announce_queue, self._on_announce_request))

        # Wire layout_callbacks so HTTP-driven updates (UpdateWell, UpdateRack, etc.)
        # publish layout.updated to the broker, not just task-completion updates.
        self.lh_iface.layout_callbacks.append(self._emit_layout_updated)

        # Start GSIOC serial listener and client loop if a GSIOC port is configured.
        if self._gsioc is not None:
            asyncio.create_task(self._gsioc.listen())
            asyncio.create_task(self._gsioc_client_loop())

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

        # Autocontrol wraps the method in method_data.method_list[0].
        method_list = payload.get('method_data', {}).get('method_list', [])
        if method_list:
            method_name = method_list[0].get('method_name', '')
            raw_params = method_list[0].get('method_data', {})
        else:
            method_name = payload.get('method_name', '')
            raw_params = payload.get('parameters', {})

        parameters = {**raw_params, 'sample_id': sample_id, 'task_id': task_id}

        # Idempotency: if this task_id is already in job history, re-publish completed
        with LHJobHistory() as history:
            existing = history.get_job_by_uuid(task_id)
        if existing is not None and existing.get_result_status() == ResultStatus.SUCCESS:
            logger.info("[%s] task %s already complete — re-publishing.", self.device_id, task_id)
            await self._emit(TASK_COMPLETED, envelope, {})
            return

        if method_name not in self.lh_iface.methods:
            logger.error("[%s] Unknown method: %s", self.device_id, method_name)
            print(payload)
            await self._emit(TASK_FAILED, envelope, {'error': f'Unknown method: {method_name}'})
            return

        await self._emit(TASK_ACCEPTED, envelope, {})

        # Set up GSIOC broker correlation for this task window.
        dead_volume_task = None
        if self._gsioc is not None:
            self._current_gsioc_task_id = task_id
            self._dead_volume_event.clear()
            dead_volume_task = asyncio.create_task(self._subscribe_dead_volume(task_id))

        method = self.lh_iface.methods[method_name]
        result = await method.start(**parameters)  # blocks; message un-acked until complete

        if dead_volume_task is not None:
            dead_volume_task.cancel()
            self._current_gsioc_task_id = None

        if result.result.get('error'):
            await self._emit(TASK_FAILED, envelope, {'error': result.result['error']})
        else:
            payload_out: dict = {}
            rc = result.result.get('resolved_composition')
            if rc:
                payload_out['resolved_composition'] = rc
                # Publish composition.transfer so any peer device in the same
                # MethodGroup (e.g. injection system) can receive the resolved
                # composition before running its own method.
                await self._publish_composition_transfer(task_id, envelope, rc)
            for waste_data in result.result.get('waste', []):
                await self._emit_waste_raw(waste_data)
            await self._emit_layout_updated()
            await self._emit(TASK_COMPLETED, envelope, payload_out)

    # ------------------------------------------------------------------
    # Inbound: SUBPROTOCOL_COMPLETED / SUBPROTOCOL_FAILED (reservation cleanup)
    # ------------------------------------------------------------------

    async def _on_subprotocol_end(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        payload = envelope.payload or {}
        sample_id = payload.get('sample_id') or str(envelope.sample_id or '')
        if sample_id:
            reservation_store.release_sample(sample_id)
            logger.info("[%s] reservation cleanup for sample %s (%s).",
                        self.device_id, sample_id, message.routing_key)

    # ------------------------------------------------------------------
    # GSIOC serial ↔ broker translation (gilson_lh side)
    # ------------------------------------------------------------------

    async def _gsioc_client_loop(self) -> None:
        """Translate GSIOC serial commands to/from broker messages.

        Holds gsioc.client_lock so the listener delivers commands via message_queue.
        Commands handled:
          'Q' — IS busy query: responds 'waiting' if a task is active (IS is at hold point), 'idle' otherwise.
          'T' — trigger: publishes gsioc.trigger.<task_id>, responds ok.
          'V' — dead volume: waits for gsioc.dead_volume.<task_id> from IS via broker,
                then responds with the value. Safely blocks here — the serial listener
                awaits response_queue.get() with no timeout, so this does not violate
                GSIOC timing constraints (the 20ms limit applies to serial byte reads).
        """
        if self._gsioc is None:
            return
        async with self._gsioc.client_lock:
            try:
                while True:
                    data: GSIOCMessage = await self._gsioc.message_queue.get()

                    if data.data == 'Q':
                        response = 'waiting' if self._current_gsioc_task_id else 'idle'
                        await self._gsioc.response_queue.put(response)

                    elif data.data == 'T':
                        if self._current_gsioc_task_id and self._exchange:
                            rk = gsioc_trigger_key(self._current_gsioc_task_id)
                            msg = build(
                                device_id=self.device_id,
                                routing_key=rk,
                                payload={"task_id": self._current_gsioc_task_id},
                            )
                            await publish(self._exchange, rk, msg)
                            logger.debug("[%s] GSIOC trigger published for task %s.", self.device_id, self._current_gsioc_task_id)
                        await self._gsioc.response_queue.put('ok')

                    elif data.data == 'V':
                        await self._dead_volume_event.wait()
                        await self._gsioc.response_queue.put(self._dead_volume_value)
                        self._dead_volume_event.clear()
                        self._dead_volume_value = ''

                    else:
                        logger.warning("[%s] Unknown GSIOC command: %r", self.device_id, data.data)
                        if data.messagetype == GSIOCCommandType.IMMEDIATE:
                            await self._gsioc.response_queue.put('error')

            except asyncio.CancelledError:
                pass

    async def _subscribe_dead_volume(self, task_id: str) -> None:
        """Subscribe to gsioc.dead_volume.<task_id> and store the value when it arrives.

        The IS broker plugin publishes this after the method calls dead_volume.put().
        The stored value is read by _gsioc_client_loop when Trilution sends 'V'.

        Uses _gsioc_amqp_channel (no prefetch limit) rather than _amqp_channel.
        _amqp_channel has prefetch_count=1 and holds the submit_task message unacked
        for the entire Trilution job; sharing it would permanently block delivery here.
        """
        if self._gsioc_amqp_channel is None:
            return
        rk = gsioc_dead_volume_key(task_id)
        queue = await self._gsioc_amqp_channel.declare_queue(exclusive=True, auto_delete=True)
        await queue.bind(self._exchange, routing_key=rk)

        async def _on_dead_volume(msg: aio_pika.abc.AbstractIncomingMessage) -> None:
            async with msg.process():
                ev = Envelope.model_validate_json(msg.body)
                dv = (ev.payload or {}).get("dead_volume", 0.0)
                self._dead_volume_value = str(dv)
                self._dead_volume_event.set()
                logger.debug("[%s] dead_volume %.2f received for task %s.", self.device_id, dv, task_id)

        consumer_tag = await queue.consume(_on_dead_volume)
        try:
            await asyncio.Event().wait()  # hold open until cancelled from _on_submit_task
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await queue.cancel(consumer_tag)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Outbound helpers
    # ------------------------------------------------------------------

    async def _publish_composition_transfer(
        self, task_id: str, envelope: Envelope, resolved_composition: dict
    ) -> None:
        """Publish composition.transfer.<task_id> so MethodGroup peers receive the composition."""
        if self._exchange is None:
            return
        rk = composition_transfer_key(task_id)
        msg = build(
            device_id=self.device_id,
            routing_key=rk,
            task_id=envelope.task_id,
            sample_id=envelope.sample_id,
            payload={"resolved_composition": resolved_composition},
        )
        await publish(self._exchange, rk, msg)
        logger.debug("[%s] composition.transfer published for task %s.", self.device_id, task_id)

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
                "methods": self._methods_schema,
            },
        )
        await publish(self._exchange, DEVICE_REGISTERED, msg)
        logger.info("[%s] device.registered published (%d methods).", self.device_id, len(self._methods_schema))

    async def _on_announce_request(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        """Re-emit device.registered when lh_manager requests a fresh announcement."""
        logger.debug("[%s] announce_request received — re-publishing device.registered.", self.device_id)
        await self._emit_device_registered()

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
