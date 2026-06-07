"""Broker integration for lh_devices — injection and QCMD device services.

Runs entirely within the existing aiohttp asyncio event loop (no separate
thread). Provides:

  Inbound (broker → device):
    command.<device_id>.submit_task  — replaces POST /SubmitTask from autocontrol

  Outbound (device → broker):
    task.accepted          — immediately on receipt of a valid command
    task.completed         — after method finishes successfully (Claim Check)
    task.failed            — after method finishes with an error
    channel.status_changed — BUSY on dispatch, IDLE on completion
    waste.generated        — replaces outbound POST to lh_manager/Waste/AddWaste/

Idempotency:
    Before executing any physical action, checks HistoryDB for the task_id.
    If already present, re-publishes task.completed and acks without running
    hardware. Prevents duplicate physical actions on message redelivery.

Claim Check:
    task.completed carries data_reference.retrieval_uri pointing to the local
    GET /GetTaskData endpoint. Full MethodResult stays in SQLite.
"""

import asyncio
import dataclasses
import logging
import pathlib
import types
from typing import Optional, Protocol, runtime_checkable

import aio_pika

from roadmap_broker_client.connection import get_connection
from roadmap_broker_client.consumer import consume
from roadmap_broker_client.envelope import DataReference, Envelope, build
from roadmap_broker_client.publisher import publish
from roadmap_broker_client.topology import declare_node_queue, declare_topology
from roadmap_broker_client.topics import (
    CHANNEL_STATUS_CHANGED,
    DEVICE_ANNOUNCE_REQUEST,
    DEVICE_REGISTERED,
    INSTRUMENT_EXCHANGE,
    LAYOUT_UPDATED,
    TASK_ACCEPTED,
    TASK_COMPLETED,
    TASK_FAILED,
    WASTE_GENERATED,
    composition_transfer_key,
    gsioc_dead_volume_key,
    gsioc_trigger_key,
)

from .history import HistoryDB
from .methods import (
    MethodBasewithBrokerTrigger,
    MethodBasewithCompositionRelay,
    MethodBasewithCompositionReceive,
    MethodResult,
)
from .waste import WasteInterfaceBase, WasteResponse

logger = logging.getLogger(__name__)


_PY_TO_JSON = {'float': 'number', 'int': 'number', 'str': 'string', 'bool': 'boolean'}

def _resolve_json_type(t) -> str:
    """Map a Python type annotation to a JSON-compatible type string.

    Handles plain types (int, float, str, bool) and union types like str|int,
    str|float — the latter are common in MethodDefinition fields where the code
    accepts either a string (from JSON) or the actual numeric type.
    """
    if isinstance(t, str):
        return _PY_TO_JSON.get(t, t)
    if isinstance(t, types.UnionType):
        for arg in t.__args__:
            if arg is not str and arg is not type(None):
                return _resolve_json_type(arg)
        return 'string'
    name = getattr(t, '__name__', None)
    if name:
        return _PY_TO_JSON.get(name, name)
    return repr(t)


def _schema_from_method_class(method_class: type, method_type: str = 'none') -> dict:
    """Serialize a MethodBase subclass's MethodDefinition to a JSON-compatible schema dict."""
    try:
        method_def = method_class.MethodDefinition
        dc_fields = dataclasses.fields(method_def)
        display_name = next(
            (f.default for f in dc_fields if f.name == 'name' and f.default is not dataclasses.MISSING),
            method_class.__name__,
        )
        field_names = [f.name for f in dc_fields if f.name != 'name']
        properties: dict = {}
        for f in dc_fields:
            if f.name == 'name':
                continue
            type_name = _resolve_json_type(f.type)
            prop: dict = {'type': type_name}
            if f.default is not dataclasses.MISSING:
                try:
                    prop['default'] = f.default
                except Exception:
                    pass
            elif f.default_factory is not dataclasses.MISSING:
                try:
                    val = f.default_factory()
                    prop['default'] = val.model_dump() if hasattr(val, 'model_dump') else val
                except Exception:
                    pass
            properties[f.name] = prop
        return {
            'fields': field_names,
            'display': True,
            'display_name': display_name,
            'method_type': method_type,
            'origin': 'device',
            'schema': {'type': 'object', 'properties': properties},
        }
    except Exception:
        logger.debug("Could not build schema for %s", method_class.__name__, exc_info=True)
        return {'fields': [], 'display': False, 'display_name': method_class.__name__, 'method_type': method_type, 'origin': 'device', 'schema': {}}


@runtime_checkable
class BrokerAssembly(Protocol):
    """Structural interface required by DeviceBrokerWorker.

    Satisfied by any AutocontrolPlugin subclass (single-channel, channels=[self])
    or any multi-channel assembly that sets self.channels = [...] in __init__.
    The channel count here must match the multichannel/n_channels declaration in
    lh_manager — autocontrol enforces this at dispatch time, and the channels list
    length enforces it again at the device side.
    """
    database_path: pathlib.Path | None
    channels: list          # elements must have method_callbacks: list and run_method()
    layout_callbacks: list  # zero-arg async callables; fired by trigger_layout_update()


class DeviceBrokerWorker:
    """Broker consumer/publisher for a single lh_devices service instance.

    Wire into an app by calling await worker.start() after device initialization.
    The worker registers _completion_callback on every channel's method_callbacks
    so that broker events fire automatically on method completion.
    """

    def __init__(
        self,
        device_id: str,
        assembly: BrokerAssembly,
        local_port: int,
        display_name: str = '',
        device_type: str = '',
        num_channels: Optional[int] = None,
        allow_sample_mixing: bool = True,
    ) -> None:
        self.device_id = device_id
        self.assembly = assembly
        self.local_port = local_port
        self.display_name = display_name
        self.device_type = device_type
        self._num_channels = num_channels  # resolved to len(channels) in start() if None
        self.allow_sample_mixing = allow_sample_mixing

        # Maps task_id → inbound Envelope so the completion callback can build
        # the correct outbound envelope (sample_id, assigned_channel, policy).
        self._pending: dict[str, Envelope] = {}
        # Method schemas built in start() and included in device.registered.
        self._methods_schema: dict[str, dict] = {}
        # Events used to signal GSIOC trigger subscription tasks when methods finish.
        self._method_done_events: dict[str, asyncio.Event] = {}

        self._exchange: Optional[aio_pika.abc.AbstractExchange] = None
        # Stored in start() so composition transfer and GSIOC subscriptions can declare temporary queues.
        self._amqp_channel: Optional[aio_pika.abc.AbstractChannel] = None
        # Filled in during start() so BrokerWasteInterface can publish.
        self.waste_interface: Optional['BrokerWasteInterface'] = None
        # Active dead-volume relay tasks keyed by task_id; cancelled on method completion.
        self._relay_tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to RabbitMQ, subscribe to command queue, register callbacks."""
        connection = await get_connection()
        channel = await connection.channel()
        self._amqp_channel = channel
        await channel.set_qos(prefetch_count=1)
        await declare_topology(channel)

        self._exchange = await channel.get_exchange(INSTRUMENT_EXCHANGE)

        if self._num_channels is None:
            self._num_channels = len(self.assembly.channels)

        if self.waste_interface is not None:
            self.waste_interface._exchange = self._exchange

        # Build method schema from each channel's method_runner (deduplicated by name).
        seen: set = set()
        for ch in self.assembly.channels:
            mr = getattr(ch, 'method_runner', None)
            if mr is None:
                continue
            for method_name, method_instance in mr.methods.items():
                if method_name not in seen:
                    seen.add(method_name)
                    method_type = mr.method_types.get(method_name, 'none')
                    self._methods_schema[method_name] = _schema_from_method_class(
                        type(method_instance), method_type=method_type
                    )

        cmd_queue = await declare_node_queue(channel, self.device_id, INSTRUMENT_EXCHANGE)

        for ch in self.assembly.channels:
            ch.method_callbacks.append(self._completion_callback)

        self.assembly.layout_callbacks.append(self._emit_layout_updated)

        # Subscribe to re-announce requests so lh_manager can trigger re-registration.
        announce_queue = await channel.declare_queue(
            f"{self.device_id}.announce_request",
            durable=False,
            auto_delete=True,
        )
        await announce_queue.bind(self._exchange, routing_key=DEVICE_ANNOUNCE_REQUEST)
        asyncio.create_task(consume(announce_queue, self._on_announce_request))

        await self._emit_device_registered()
        asyncio.create_task(consume(cmd_queue, self._on_command))
        logger.info("DeviceBrokerWorker [%s] running on port %d.", self.device_id, self.local_port)

    # ------------------------------------------------------------------
    # Inter-device composition transfer (MethodGroup coordination)
    # ------------------------------------------------------------------

    async def _publish_composition_transfer(
        self, task_id: str, envelope: Envelope, resolved_composition: dict
    ) -> None:
        """Publish composition.transfer.<task_id> after a task that produces a composition.

        Peer devices in the same MethodGroup subscribe to this key before running
        their own method (via await_composition_transfer flag in the task payload).
        """
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

    async def _feed_incoming_composition(
        self,
        method_instance: 'MethodBasewithCompositionReceive',
        task_id: str,
        subscription_ready: asyncio.Event,
    ) -> None:
        """Subscribe to composition.transfer.<task_id> and feed into the method's queue.

        Sets subscription_ready once the exclusive queue is bound, so the caller
        can proceed to call run_method() knowing no messages will be missed even
        if the source device publishes immediately.
        """
        if self._amqp_channel is None:
            logger.error("[%s] AMQP channel not available for composition transfer.", self.device_id)
            subscription_ready.set()
            return

        rk = composition_transfer_key(task_id)
        queue = await self._amqp_channel.declare_queue(exclusive=True, auto_delete=True)
        await queue.bind(self._exchange, routing_key=rk)
        subscription_ready.set()

        result: Optional[dict] = None
        received = asyncio.Event()

        async def _handler(msg: aio_pika.abc.AbstractIncomingMessage) -> None:
            nonlocal result
            async with msg.process():
                ev = Envelope.model_validate_json(msg.body)
                result = (ev.payload or {}).get("resolved_composition")
            received.set()

        consumer_tag = await queue.consume(_handler)
        try:
            await received.wait()
        finally:
            try:
                await queue.cancel(consumer_tag)
            except Exception:
                pass
        await method_instance._incoming_composition.put(result)

    # ------------------------------------------------------------------
    # GSIOC broker synchronisation (IS side)
    # ------------------------------------------------------------------

    async def _relay_dead_volume(self, method: 'MethodBasewithBrokerTrigger', task_id: str) -> None:
        """Relay every dead volume value the method puts in its queue to the broker.

        Loops until cancelled (by _completion_callback when the method finishes).
        Most methods put exactly one value (the physical dead volume in µL).
        DirectInjectBubbleSensor puts repeated 0/1 sensor readings during
        air-gap traversal, so all values must be forwarded — not just the first.

        gilson_lh subscribes to gsioc.dead_volume.<task_id> and returns each
        value in response to Trilution's 'V' GSIOC query.
        """
        rk = gsioc_dead_volume_key(task_id)
        try:
            while True:
                dead_volume = await method.dead_volume.get()
                if self._exchange is None:
                    continue
                msg = build(
                    device_id=self.device_id,
                    routing_key=rk,
                    payload={"dead_volume": dead_volume},
                )
                await publish(self._exchange, rk, msg)
                logger.debug("[%s] dead_volume relayed for task %s: %s", self.device_id, task_id, dead_volume)
        except asyncio.CancelledError:
            return

    async def _await_gsioc_triggers(
        self, method: 'MethodBasewithBrokerTrigger', task_id: str, done_event: asyncio.Event
    ) -> None:
        """Subscribe to gsioc.trigger.<task_id> and call activate_trigger() on each message.

        Loops until done_event is set (method completes). A single method may need
        multiple triggers (e.g. LoadLoopBubbleSensor calls wait_for_trigger() three times).
        """
        if self._amqp_channel is None:
            return
        rk = gsioc_trigger_key(task_id)
        queue = await self._amqp_channel.declare_queue(exclusive=True, auto_delete=True)
        await queue.bind(self._exchange, routing_key=rk)

        async def _on_trigger(msg: aio_pika.abc.AbstractIncomingMessage) -> None:
            async with msg.process():
                method.activate_trigger()
                logger.debug("[%s] gsioc trigger activated for task %s.", self.device_id, task_id)

        consumer_tag = await queue.consume(_on_trigger)
        try:
            await done_event.wait()
        finally:
            try:
                await queue.cancel(consumer_tag)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Service discovery
    # ------------------------------------------------------------------

    async def _emit_device_registered(self) -> None:
        if self._exchange is None:
            return
        nc = self._num_channels
        msg = build(
            device_id=self.device_id,
            routing_key=DEVICE_REGISTERED,
            payload={
                "device_id": self.device_id,
                "display_name": self.display_name,
                "device_type": self.device_type,
                "num_channels": nc,
                "allow_sample_mixing": self.allow_sample_mixing,
                "address": f"http://localhost:{self.local_port}",
                "methods": self._methods_schema,
            },
        )
        await publish(self._exchange, DEVICE_REGISTERED, msg)
        logger.info("[%s] device.registered published (%d channels, %d methods).",
                    self.device_id, nc, len(self._methods_schema))

    async def _on_announce_request(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        """Re-emit device.registered when lh_manager requests a fresh announcement."""
        logger.debug("[%s] announce_request received — re-publishing device.registered.", self.device_id)
        await self._emit_device_registered()

    # ------------------------------------------------------------------
    # Inbound: command.<device_id>.submit_task
    # ------------------------------------------------------------------

    async def _on_command(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        rk = message.routing_key or ""
        verb = rk.split(".")[-1]

        if verb == "cancel_task":
            task_id = str(envelope.task_id)
            if task_id in self._pending:
                pending_env = self._pending.pop(task_id)
                logger.info("[%s] cancel_task for in-flight task %s — publishing task.failed.", self.device_id, task_id)
                await self._emit(TASK_FAILED, pending_env, {"error": "Task cancelled by operator"})
                # Stop the GSIOC trigger subscription loop if running.
                done_event = self._method_done_events.pop(task_id, None)
                if done_event is not None:
                    done_event.set()
                # Cancel the physical method asyncio task so on_cancel() releases devices.
                channel_index = pending_env.assigned_channel if pending_env.assigned_channel is not None else 0
                if channel_index < len(self.assembly.channels):
                    self.assembly.channels[channel_index].method_runner.cancel_methods_by_id(task_id)
            return

        if verb != "submit_task":
            logger.warning("[%s] Unknown command verb '%s'", self.device_id, verb)
            return

        task_id = str(envelope.task_id)
        payload = envelope.payload

        # Idempotency: skip hardware if we already ran this task.
        if self.assembly.database_path is not None:
            db_path = self.assembly.database_path
            existing = await asyncio.to_thread(
                lambda: HistoryDB(db_path).search_id(task_id)
            )
            if existing is not None:
                logger.info("[%s] task %s already complete — re-publishing.", self.device_id, task_id)
                await self._publish_completed_from_result(existing, envelope)
                return

        # INIT tasks: hardware is already initialized at startup; just ack.
        if payload.get("task_type") == "init":
            await self._emit_init_completed(envelope)
            return

        channel_index = envelope.assigned_channel
        if channel_index is None:
            channel_index = payload.get("channel")
        if channel_index is None:
            channel_index = 0

        try:
            method_list = payload["method_data"]["method_list"]
            method_name: str = method_list[0]["method_name"]
            method_data: dict = method_list[0].get("method_data", {})
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("[%s] Malformed submit_task payload: %s", self.device_id, exc)
            raise

        if channel_index >= len(self.assembly.channels):
            logger.error("[%s] channel %d does not exist", self.device_id, channel_index)
            raise ValueError(f"channel {channel_index} does not exist")

        self._pending[task_id] = envelope

        await self._emit(TASK_ACCEPTED, envelope, {})
        await self._emit(CHANNEL_STATUS_CHANGED, envelope, {"status": "busy", "channel": channel_index})

        ch = self.assembly.channels[channel_index]
        method_instance = ch.method_runner.methods.get(method_name)

        # Start GSIOC broker synchronisation background tasks if needed.
        # dead_volume relay fires once; trigger subscription loops until method completes.
        if isinstance(method_instance, MethodBasewithBrokerTrigger):
            done_event = asyncio.Event()
            self._method_done_events[task_id] = done_event
            self._relay_tasks[task_id] = asyncio.create_task(
                self._relay_dead_volume(method_instance, task_id)
            )
            asyncio.create_task(self._await_gsioc_triggers(method_instance, task_id, done_event))

        # If the method calls receive_composition(), subscribe to composition.transfer
        # before starting the method so no message from the source device is missed.
        # Reset the queue so a leftover from a previous failed run is not returned.
        if isinstance(method_instance, MethodBasewithCompositionReceive):
            method_instance._incoming_composition = asyncio.Queue(maxsize=1)
            subscription_ready = asyncio.Event()
            asyncio.create_task(self._feed_incoming_composition(method_instance, task_id, subscription_ready))
            await subscription_ready.wait()

        ch.run_method(method_name, method_data, id=task_id)

    # ------------------------------------------------------------------
    # Outbound: completion callback (registered on each channel)
    # ------------------------------------------------------------------

    async def _completion_callback(self, result: MethodResult) -> None:
        if result.id is None:
            return

        # Signal any active GSIOC trigger subscription to stop.
        done_event = self._method_done_events.pop(result.id, None)
        if done_event is not None:
            done_event.set()

        # Cancel the dead-volume relay loop now that the method is finished.
        relay_task = self._relay_tasks.pop(result.id, None)
        if relay_task is not None:
            relay_task.cancel()

        envelope = self._pending.pop(result.id, None)
        if envelope is None:
            return

        channel_index = envelope.assigned_channel if envelope.assigned_channel is not None else 0
        await self._emit(CHANNEL_STATUS_CHANGED, envelope, {"status": "idle", "channel": channel_index})

        if result.result and result.result.get("error"):
            await self._emit(TASK_FAILED, envelope, {
                "error": result.result["error"],
                "method_name": result.method_name,
            })
        else:
            # If the method called emit_composition(), relay it to MethodGroup peers
            # before publishing task.completed so peers receive it while still waiting.
            ch = self.assembly.channels[channel_index]
            method_instance = ch.method_runner.methods.get(result.method_name)
            if isinstance(method_instance, MethodBasewithCompositionRelay):
                pending = method_instance._pending_composition_transfer
                if pending is not None:
                    await self._publish_composition_transfer(str(result.id), envelope, pending)
                    method_instance._pending_composition_transfer = None
            await self._publish_completed_from_result(result, envelope)

        await self._emit_layout_updated()

    async def _publish_completed_from_result(self, result: MethodResult, envelope: Envelope) -> None:
        if self._exchange is None:
            return
        retrieval_uri = f"http://localhost:{self.local_port}/GetTaskData?task_id={result.id}"
        msg = build(
            device_id=self.device_id,
            routing_key=TASK_COMPLETED,
            task_id=envelope.task_id,
            sample_id=envelope.sample_id,
            assigned_channel=envelope.assigned_channel,
            execution_policy=envelope.execution_policy or "infrastructure",
            payload={"method_name": result.method_name},
            data_reference=DataReference(retrieval_uri=retrieval_uri),
        )
        await publish(self._exchange, TASK_COMPLETED, msg)

    # ------------------------------------------------------------------
    # Internal publish helper
    # ------------------------------------------------------------------

    async def _emit_init_completed(self, envelope: Envelope) -> None:
        if self._exchange is None:
            return
        msg = build(
            device_id=self.device_id,
            routing_key=TASK_COMPLETED,
            task_id=envelope.task_id,
            sample_id=envelope.sample_id,
            assigned_channel=envelope.assigned_channel,
            execution_policy=envelope.execution_policy or "infrastructure",
            payload={"task_type": "init"},
        )
        await publish(self._exchange, TASK_COMPLETED, msg)

    async def _emit_layout_updated(self) -> None:
        if self._exchange is None:
            return
        base_url = f"http://localhost:{self.local_port}"
        msg = build(
            device_id=self.device_id,
            routing_key=LAYOUT_UPDATED,
            payload={"device_name": self.device_id, "retrieval_uri": base_url},
        )
        await publish(self._exchange, LAYOUT_UPDATED, msg)

    async def _emit(self, routing_key: str, envelope: Envelope, extra: dict) -> None:
        if self._exchange is None:
            return
        msg = build(
            device_id=self.device_id,
            routing_key=routing_key,
            task_id=envelope.task_id,
            sample_id=envelope.sample_id,
            assigned_channel=envelope.assigned_channel,
            execution_policy=envelope.execution_policy or "infrastructure",
            payload=extra,
        )
        await publish(self._exchange, routing_key, msg)


# ---------------------------------------------------------------------------
# BrokerWasteInterface
# ---------------------------------------------------------------------------

class BrokerWasteInterface(WasteInterfaceBase):
    """Replaces RoadmapWasteInterface: publishes waste.generated instead of HTTP POST.

    Usage in app.py:
        waste_interface = BrokerWasteInterface()
        broker_worker = DeviceBrokerWorker(DEVICE_ID, assembly, local_port=5003)
        broker_worker.waste_interface = waste_interface
        # worker.start() wires the exchange into waste_interface automatically.
    """

    def __init__(self) -> None:
        super().__init__()
        self._exchange: Optional[aio_pika.abc.AbstractExchange] = None

    async def submit(self, waste) -> WasteResponse:
        if self._exchange is None:
            logger.warning("BrokerWasteInterface: exchange not set, dropping waste event.")
            return WasteResponse(success=False, response={"error": "broker not connected"})

        try:
            payload = waste.model_dump()
        except AttributeError:
            payload = {}

        envelope = build(
            device_id="lh_devices",
            routing_key=WASTE_GENERATED,
            payload=payload,
        )
        await publish(self._exchange, WASTE_GENERATED, envelope)
        return WasteResponse(success=True)
