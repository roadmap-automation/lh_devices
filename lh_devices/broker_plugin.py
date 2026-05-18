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
import logging
import pathlib
from typing import Optional, Protocol, runtime_checkable

import aio_pika

from roadmap_broker_client.connection import get_connection
from roadmap_broker_client.consumer import consume
from roadmap_broker_client.envelope import DataReference, Envelope, build
from roadmap_broker_client.publisher import publish
from roadmap_broker_client.topology import declare_node_queue, declare_topology
from roadmap_broker_client.topics import (
    CHANNEL_STATUS_CHANGED,
    INSTRUMENT_EXCHANGE,
    LAYOUT_UPDATED,
    TASK_ACCEPTED,
    TASK_COMPLETED,
    TASK_FAILED,
    WASTE_GENERATED,
)

from .history import HistoryDB
from .methods import MethodResult
from .waste import WasteInterfaceBase, WasteResponse

logger = logging.getLogger(__name__)


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

    def __init__(self, device_id: str, assembly: BrokerAssembly, local_port: int) -> None:
        self.device_id = device_id
        self.assembly = assembly
        self.local_port = local_port

        # Maps task_id → inbound Envelope so the completion callback can build
        # the correct outbound envelope (sample_id, assigned_channel, policy).
        self._pending: dict[str, Envelope] = {}

        self._exchange: Optional[aio_pika.abc.AbstractExchange] = None
        # Filled in during start() so BrokerWasteInterface can publish.
        self.waste_interface: Optional['BrokerWasteInterface'] = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect to RabbitMQ, subscribe to command queue, register callbacks."""
        connection = await get_connection()
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        await declare_topology(channel)

        self._exchange = await channel.get_exchange(INSTRUMENT_EXCHANGE)

        if self.waste_interface is not None:
            self.waste_interface._exchange = self._exchange

        cmd_queue = await declare_node_queue(channel, self.device_id, INSTRUMENT_EXCHANGE)

        for ch in self.assembly.channels:
            ch.method_callbacks.append(self._completion_callback)

        self.assembly.layout_callbacks.append(self._emit_layout_updated)

        asyncio.create_task(consume(cmd_queue, self._on_command))
        logger.info("DeviceBrokerWorker [%s] running on port %d.", self.device_id, self.local_port)

    # ------------------------------------------------------------------
    # Inbound: command.<device_id>.submit_task
    # ------------------------------------------------------------------

    async def _on_command(
        self, envelope: Envelope, message: aio_pika.abc.AbstractIncomingMessage
    ) -> None:
        rk = message.routing_key or ""
        verb = rk.split(".")[-1]

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
            channel_index = payload.get("channel", 0)

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

        self.assembly.channels[channel_index].run_method(method_name, method_data, id=task_id)

    # ------------------------------------------------------------------
    # Outbound: completion callback (registered on each channel)
    # ------------------------------------------------------------------

    async def _completion_callback(self, result: MethodResult) -> None:
        if result.id is None:
            return

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
