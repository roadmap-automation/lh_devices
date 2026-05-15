"""
Broker integration test for lh_devices injection system.

Starts the simulated injection app in-process, then uses a BrokerTestClient
to submit tasks and assert on the events that come back.

Tests:
  - RoadmapChannelInit   → task.accepted + task.completed
  - RoadmapChannelSleep  → task.accepted + task.completed (after sleep)
  - Idempotency          → re-submitting same task_id → task.completed, no re-execution

Requires RabbitMQ running at localhost:5672:
  docker-compose up -d rabbitmq

Usage:
  cd lh_devices
  python tests/broker_test.py
"""

import asyncio
import logging
import sys
import uuid
from pathlib import Path

import aio_pika

sys.path.insert(0, str(Path(__file__).parent.parent))

from roadmap_broker_client.connection import get_connection
from roadmap_broker_client.envelope import Envelope, build
from roadmap_broker_client.publisher import publish
from roadmap_broker_client.topology import declare_topology
from roadmap_broker_client.topics import (
    CHANNEL_STATUS_CHANGED,
    INSTRUMENT_EXCHANGE,
    TASK_ACCEPTED,
    TASK_COMPLETED,
    TASK_FAILED,
    command_key,
)

from lh_devices.injection.simulated_app import run_injection_system

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

EVENT_TIMEOUT = 10.0
APP_START_WAIT = 3.0

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results: list[tuple[str, bool]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail and not condition else ""
    print(f"  [{status}] {name}{suffix}")
    _results.append((name, condition))


# ---------------------------------------------------------------------------
# BrokerTestClient
# ---------------------------------------------------------------------------

class BrokerTestClient:
    def __init__(self) -> None:
        self._exchange: aio_pika.abc.AbstractExchange | None = None
        self._received: list[tuple[str, Envelope]] = []
        self._condition = asyncio.Condition()

    async def setup(self, channel: aio_pika.abc.AbstractChannel) -> None:
        self._exchange = await channel.get_exchange(INSTRUMENT_EXCHANGE)
        q = await channel.declare_queue(
            "test.lhdevices.events", durable=False, auto_delete=True
        )
        await q.bind(self._exchange, routing_key="task.#")
        await q.bind(self._exchange, routing_key="channel.#")
        asyncio.create_task(self._consume(q))

    async def _consume(self, queue: aio_pika.abc.AbstractQueue) -> None:
        async with queue.iterator() as msgs:
            async for msg in msgs:
                try:
                    env = Envelope.model_validate_json(msg.body)
                    async with self._condition:
                        self._received.append((msg.routing_key or "", env))
                        self._condition.notify_all()
                    await msg.ack()
                except Exception:
                    logging.exception("[TestClient] error processing message")
                    await msg.nack(requeue=False)

    async def wait_for(
        self,
        routing_key: str,
        task_id: uuid.UUID | None = None,
        timeout: float = EVENT_TIMEOUT,
    ) -> tuple[str, Envelope] | tuple[None, None]:
        deadline = asyncio.get_event_loop().time() + timeout

        def matches(rk: str, env: Envelope) -> bool:
            key_ok = rk == routing_key or rk.startswith(routing_key.rstrip("#").rstrip("."))
            id_ok = task_id is None or env.task_id == task_id
            return key_ok and id_ok

        async with self._condition:
            while True:
                for rk, env in self._received:
                    if matches(rk, env):
                        return rk, env
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return None, None
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return None, None

    def clear(self) -> None:
        self._received.clear()

    async def submit(
        self,
        device_id: str,
        method_name: str,
        method_data: dict | None = None,
        channel: int = 0,
        task_id: uuid.UUID | None = None,
        sample_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        task_id = task_id or uuid.uuid4()
        rk = command_key(device_id, "submit_task")
        payload = {
            "method_data": {"method_list": [
                {"method_name": method_name, "method_data": method_data or {}}
            ]},
            "channel": channel,
        }
        env = build(
            device_id="test_client",
            routing_key=rk,
            task_id=task_id,
            sample_id=sample_id,
            assigned_channel=channel,
            execution_policy="infrastructure",
            payload=payload,
        )
        await publish(self._exchange, rk, env)
        return task_id


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

async def test_init(client: BrokerTestClient, device_id: str) -> None:
    print("\n[test_init] RoadmapChannelInit → task.accepted + task.completed")
    client.clear()

    task_id = await client.submit(device_id, "RoadmapChannelInit", channel=0)

    _, env = await client.wait_for(TASK_ACCEPTED, task_id=task_id)
    check("task.accepted received", env is not None)

    _, env = await client.wait_for(TASK_COMPLETED, task_id=task_id)
    check("task.completed received", env is not None)
    check("device_id correct", env is not None and env.device_id == device_id)


async def test_sleep(client: BrokerTestClient, device_id: str) -> None:
    print("\n[test_sleep] RoadmapChannelSleep(0.05 min) → accepted + completed + Claim Check")
    client.clear()

    task_id = await client.submit(
        device_id, "RoadmapChannelSleep",
        method_data={"sleep_time": 0.05},
        channel=0,
    )

    _, env = await client.wait_for(TASK_ACCEPTED, task_id=task_id)
    check("task.accepted received", env is not None)

    # Sleep method takes ~3s; give it headroom
    _, env = await client.wait_for(TASK_COMPLETED, task_id=task_id, timeout=15.0)
    check("task.completed received after sleep", env is not None)
    check("data_reference present (Claim Check)",
          env is not None and env.data_reference is not None)
    check("retrieval_uri non-empty",
          env is not None
          and env.data_reference is not None
          and bool(env.data_reference.retrieval_uri))


async def test_channel_status(client: BrokerTestClient, device_id: str) -> None:
    print("\n[test_channel_status] RoadmapChannelSleep → channel BUSY then IDLE")
    client.clear()

    task_id = await client.submit(
        device_id, "RoadmapChannelSleep",
        method_data={"sleep_time": 0.05},
        channel=0,
    )

    _, env = await client.wait_for(CHANNEL_STATUS_CHANGED, task_id=task_id)
    check("channel.status_changed (busy) received",
          env is not None and env.payload.get("status") == "busy")

    _, env = await client.wait_for(TASK_COMPLETED, task_id=task_id, timeout=15.0)
    check("task completed before checking idle status", env is not None)

    # Find the idle event in received list
    idle_events = [
        e for rk, e in client._received
        if rk == CHANNEL_STATUS_CHANGED
        and e.task_id == task_id
        and e.payload.get("status") == "idle"
    ]
    check("channel.status_changed (idle) received", len(idle_events) > 0)


async def test_idempotency(client: BrokerTestClient, device_id: str) -> None:
    print("\n[test_idempotency] Re-submit same task_id → completed without re-executing")
    client.clear()

    task_id = await client.submit(device_id, "RoadmapChannelInit", channel=0)
    _, env = await client.wait_for(TASK_COMPLETED, task_id=task_id)
    check("first submission completes", env is not None)

    client.clear()
    # Re-submit the exact same task_id
    await client.submit(device_id, "RoadmapChannelInit", channel=0, task_id=task_id)
    _, env = await client.wait_for(TASK_COMPLETED, task_id=task_id)
    check("idempotent re-submit returns task.completed", env is not None)
    # Should NOT get task.accepted (no re-execution)
    _, accepted = await client.wait_for(TASK_ACCEPTED, task_id=task_id, timeout=1.0)
    check("no task.accepted on idempotent re-submit", accepted is None)


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

async def run_harness() -> None:
    device_id = "injection"

    print("Starting simulated injection system ...")
    asyncio.create_task(run_injection_system())
    print(f"Waiting {APP_START_WAIT}s for app and broker worker to connect ...")
    await asyncio.sleep(APP_START_WAIT)

    connection = await get_connection()
    async with connection:
        ch = await connection.channel()
        await ch.set_qos(prefetch_count=10)
        await declare_topology(ch)

        client = BrokerTestClient()
        await client.setup(ch)
        await asyncio.sleep(0.3)

        print("\n" + "=" * 60)
        print("Running lh_devices broker tests")
        print("=" * 60)

        try:
            await test_init(client, device_id)
            await test_sleep(client, device_id)
            await test_channel_status(client, device_id)
            await test_idempotency(client, device_id)
        except Exception:
            logging.exception("Unhandled exception in test run")

    passed = sum(1 for _, ok in _results if ok)
    total = len(_results)
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed")
    if passed < total:
        print("Failed:")
        for name, ok in _results:
            if not ok:
                print(f"  - {name}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s  %(name)-20s  %(levelname)s  %(message)s",
    )
    asyncio.run(run_harness())
