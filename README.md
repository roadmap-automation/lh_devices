# Liquid handling devices (`lh_devices`)
Control and visualization framework for automated liquid handling applications

## Introduction and core concepts
`lh_devices` is a set of libraries for interacting with physical **devices** or **components** commonly used in liquid handling applications, such as valve positioners, syringe pumps and flow cells. The core organizational concept is that **assemblies** of physical devices are connected via fluid paths into a **network**, where each **node** of the network corresponds to a fluid **port** on a physical device.

**Methods** involving multiple devices or assemblies can be used to operate on the devices. These automatically generate logs using the `logging` module, return any results from the method (*e.g.* a measurement), and can save the results to a SQLite3 database.

Both devices and assemblies have a built in recursive web application builder for visualization and rudimentary control, with updates performed over a SocketIO connection.

## Structure
Base (application-independent) classes are contained in the modules in the base `lh_devices` directory. Subdirectories contain application-specific code:

* `gilson`: communications with Gilson devices, specifically through the GSIOC protocol
* `hamilton`: communications specifically with Hamilton OEM syringe pumps and valves
* `injection`: the ROADMAP project multichannel injection system. Requires an active setup to run.
* `qcmd`: the ROADMAP project QCMD measurement interface.

## Getting started

Via process-compose (recommended):

```bash
process-compose up
```

Or individually:

```bash
python -m lh_devices.injection.app       # multichannel injection system (port 5003)
python -m lh_devices.injection.simulated_app  # simulated (no hardware required)
python -m lh_devices.qcmd.app            # QCMD array (port 5005)
python -m lh_devices.gilson_lh.app      # Gilson liquid handler (port 5009)
```

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `AMQP_URL` | `amqp://roadbot:roadbot_dev@localhost/` | RabbitMQ connection |
| `ROADMAP_SERIAL` | — | Serial port for GSIOC (gilson_lh only, e.g. `COM13`) |

---

## Role in the System

`lh_devices` hosts three physical instrument controllers, all on `exchange.instrument` as
**Tier 2 device controllers**:

| Application | Device ID | Channels | Error domain |
|---|---|---|---|
| Injection system | `injection` | 3 | `uncertain` (TRANSFER tasks) |
| QCMD array | `qcmd` | 3 | `repeatable` (MEASURE tasks) |
| Gilson liquid handler | `lh` | 1 | `irreversible` (PREPARE tasks) |

---

## Physical Constraints

### Injection System (`injection/`, port 5003)

**3-channel sticky affinity.** `TaskData.channel` routes tasks to a specific physical
channel (syringe pump + loop valve assembly). Channel assignment is set by lh_manager and
must never change mid-workflow.

**Hardware locking at method level.** Each method calls `reserve_all()` at start and
`release_all()` at completion. The distribution valve is shared across channels and locked
per method invocation.

**TRANSFER → Uncertain error domain.** `LoadLoop` and `DirectInject` physically move
sample. A failure leaves sample location unknown. The system must not auto-retry. Requires
a manual Clear Fault via the operator UI.

**Bubble sensors enforce transfer verification.** `LoadLoopBubbleSensor` and
`DirectInjectBubbleSensor` poll SMD sensors on the Hamilton device to verify liquid
presence before completing a transfer.

**Composition relay.** After injection, the injection system publishes
`composition.transfer.<task_id>` (or emits the composition via `MethodBasewithCompositionRelay`)
so that QCMD and the reflectometer can update their flow-cell well records.

### QCMD Measurement System (`qcmd/`, port 5005)

**3 independent measurement channels.** Channels do not share hardware; concurrent
operation across channels is physically safe.

**MEASURE → Repeatable error domain.** If the openQCM HTTP API disconnects mid-measurement,
`throw_error(critical=False)` pauses the method without aborting. Auto-retry is safe.

**Large data — Claim Check required.** Raw QCMD time-series data is saved to
`history/qcmd.db`. The broker carries only `retrieval_uri` pointing to
`GET /GetTaskData?task_id=<id>`. Camera images from `QCMDRecordTagwithCamera` are also
stored in SQLite; they must never appear in broker message bodies.

**Composition tracking.** `QCMDAcceptTransfer` updates `Well.composition` when a new
sample arrives, using the composition delivered via `MethodBasewithCompositionReceive`.

### Gilson Liquid Handler (`gilson_lh/`, port 5009)

**PREPARE → Irreversible error domain.** A failure during reagent mixing or aspiration
has ruined the sample. Immediately abort and release downstream locks.

**Well reservation system.** `reservation_store` tracks UUID-keyed well allocations
minted by SubProtocol `$alloc` fields. Allocations are released when
`protocol_studio.subprotocol.completed` is received. Prevents well conflicts between
concurrent trials.

**GSIOC serial protocol ownership.** gilson_lh owns the COM port connection to Trilution.
The injection system no longer holds a GSIOC connection. gilson_lh translates GSIOC
`'T'` bytes into `gsioc.trigger.<task_id>` broker messages and responds to Trilution's
`'V'` dead-volume queries using values received via `gsioc.dead_volume.<task_id>`.

---

## Broker Interface (shared across all three applications)

All three applications use `DeviceBrokerWorker` (or `GilsonLHBrokerWorker`), which
provides a standard interface:

### Subscribes to
| Routing key | Queue | Publisher |
|---|---|---|
| `command.<device_id>.submit_task` | `command.<device_id>` (durable) | autocontrol |
| `command.<device_id>.cancel_task` | same queue | autocontrol |
| `device.announce_request` | transient | lh_manager |
| `protocol_studio.subprotocol.completed` | transient | lh_manager (gilson_lh only, for well reservation cleanup) |

### Publishes
| Routing key | When |
|---|---|
| `device.registered` | On startup (announces device_id, num_channels, method schemas) |
| `task.accepted` | After idempotency check passes |
| `channel.status_changed` BUSY | Before `run_method()` |
| `task.completed` | After method completes; `data_reference.retrieval_uri` set |
| `task.failed` | After method raises an error |
| `channel.status_changed` IDLE | After completion |
| `layout.updated` | After well/rack layout changes |
| `waste.generated` | After injection system method (replaces HTTP call to lh_manager) |
| `composition.transfer.<task_id>` | gilson_lh after formulation; injection system after inject |
| `gsioc.trigger.<task_id>` | gilson_lh when Trilution sends GSIOC `'T'` |
| `gsioc.dead_volume.<task_id>` | Injection system at method start |

---

## REST Endpoints (remain after broker refactor)

| Endpoint | Port | Purpose |
|---|---|---|
| `GET /GetTaskData?task_id=<id>` | 5003 / 5005 / 5009 | Claim Check retrieval for completed task results |
| SocketIO web UI | each port | Local device monitoring and manual control |
