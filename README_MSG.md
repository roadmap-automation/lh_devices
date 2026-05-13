# lh_devices — Refactor Context for AI Assistant

## 1. Core Responsibility
`lh_devices` is an async (asyncio/aiohttp) device control framework hosting two physical instrument applications: a **3-channel multichannel injection system** (Hamilton syringe pumps + valves, GSIOC-synchronized with the Gilson LH) and a **3-channel QCMD measurement system** (quartz crystal microbalance via openQCM HTTP API), both exposed to the Autocontrol scheduler via a shared REST interface.

---

## 2. Physical Constraints

### Injection System (`injection/`, runs at `:5003`)
- **3 parallel channels with sticky channel affinity.** The `TaskData.channel` field routes tasks to a specific physical channel (syringe pump + loop valve assembly). Never reassign a channel mid-workflow.
- **Device-level hardware locking is enforced at runtime.** Each method calls `reserve_all()` at start and `release_all()` at completion, preventing two methods from using the same physical device simultaneously. The distribution valve is shared across channels and is locked per method invocation.
- **GSIOC synchronization with the Gilson LH is blocking.** `LoadLoop`, `DirectInject`, and all LH-coordinated methods (`MethodBaseDeadVolume` subclasses) use an async trigger/wait handshake over GSIOC serial. The method suspends at `await self.wait_for_trigger()` until the LH sends a serial trigger. This timing is safety-critical. The refactor must preserve this synchronous handshake — broker messages cannot replace GSIOC; they only replace the Autocontrol HTTP task submission.
- **TRANSFER / INJECT → Uncertain error domain.** `LoadLoop` and `DirectInject` physically move sample. A failure (syringe pump error, GSIOC timeout, bubble sensor detecting air mid-injection) leaves sample location unknown. The system **must not auto-retry**. `MethodException(retry=False)` blocks execution until a human operator clears the error via the web UI. This maps to the Uncertain error handling rule in `SYSTEM_REQUIREMENTS.md`.
- **Bubble sensors enforce transfer verification.** `LoadLoopBubbleSensor` and `DirectInjectBubbleSensor` poll SMD sensors on the Hamilton device to verify liquid presence before completing a transfer. If air is detected, the method throws a non-critical error and blocks.

### QCMD Measurement System (`qcmd/`, runs at `:5005`)
- **3 independent measurement channels.** Each channel has one QCMD instrument and one USB camera. Channels do not share hardware; concurrent operation across channels is physically safe.
- **MEASURE → Repeatable error domain.** If the openQCM HTTP API disconnects mid-measurement, `throw_error(critical=False)` pauses the method (does not abort) and waits for reconnection. Auto-retry is safe here.
- **Large data payloads — Claim Check required.** Raw QCMD time-series data is fetched from the openQCM API at method completion and saved to `history/qcmd.db` (SQLite). The broker must only carry `{task_id, retrieval_uri}`. Full data is retrieved via `GET /GetTaskData`.
- **Camera images — Claim Check required.** `QCMDRecordTagwithCamera` captures before/after images as base64 payloads. These are saved in the `MethodResult.result` dict in SQLite. Do not put image data in broker messages.
- **Flow cell composition tracking.** `QCMDAcceptTransfer` updates the layout's `Well.composition` for each QCMD channel when a new sample arrives. This layout state is shared with `lh_manager` via `LayoutPlugin`.

---

## 3. Deprecated REST Endpoints

### Autocontrol-facing (task interface — will become broker subscriber)
- `POST /SubmitTask` — receives `TaskData` from Autocontrol; extracts `method_name` + `method_data` + `channel`, dispatches to channel's `run_method()`
- `GET /GetStatus` — returns `{status: IDLE|BUSY, channel_status: [IDLE|BUSY, ...]}` polled by Autocontrol

### Task result retrieval (will become Claim Check lookup)
- `GET /GetTaskData` — retrieves stored `MethodResult` by task ID from SQLite (`history/injection_system.db` or `history/qcmd.db`)

### Layout sync (currently pulled by lh_manager — will become broker event)
- `GET /GetLayout` — returns `LHBedLayout` JSON (flow cell / sample loop compositions); called by `lh_manager` `LayoutPlugin` to sync bed layout
- `POST /UpdateLayout` — pushes layout update into device (used when lh_manager resolves well compositions after prep)

### Outbound call to lh_manager (will become broker publish)
- `POST http://localhost:5001/Waste/AddWaste/` — called by `RoadmapWasteInterface` after each method to submit waste volume/composition. This is an **outbound** call from lh_devices to lh_manager, not an inbound endpoint. Will become a broker publish event.

---

## 4. Future Telemetry Needs

### State change events (lightweight, publish on change)
- `Task.Accepted` — when `SubmitTask` is received and dispatched to method runner
- `Task.Completed` — method finished successfully; payload: `{task_id, retrieval_uri}` only (Claim Check — full `MethodResult` stays in SQLite)
- `Task.Failed` — `MethodException` raised; payload: `{task_id, error, retry: bool}`; the `retry` flag distinguishes Repeatable (retry=True) from Uncertain (retry=False) error domains
- `Channel.StatusChanged` — per-channel IDLE/BUSY transitions (driven by `reserve_all` / `release_all`)
- `Layout.Updated` — flow cell or sample loop composition changed (after `QCMDAcceptTransfer` or injection completion)
- `Waste.Generated` — waste volume and composition after each method (replaces outbound HTTP call to lh_manager)

### Large data payloads (Claim Check — do not carry in broker message body)
- **QCMD time-series data** — frequency vs. time measurement result from openQCM API; save to `history/qcmd.db`, publish only `retrieval_uri`
- **Camera images** — before/after microscopy images from `QCMDRecordTagwithCamera`; save to SQLite `result` JSON, publish only `retrieval_uri`
- **Full `MethodResult` logs** — per-method execution log with timestamps; save to `history/*.db`, reference by task UUID

### Reference / query endpoints (likely stay REST)
- `GET /GetTaskData` — task result lookup by ID (used by upstream after receiving `Task.Completed` event)
- SocketIO web UI — local device monitoring and manual control; not replaced by broker
