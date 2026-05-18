# lh_devices — Improvement TODO

## Context
Identified friction points for adding new devices (especially non-fluid ones like `nice_device`).
Priority order is listed under each item.

---

## 1. Method auto-registration via decorator (Priority 1)

Right now `method_runner.methods` must be populated manually — a common error source.
Add a `@register_method` decorator (or `__init_subclass__` metaclass scan) so `MethodPlugin`
auto-discovers `MethodBase` subclasses defined on the class:

```python
class MyDevice(MethodPlugin, DeviceBase):

    @register_method
    class Measure(MethodBase):
        @dataclass
        class MethodDefinition:
            name: str = 'Measure'
            duration: float = 10.0

        async def run(self, duration: float = 10.0):
            ...
```

---

## 2. `DeviceApp` scaffold class (Priority 2)

Every module has a hand-rolled `main()` doing: create device → `create_web_app()` →
`run_socket_app()` → `asyncio.Event().wait()` → cleanup. Collapse to:

```python
app = DeviceApp(device=CANDOR('localhost'), host='localhost', port=5056)
asyncio.run(app.run())
```

---

## 3. Move Autocontrol REST routes out of `MethodPlugin` (Priority 3)

`/SubmitTask`, `/GetStatus`, `/GetTaskData` are defined in `methods.py` but are
Autocontrol-protocol-specific. `MethodPlugin` should be a generic dispatcher;
`AutocontrolPlugin` (which already exists in `autocontrolplugin.py`) is the right
place for those routes. Also required by the broker refactor, which replaces those
endpoints with broker consumers.

---

## 4. Split into `lh_devices_core` / `lh_devices` / `roadmap_devices` (Priority 4)

The package conflates three concerns:

| Layer | Contents | Who needs it |
|---|---|---|
| **Core** | `device.py`, `methods.py`, `webview.py` | every instrument |
| **Fluid** | `connections.py`, `assemblies.py`, valves, syringes | liquid handling only |
| **Roadmap-specific** | `injection/`, `qcmd/`, `layout.py`, `waste.py` | this project only — imports `lh_manager` |

`nice_device` only needs the core but currently pulls in the whole package.
Separate `roadmap_devices` (or `lh_devices_roadmap`) so the core/fluid layers
can be used without `lh_manager` as a dependency. The monorepo layout stays
convenient for local dev; the separation is just a `pyproject.toml` split.

Do this as part of the broker refactor since the structure is already changing.

---

## 5. Kill the global `sio` singleton in `webview.py` (Priority 5)

`sio = socketio.AsyncServer(...)` at module level means all devices share one
socketio server and tests cannot create isolated instances. Pass `sio` as a
constructor parameter or use a factory. Defer until tests become a priority.

---

## 6. Replace `setup.py` with `pyproject.toml` + auto `find_packages`

Currently every sub-package must be listed manually in `setup.py`. With
`pyproject.toml` and `find_packages` new sub-packages are auto-discovered.
Also enables optional dependency groups: `[fluid]`, `[camera]`, `[hamilton]`, etc.

---

## 7. Add a minimal starter template

`nice_device` is the cleanest example of a non-fluid device. Add a
`lh_devices/starter/` directory (or a cookiecutter) with a documented ~50-line
stub showing exactly what to override:
- `initialize_device`
- `update_status`
- `run` (on a `MethodBase` subclass)
- `get_info`
- `event_handler`

---

## 8. Service discovery — device presence protocol

**Not yet implemented.** Design is in `ARCHITECTURE.md § Service Discovery Pattern`.

Each device needs to participate in the bidirectional announce pattern:

1. **On startup:** subscribe to `lh_manager.online`, then publish `device.online`
2. **On receiving `lh_manager.online`:** re-publish `device.online` (handles lh_manager
   restarting after devices are already online)
3. **Every 30 s:** publish `device.heartbeat`
4. **On clean shutdown:** publish `device.offline`

`device.online` payload: `{device_id, device_type, base_url, n_channels}`

**Implementation location:** `broker_plugin.py` — extend `DeviceBrokerWorker.start()`
to bind a subscription to `lh_manager.online` and publish `device.online`. Add a
periodic heartbeat task to the asyncio event loop alongside the consumer task.

**Queue note:** lh_manager's `device.online` queue must be durable; presence messages
themselves must NOT be published as persistent (delivery_mode=TRANSIENT) — presence is
ephemeral and a persisted stale message would misrepresent reality after device restart.

---

## 9. Schema-driven method launcher in the UI

**Goal:** allow methods to be triggered directly from `roadmap.html` without needing the
broker or a separate API call. Useful for standalone dev and manual operation.

**Design decision:** use this alongside (not instead of) the existing explicit `controls` +
`event_handler` pattern. Explicit controls remain better for polished per-device UX
(plots, progress bars, monitoring widgets). Auto-generation covers the "just run it"
dev/standalone case with zero per-method boilerplate.

**Scope:** ~30 lines Python in `methods.py`, ~60 lines JS in `roadmap.html`. No new files.

### Complications to handle

- **`MethodDefinition.name` must be filtered** — it's the method name, not a user arg.
- **Type mapping is partial:** `float`/`int` → number input, `str` → textbox, `bool` →
  checkbox, `Literal[...]` → select. Anything else (lists, dicts, custom objects) gets
  a raw JSON textbox with a warning, or is excluded. Use `typing.get_type_hints()` for
  reliable resolution — `.type` on a `Field` may be a string annotation.
- **`is_ready()` must be surfaced** so the Run button can be disabled when devices are
  reserved. Without this, clicking Run during an active method is a silent no-op.
- **The existing `number` control sends `{n_prime: value}` as the data key** (hardcoded
  in `update_controls()` at line 382 of `roadmap.html`). The method launcher needs its
  own emit path: `issue_command(id, 'run_method', {method_name, method_data})` where
  `method_data` is `{field_name: value}` for all fields.
- **`update_controls()` bails when an input has focus** — correct behavior, but means
  `is_ready` state won't refresh mid-form-fill. Acceptable trade-off.
- **Multi-channel assemblies:** `MultiChannelAssembly` dispatches by `task.channel`.
  A UI-triggered call has no channel. Exclude multi-channel assemblies from
  auto-generation for now (or add a channel selector).

### Server-side changes (`methods.py`)

1. Add `method_schema` to `MethodPlugin.get_info()` alongside `active_methods`:

```python
'method_schema': {
    method_name: {
        'is_ready': m.is_ready(),
        'fields': [
            {'name': 'volume', 'type': 'number', 'default': 100.0},
            {'name': 'flow_rate', 'type': 'number', 'default': 1.0},
        ]
    }
    for method_name, m in self.methods.items()
}
```

2. Add `run_method` branch to `MethodPlugin.event_handler()`:

```python
elif command == 'run_method':
    method_name = data['method_name']
    method_data = data['method_data']
    if method_name in self.methods:
        self.run_method(method_name, method_data)
```

### Client-side changes (`roadmap.html`)

Add `update_method_launcher()` — reads `data['method_schema']`, renders one
`<details>`/`<summary>` per method (collapsed by default), with field inputs and a
Run button disabled when `is_ready == false`. Wire it into the assembly branch of
`update()` alongside `update_active_methods()` and `update_controls()`.

### Relationship to service discovery

The device-side schema serialization (`MethodDefinition` dataclass → `{name, type,
default}` field list) is **shared infrastructure** with any future broker-based service
discovery. Implement it once here and reuse it there.

However, service discovery for lh_manager is a **separate and harder problem**:

- `lh_manager.device_list.updated` (in ARCHITECTURE.md) is about device availability,
  not method schemas. No broker-based method schema transfer is currently designed or
  implemented.
- lh_manager currently learns about device methods via **static Python imports** in
  `roadmapmethods.py` (`@register(origin='ROADMAP')` on imported lh_devices classes).
- lh_manager's `MethodManager` does more than store field names — `BaseMethod.render_method()`
  translates abstract parameters into Gilson-specific command format. This logic is
  Python, not derivable from a schema. Pure schema-based dynamic registration would
  lose this translation layer.
- Replacing static registration with dynamic discovery would require Pydantic
  `create_model()` on the lh_manager side, or a "remote method" wrapper class that
  passes method_data through without Pydantic validation.

**Practical guidance:** implement the device-side serialization here (for the UI
launcher), but do not conflate it with full lh_manager service discovery. The
lh_manager-side dynamic registration is a separate architectural decision with
significant complexity that should be designed independently.
