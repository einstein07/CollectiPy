# External Integrations

**Analysis Date:** 2026-04-08

## APIs & External Services

None. CollectiPy is a self-contained research simulation framework with no calls to external APIs, cloud services, or third-party web services.

## Data Storage

**Databases:**
- None. No relational or document databases are used.

**File Storage:**
- Local filesystem only
  - Simulation results: `./data/config_folder_N/run_N/` (auto-created at runtime)
  - Per-agent spatial data: `{agent_id}.pkl` (Python pickle, highest protocol)
  - Per-agent spin model data: `{agent_id}_spins.pkl`
  - Mean-field neural/perception/sensory/position data: `{agent_id}_neural.csv`, `{agent_id}_perception.csv`, `{agent_id}_sensory.csv`, `{agent_id}_targets.csv`, `{agent_id}_position.csv`
  - Graph snapshots: `graphs/messages/step_NNNNNNNNN.pkl`, `graphs/detection/step_NNNNNNNNN.pkl`
  - All run folders compressed to `.zip` archives after run completes (deflate, level 9)
  - Log files: `./logs/YYYYMMDD-HHMMSS_<hash>.log.zip`
  - Config snapshots: `./logs/configs/`
  - Config-to-log mapping: `./logs/logs_configs_mapping.csv`
  - Implementation: `src/dataHandling.py` (`DataHandling`, `SpaceDataHandling`)

**Caching:**
- None

## Authentication & Identity

**Auth Provider:**
- Not applicable. No user authentication; the system is a local research tool.

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry, Rollbar, or similar)

**Logs:**
- Python standard `logging` module via `src/logging_utils.py`
- Log output: optional file sink (compressed `.log.zip` archives in `./logs/`) and optional console stream
- Logging enabled only when a `"logging"` section is present in the JSON config
- Log namespace: `"sim"` (all loggers use `sim.<component>` hierarchy)
- Severity levels configurable per-sink: `file_level` and `console_level` in config

## CI/CD & Deployment

**Hosting:**
- Not applicable. Local execution only.

**CI Pipeline:**
- None detected. No `.github/`, `.gitlab-ci.yml`, or similar CI config files present.

## Plugin System

**External Plugins:**
- The simulator supports runtime-loaded Python plugins declared in the config JSON under `"plugins"` or `"environment.plugins"` keys
- Plugins are imported via `importlib.import_module` at startup (`src/plugin_registry.py` `load_plugins_from_config`)
- Example plugin: `plugins/random_waypoint_cleanup.py`
- Plugin protocols defined in `src/plugin_base.py`: `MovementModel`, `LogicModel`, `DetectionModel`, `MessageBusModel`, `MotionModel`
- Registering: plugins call `register_movement_model`, `register_motion_model`, `register_logic_model`, `register_detection_model`, or `register_message_bus` from `src/plugin_registry.py`

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Environment Configuration

**Required env vars:**
- None. Configuration is entirely file-based (JSON config files in `config/`).

**Secrets location:**
- Not applicable. No secrets, credentials, or API keys are used.

## Inter-process Communication

**Multiprocessing:**
- Python `multiprocessing` standard library used in `src/environment.py` for parallel parameter sweeps
- GUI runs in a dedicated process communicating via `multiprocessing.Queue` objects (`gui_in_queue`, `gui_control_queue`)
- CPU core affinity for simulation processes managed via `psutil.Process.cpu_affinity` (Linux only)

---

*Integration audit: 2026-04-08*
