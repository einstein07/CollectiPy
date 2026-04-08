# Architecture

**Analysis Date:** 2026-04-08

## Pattern

**Simulation Framework** — A discrete-time, multi-agent swarm simulator with a plugin-based behavioural model system. The architecture follows a process-per-agent-group design where agents run in isolated processes and communicate via IPC (Pipe/Queue), with a central EntityManager coordinating the tick cycle.

## Layers

```
┌─────────────────────────────────────────┐
│           Entry Point (main.py)         │
│         Config parsing, plugin loading  │
├─────────────────────────────────────────┤
│         Environment / Orchestration     │
│  environment.py — process management,   │
│  experiment loop, parallel runs         │
├──────────────┬──────────────────────────┤
│  EntityManager│      GUI (gui.py)        │
│  (tick loop, │  PySide6/Qt6 renderer,   │
│  IPC sync)   │  neural activation plots │
├──────────────┴──────────────────────────┤
│         Simulation Domain               │
│  entity.py — Entity/Agent classes       │
│  arena.py / arena_hierarchy.py          │
│  collision_detector.py                  │
│  messagebus.py — agent messaging        │
├─────────────────────────────────────────┤
│         Model Layer (src/models/)       │
│  movement/ — SpinModel, MeanFieldModel, │
│              RandomWalk, RandomWaypoint │
│  motion/ — Unicycle kinematics          │
│  detection/ — Visual, GPS               │
│  logic/ — HierarchyConfinement          │
│  spinsystem.py — Ising spin dynamics    │
│  mean_field_systems.py — ring attractor │
├─────────────────────────────────────────┤
│         Plugin System                   │
│  plugin_base.py — Protocol interfaces  │
│  plugin_registry.py — registration hub │
│  plugins/ — external plugin examples   │
├─────────────────────────────────────────┤
│         Utilities                       │
│  geometry_utils/ — Vector3D, SpatialGrid│
│  bodies/shapes3D.py — body geometry     │
│  logging_utils.py — structured logging  │
│  dataHandling.py — output serialisation │
└─────────────────────────────────────────┘
```

## Data Flow

1. **Startup:** `main.py` parses CLI args → loads `Config` from JSON → loads plugins → `EnvironmentFactory.create_environment(config)` → `env.start()`
2. **Per-experiment:** `Environment` spawns one process per agent group (split via `_split_agents`); each process runs an `EntityManager` tick loop
3. **Per-tick (EntityManager):**
   - Applies `LogicModel.step()` for each agent (pre-movement decisions)
   - Applies `MovementModel.step()` for each agent (position updates)
   - Syncs agent positions with arena / collision detector via IPC
   - Sends snapshot to GUI process (if rendering enabled)
4. **IPC:** `_PipeQueue` (Pipe-backed) used for agent→environment communication; `mp.Queue` used for GUI snapshots
5. **Output:** `dataHandling.py` serialises snapshots to `.pkl` / `.csv`, then compresses to `.zip` in `data/<experiment_name>/`

## Abstractions

**Plugin Protocols** (`plugin_base.py`):
- `MovementModel` — `step(agent, tick, arena_shape, objects, agents)`
- `LogicModel` — `step(agent, tick, arena_shape, objects, agents)`
- `DetectionModel` — `sense(agent, objects, agents, arena_shape)`
- `MessageBus` — decoupled agent-to-agent messaging

**Entity Hierarchy:**
- `Entity` → `StaticAgent`, `MovableAgent`, `StaticObject`, `MovableObject`
- All created via `EntityFactory.create_entity(entity_type, config, id)`
- Entity type string `"agent_movable"` parsed to select concrete class

**Arena Hierarchy:**
- `ArenaFactory.create_arena(config)` selects arena variant
- `ArenaHierarchy` manages spatial containment zones (for confinement logic)

## Entry Points

- **Simulation:** `python src/main.py -c config/my_config.json`
- **Batch sweeps:** `run-mean-sweep-*.sh` scripts call `main.py` in parallel with different configs
- **Plugin loading:** `plugin_registry.load_plugins_from_config(config)` — loads external modules declared under config key `plugins`

## Key Design Decisions

- **Process-per-group** parallelism: each agent group runs in its own process for CPU-level isolation; `psutil` assigns CPU affinity
- **Plugin system over inheritance:** Behaviours injected via `Protocol` registries rather than subclassing, keeping core classes stable
- **Numba JIT for mean-field ODE:** Hot integration loops in `mean_field_systems.py` decorated with `@njit` to approach C speed
- **Config-driven everything:** No hardcoded experiment parameters; all topology, models, and parameters come from JSON config files

---

*Architecture analysis: 2026-04-08*
