# Directory Structure

**Analysis Date:** 2026-04-08

## Layout

```
CollectiPy/
├── src/                          # All simulation source code
│   ├── main.py                   # Entry point — CLI arg parsing, env startup
│   ├── config.py                 # Config loader (JSON → Config object)
│   ├── environment.py            # Env orchestration, process management
│   ├── entityManager.py          # Per-process tick loop, IPC sync
│   ├── entity.py                 # Entity/Agent class hierarchy + factory
│   ├── arena.py                  # Arena factory and base implementations
│   ├── arena_hierarchy.py        # Multi-zone arena with hierarchy support
│   ├── collision_detector.py     # Agent/object collision detection
│   ├── messagebus.py             # Agent messaging infrastructure
│   ├── gui.py                    # PySide6 Qt6 visualisation + neural plots
│   ├── dataHandling.py           # Output serialisation (pkl, csv, zip)
│   ├── logging_utils.py          # Structured logging configuration
│   ├── plugin_base.py            # Protocol interfaces for plugins
│   ├── plugin_registry.py        # Plugin registration & lookup hub
│   ├── geometry_utils/
│   │   ├── vector3D.py           # 3D vector math
│   │   └── spatialgrid.py        # Spatial hash grid for neighbour queries
│   ├── bodies/
│   │   └── shapes3D.py           # 3D body/shape geometry
│   └── models/
│       ├── utils.py              # Shared math utilities (normalize_angle, etc.)
│       ├── spinsystem.py         # Ising spin system (O(N²) Hamiltonian)
│       ├── mean_field_systems.py # Mean-field ring attractor (Numba JIT)
│       ├── detection/
│       │   ├── visual.py         # Visual/angular detection model
│       │   └── gps.py            # GPS (global position) detection model
│       ├── logic/
│       │   ├── common.py         # Shared logic utilities
│       │   └── hierarchy_confinement.py  # Zone confinement logic model
│       ├── motion/
│       │   └── unicycle.py       # Unicycle kinematic motion model
│       └── movement/
│           ├── spin_model.py     # SpinSystem-driven movement
│           ├── mean_field_model.py # MeanField ring attractor movement
│           ├── random_walk.py    # Random walk movement
│           └── random_way_point.py # Random waypoint movement
│
├── plugins/                      # External plugin examples
│   ├── __init__.py
│   ├── random_waypoint_cleanup.py
│   └── examples/
│       ├── collision_handshake_plugin.py
│       └── group_stats_plugin.py
│
├── config/                       # JSON experiment configuration files
│   ├── mean_field_1_target.json
│   ├── mean_field_1_target_1_guard.json
│   └── ...
│
├── data/                         # Experiment output (auto-generated)
│   └── <experiment_name>/
│       └── config_folder_N/
│           ├── config.json       # Config snapshot for that run
│           └── run_N.zip         # Compressed simulation data
│
├── attention-2-beta/             # Legacy/archived experiment data
│   └── config_folder_N/
│
├── .venv/                        # Python virtual environment
├── compile.sh                    # Numba pre-compilation script
├── run.sh / run-mean.sh          # Single-run launch scripts
├── run-mean-sweep-*.sh           # Parameter sweep batch scripts
├── requirements.txt              # Unpinned pip dependencies
├── pyrightconfig.json            # Pyright type-checker config
└── CONTRIBUTORS.md
```

## Key Locations

| What | Where |
|------|-------|
| Simulation entry point | `src/main.py` |
| Config loading | `src/config.py` |
| Experiment config files | `config/*.json` |
| Neural models (ring attractor, spin) | `src/models/mean_field_systems.py`, `src/models/spinsystem.py` |
| Movement models | `src/models/movement/` |
| Detection models | `src/models/detection/` |
| Plugin interfaces | `src/plugin_base.py` |
| Plugin registration | `src/plugin_registry.py` |
| Custom plugins | `plugins/` |
| Output data | `data/` |
| GUI renderer | `src/gui.py` |

## Naming Conventions

- **Source modules:** `snake_case` (`entityManager.py`, `collision_detector.py`)
- **Classes:** `PascalCase` (`EntityManager`, `SpinMovementModel`)
- **Plugin protocol classes:** `PascalCase` suffixed with role (`MovementModel`, `LogicModel`, `DetectionModel`)
- **Factory classes:** `PascalCase` + `Factory` suffix (`EntityFactory`, `ArenaFactory`, `GuiFactory`)
- **Private helpers:** leading underscore (`_blocking_get`, `_split_agents`, `_PipeQueue`)
- **Config keys:** `snake_case` matching JSON field names
- **Data output folders:** `config_folder_N` (N = integer index)

---

*Structure analysis: 2026-04-08*
