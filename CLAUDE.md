<!-- GSD:project-start source:PROJECT.md -->
## Project

**CollectiPy**

CollectiPy is a Python simulation framework for studying collective decision-making in multi-agent swarms using ring attractor neural dynamics. Agents implement biologically-inspired attractor models (Ising spin system, mean-field ring attractor) and interact in a configurable arena. It is used by a PhD researcher to run decision-making experiments — binary choices and N-option competition — and generate results for publications.

**Core Value:** Agents running ring attractor dynamics should produce measurable, reproducible collective decisions that can be systematically explored via parameter sweeps.

### Constraints

- **Tech stack**: Python 3.10 + NumPy/SciPy/Numba — all new code must remain compatible with this stack
- **Output format**: pkl/csv output kept as-is — analysis tooling wraps it, not replaces it
- **Research pace**: Framework improvements must not break existing experiment configs
- **Hardware**: Runs on a single Linux workstation — no distributed compute constraints to design for
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.10 - All simulation, modeling, GUI, and data handling code
- Bash - Shell scripts for running simulation scenarios (`run.sh`, `run-mean.sh`, and all `run-mean-sweep-*.sh` scripts)
- JSON - Configuration format for all experiment definitions (`config/`)
## Runtime
- CPython 3.10.12 (Linux)
- pip (via venv)
- Lockfile: Not present (only `requirements.txt` with unpinned package names)
- Virtual environment: `.venv/` (Python 3.10 venv)
## Frameworks
- PySide6 6.10.1 - Qt6 bindings for the 2D simulation visualization (`src/gui.py`)
- NumPy 2.2.6 - Array operations throughout all model files, data handling, geometry utilities
- SciPy 1.15.3 - Scientific algorithms (imported by models)
- Numba 0.62.1 - JIT compilation for performance-critical loops in `src/models/mean_field_systems.py` (`@njit`, `prange`)
- Matplotlib 3.10.7 - Plotting, colormaps (`cm`), neural activation plots in GUI panels
## Key Dependencies
- `numpy` 2.2.6 - Core array computing used everywhere; changing major version may break numba compatibility
- `numba` 0.62.1 - JIT-compiles the mean-field ODE integration loops; requires `llvmlite` to match
- `PySide6` 6.10.1 - Qt6 GUI; includes `shiboken6` and separate `pyside6_essentials`/`pyside6_addons` packages
- `scipy` 1.15.3 - Scientific algorithms for model computations
- `matplotlib` 3.10.7 - Used both standalone for output plots and embedded in the Qt GUI
- `psutil` 7.1.3 - CPU affinity management in `src/environment.py` for multi-process core assignment
- `llvmlite` 0.45.1 - LLVM bindings required by numba for JIT compilation
- `pillow` 12.0.0 - Image support for matplotlib
- `six` 1.17.0 - Python 2/3 compatibility shim (pulled in transitively)
## Configuration
- All simulation parameters are loaded from JSON config files in `config/`
- No `.env` files or environment variables used; configuration is entirely file-based
- Config is loaded by `src/config.py` (`Config` class) and passed through the system
- Key config sections: `environment.arena`, `environment.agents`, `environment.objects`, `environment.gui`, `environment.logging`, `environment.results`
- `pyrightconfig.json` - Type checker configuration pointing to `.venv` and adding `./src` to extra paths
- No `setup.py`, `pyproject.toml`, or build toolchain; project is run directly from source
## Platform Requirements
- Python 3.10+ (scripts explicitly look for `python3.10` first)
- Linux (CPU affinity calls use `psutil.Process.cpu_affinity` which requires Linux)
- Virtual environment at project root (`.venv/`)
- Qt6 runtime libraries (bundled with PySide6)
- No packaging or deployment pipeline; simulations are run directly via `run*.sh` scripts
- Parallel parameter sweeps use `multiprocessing` from the Python standard library with manual CPU core affinity assignment via `psutil`
- Output data written to `./data/` as timestamped folder structures containing `.pkl` (pickle), `.csv`, and `.zip` compressed archives
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Code Style
- **Python version:** 3.10; uses `match`-equivalent constructs, `dict[str, ...]` type hints (lowercase generics)
- **Formatting:** No formatter config present (no `black`, `ruff`, or `isort` config); code style is largely consistent but not enforced by tooling
- **Type annotations:** Used selectively — function signatures often annotated (e.g., `entity_type: str`, `config_elem: dict`), but not exhaustively; `Optional[T]` from `typing` used alongside `T | None` union syntax
- **Docstrings:** One-line docstrings on most classes and methods: `"""Entity."""`, `"""Initialize the instance."""` — minimal, non-PEP-257 style
- **Comments:** Inline comments used to explain intent; some Italian-language comments remain in `environment.py` (e.g., `"Ritorna 'num' core meno usati..."`)
- **Line length:** No enforced limit; lines are generally kept reasonable
## Naming
| Construct | Convention | Example |
|-----------|------------|---------|
| Modules | `snake_case` | `entityManager.py`, `collision_detector.py` |
| Classes | `PascalCase` | `EntityManager`, `SpinMovementModel` |
| Methods | `snake_case` | `create_entity()`, `arena_init()` |
| Private methods | `_snake_case` | `_blocking_get()`, `_split_agents()` |
| Class constants | `UPPER_SNAKE_CASE` | `PLACEMENT_MAX_ATTEMPTS`, `DEFAULT_RX_RATE` |
| Instance vars | `snake_case` | `self.entity_type`, `self.spin_system` |
| Type aliases | `PascalCase` | (rare — mostly inline annotations) |
| Config keys | `snake_case` | `"num_runs"`, `"time_limit"`, `"spin_model"` |
## Patterns
## Error Handling
- **Config validation:** Explicit `raise ValueError(...)` for invalid config shapes — e.g., `"Invalid agents configuration: expected a dictionary."`
- **Process errors:** `try/except Exception as e` with `logging.error(...)` in affinity assignment and IPC helpers (broad catch, logged and continued)
- **Startup errors:** `try/except Exception as e` in `main()` with `traceback.print_exc()` and `sys.exit(1)`
- **Bare except:** Some instances of bare `except:` present (flagged as tech debt)
- **No custom exception hierarchy:** Uses built-in exceptions (`ValueError`, `Exception`) throughout
## Logging
- Module-level loggers created with `logging.getLogger("sim.<module>")` pattern:
- Configured via `logging_utils.configure_logging(config, config_path, project_root)`
- Log level and output path set from JSON config under `environment.logging`
- `logger.setLevel(logging.DEBUG)` called directly on some module loggers
## Imports
- Standard library imports first, then third-party, then local — generally followed but not enforced
- Relative imports avoided; `src/` added to `sys.path` for flat imports within `src/`
- `# noqa: F401` used for side-effect imports: `import models  # noqa: F401  # ensure built-in models register themselves`
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern
## Layers
```
```
## Data Flow
## Abstractions
- `MovementModel` — `step(agent, tick, arena_shape, objects, agents)`
- `LogicModel` — `step(agent, tick, arena_shape, objects, agents)`
- `DetectionModel` — `sense(agent, objects, agents, arena_shape)`
- `MessageBus` — decoupled agent-to-agent messaging
- `Entity` → `StaticAgent`, `MovableAgent`, `StaticObject`, `MovableObject`
- All created via `EntityFactory.create_entity(entity_type, config, id)`
- Entity type string `"agent_movable"` parsed to select concrete class
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
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, or `.github/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
