# Code Conventions

**Analysis Date:** 2026-04-08

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

**Factory pattern** — used throughout for object creation:
```python
# src/entity.py
class EntityFactory:
    @staticmethod
    def create_entity(entity_type: str, config_elem: dict, _id: int = 0):
        ...

# src/environment.py
class EnvironmentFactory:
    @staticmethod
    def create_environment(config_elem: Config):
        ...
```

**Plugin Protocol pattern** — structural subtyping via `Protocol`:
```python
# src/plugin_base.py
class MovementModel(Protocol):
    def step(self, agent, tick, arena_shape, objects, agents) -> None: ...
```

**Registry pattern** — plugins registered by string key, looked up at runtime:
```python
# src/plugin_registry.py
register_movement_model("spin_model", SpinMovementModel)
model = get_movement_model("spin_model")
```

**Config-driven construction** — all parameters come from JSON config dicts passed through the call chain:
```python
my_config = Config(config_path=configfile)
my_env = EnvironmentFactory.create_environment(my_config)
```

**Numba JIT for hot loops** — performance-critical ODE integration in `mean_field_systems.py`:
```python
@njit
def _step_euler_core(z, M, u0, s_arr, beta, b, sigma, dt, ...):
    ...
```

## Error Handling

- **Config validation:** Explicit `raise ValueError(...)` for invalid config shapes — e.g., `"Invalid agents configuration: expected a dictionary."`
- **Process errors:** `try/except Exception as e` with `logging.error(...)` in affinity assignment and IPC helpers (broad catch, logged and continued)
- **Startup errors:** `try/except Exception as e` in `main()` with `traceback.print_exc()` and `sys.exit(1)`
- **Bare except:** Some instances of bare `except:` present (flagged as tech debt)
- **No custom exception hierarchy:** Uses built-in exceptions (`ValueError`, `Exception`) throughout

## Logging

- Module-level loggers created with `logging.getLogger("sim.<module>")` pattern:
  ```python
  logger = logging.getLogger("sim.entity_manager")
  logger.info("Agents initialized: total=%s groups=%s", ...)
  ```
- Configured via `logging_utils.configure_logging(config, config_path, project_root)`
- Log level and output path set from JSON config under `environment.logging`
- `logger.setLevel(logging.DEBUG)` called directly on some module loggers

## Imports

- Standard library imports first, then third-party, then local — generally followed but not enforced
- Relative imports avoided; `src/` added to `sys.path` for flat imports within `src/`
- `# noqa: F401` used for side-effect imports: `import models  # noqa: F401  # ensure built-in models register themselves`

---

*Conventions analysis: 2026-04-08*
