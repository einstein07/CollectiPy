# Concerns

**Analysis Date:** 2026-04-08

## Technical Debt

**Bare `except:` clauses** — `src/entityManager.py` and `src/environment.py` contain broad `except:` and `except Exception:` catches that swallow errors silently or log without re-raising. This makes debugging failures difficult.

**Duplicated IPC helpers** — `_blocking_get` and `_maybe_get` patterns appear in both `EntityManager` and `Environment`; should be consolidated into a shared utility.

**Duplicated detection-range logic** — `SpinMovementModel._resolve_detection_range()` and `MeanFieldMovementModel` both compute detection ranges independently using similar logic.

**Dead/commented-out code** — Commented print statements and disabled code blocks (e.g., `# print(f"[AFFINITY] PID ...")`) scattered through `environment.py`.

**Italian-language comments** — `environment.py` contains Italian comments: `"Ritorna 'num' core meno usati che NON sono in used_cores."` and `"Ordina i core dal meno usato"`. Should be translated for a multilingual team.

**Latent `AttributeError` in `MeanFieldSystem.reset()`** — If `reset()` is called before `__init__` completes (e.g., in error recovery), instance attributes may not exist yet.

**Entity type string parsing** — `EntityFactory.create_entity` parses entity type via `entity_type.split('_')[0]+'_'+entity_type.split('_')[1]`, which silently fails on unexpected string formats rather than giving a clear error.

## Performance Bottlenecks

**Per-tick full trajectory allocation** — `MeanFieldSystem` allocates new arrays each tick for trajectory logging, causing GC pressure in long runs.

**`numba.prange` outside `@njit(parallel=True)`** — `prange` is used in some loops that are not compiled with `parallel=True`, so no actual parallelism is achieved; the loop runs sequentially despite the intent.

**O(N²) Hamiltonian recomputation** — `SpinSystem` recomputes the full interaction Hamiltonian each tick rather than maintaining an incremental update, making it expensive for large N.

**`pop(0)` on list** — Some queue-like structures use `list.pop(0)` (O(N)) instead of `collections.deque.popleft()` (O(1)).

**Unbounded GUI queue payloads** — Full agent state snapshots sent to GUI process each tick; no throttling or payload compression, which can cause GUI lag for large swarms.

## Fragile Areas

**Race-prone config folder numbering** — `dataHandling.py` selects the next `config_folder_N` by scanning the directory; concurrent parallel runs can collide on the same folder number under a race condition.

**File handle leaks** — Some data writing paths open file handles without `with` context managers, risking leaks if an exception occurs mid-write.

**Class-level mutable entity registry** — `Entity._class_registry` and `Entity._used_prefixes` are class-level mutable dicts/sets shared across all instances; forked child processes inherit these by value but mutations in one process are not visible to others, which can cause subtle UID collision bugs.

**Unconditional `fork` on non-Linux** — CPU affinity via `psutil.Process.cpu_affinity()` is Linux-only; the code attempts it on any platform with a try/except fallback, but `multiprocessing` fork semantics differ on macOS, which could cause issues.

**Config folder numbering assumes sequential integers** — If folders are manually deleted mid-run, the counter can reuse an old index and overwrite existing data.

## Security

**No secrets in config** — JSON configs contain only simulation parameters; no API keys or credentials present.

**Pickle deserialisation** — `dataHandling.py` writes and reads `.pkl` files using Python's `pickle` module. Loading pickles from untrusted sources is a code execution risk; acceptable here since data is self-generated, but worth noting.

**No input sanitisation** — Config file paths passed directly to `Path(...).expanduser().resolve()` without validation; acceptable for a research tool used locally.

## Missing Critical Features

**Zero automated tests** — No test suite; correctness relies entirely on visual inspection of simulation output. Numerical bugs in ODE integration or ring attractor dynamics could go undetected.

**No numerical stability checks** — Euler integration in `MeanFieldSystem` uses a fixed `dt` with no stability monitoring. Divergent trajectories are not detected or reported.

**No experiment reproducibility guarantee** — While seeds are set per agent, the parallel process scheduling is non-deterministic, so exact reproduction across runs with different hardware is not guaranteed.

**No CI/CD** — No `.github/workflows/`, `Makefile`, or equivalent; there is no automated test or lint pipeline.

---

*Concerns analysis: 2026-04-08*
