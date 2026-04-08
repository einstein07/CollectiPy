# Technology Stack

**Analysis Date:** 2026-04-08

## Languages

**Primary:**
- Python 3.10 - All simulation, modeling, GUI, and data handling code

**Secondary:**
- Bash - Shell scripts for running simulation scenarios (`run.sh`, `run-mean.sh`, and all `run-mean-sweep-*.sh` scripts)
- JSON - Configuration format for all experiment definitions (`config/`)

## Runtime

**Environment:**
- CPython 3.10.12 (Linux)

**Package Manager:**
- pip (via venv)
- Lockfile: Not present (only `requirements.txt` with unpinned package names)
- Virtual environment: `.venv/` (Python 3.10 venv)

## Frameworks

**GUI:**
- PySide6 6.10.1 - Qt6 bindings for the 2D simulation visualization (`src/gui.py`)
  - Uses `QApplication`, `QWidget`, `QVBoxLayout`, `QGraphicsView`, `QGraphicsScene`, `QTimer`, `QShortcut`
  - Matplotlib embedded in Qt via `FigureCanvasQTAgg` from `matplotlib.backends.backend_qtagg`

**Scientific Computing / Numerical:**
- NumPy 2.2.6 - Array operations throughout all model files, data handling, geometry utilities
- SciPy 1.15.3 - Scientific algorithms (imported by models)
- Numba 0.62.1 - JIT compilation for performance-critical loops in `src/models/mean_field_systems.py` (`@njit`, `prange`)
  - Depends on llvmlite 0.45.1 as backend

**Visualization:**
- Matplotlib 3.10.7 - Plotting, colormaps (`cm`), neural activation plots in GUI panels
  - Companion libs: contourpy 1.3.2, cycler, kiwisolver, pyparsing, python-dateutil, pillow, fonttools, packaging

## Key Dependencies

**Critical:**
- `numpy` 2.2.6 - Core array computing used everywhere; changing major version may break numba compatibility
- `numba` 0.62.1 - JIT-compiles the mean-field ODE integration loops; requires `llvmlite` to match
- `PySide6` 6.10.1 - Qt6 GUI; includes `shiboken6` and separate `pyside6_essentials`/`pyside6_addons` packages
- `scipy` 1.15.3 - Scientific algorithms for model computations
- `matplotlib` 3.10.7 - Used both standalone for output plots and embedded in the Qt GUI
- `psutil` 7.1.3 - CPU affinity management in `src/environment.py` for multi-process core assignment

**Infrastructure:**
- `llvmlite` 0.45.1 - LLVM bindings required by numba for JIT compilation
- `pillow` 12.0.0 - Image support for matplotlib
- `six` 1.17.0 - Python 2/3 compatibility shim (pulled in transitively)

## Configuration

**Environment:**
- All simulation parameters are loaded from JSON config files in `config/`
- No `.env` files or environment variables used; configuration is entirely file-based
- Config is loaded by `src/config.py` (`Config` class) and passed through the system
- Key config sections: `environment.arena`, `environment.agents`, `environment.objects`, `environment.gui`, `environment.logging`, `environment.results`

**Build:**
- `pyrightconfig.json` - Type checker configuration pointing to `.venv` and adding `./src` to extra paths
- No `setup.py`, `pyproject.toml`, or build toolchain; project is run directly from source

## Platform Requirements

**Development:**
- Python 3.10+ (scripts explicitly look for `python3.10` first)
- Linux (CPU affinity calls use `psutil.Process.cpu_affinity` which requires Linux)
- Virtual environment at project root (`.venv/`)
- Qt6 runtime libraries (bundled with PySide6)

**Production:**
- No packaging or deployment pipeline; simulations are run directly via `run*.sh` scripts
- Parallel parameter sweeps use `multiprocessing` from the Python standard library with manual CPU core affinity assignment via `psutil`
- Output data written to `./data/` as timestamped folder structures containing `.pkl` (pickle), `.csv`, and `.zip` compressed archives

---

*Stack analysis: 2026-04-08*
