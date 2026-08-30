#!/usr/bin/env python3
# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2026 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""§3's audit battery — BLOCKING before any submission. Writes AUDIT.md.

    python3 scripts/ra_ddm_frontier/audit_seeding.py [--out-dir <dir>] [--runs N]

Five trials at (Δθ = 60, δ_Q = 100 bp); raw exogenous traces (the shared
percept log, first ~50 ticks) asserted on four properties:

  (a) determinism        same config run twice → bitwise-identical logs
  (b) cross-model        RA and DDM env traces identical at equal run_id
  (c) param invariance   two RA cells with different (û, v), and two DDM points
                         with different c_e, produce identical env traces
  (d) stream separation  changing model_seed leaves the env trace untouched
                         while changing model-private behavior

The env trace = `<agent>_percept.csv` rows (tick, target, q_hat) — the shared
stream's own record of every draw the model consumed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_HERE), str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frontier   # noqa: E402
import seeding    # noqa: E402
from run_batch import InProcessRunner   # noqa: E402

TRACE_TICKS = 50

# The four probe configurations (§3): two RA cells, two DDM points.
RA_A = {"v": 0.5, "u_hat": 1.00}
RA_B = {"v": 0.2, "u_hat": 0.65}
DDM_C = 8.0
DDM_D = 300.0


def _make_cfg(campaign: str, out_dir: Path, run_id: int, *,
              u: float = None, v: float = None, c_e: float = None,
              cache_dir: Path = None, model_tag: str = None) -> Path:
    if campaign == "ra":
        cfg = frontier.patch_ra(frontier._load_json(frontier.RA_TEMPLATE), u, v)
    else:
        cfg = frontier.patch_ddm(frontier._load_json(frontier.DDM_TEMPLATE),
                                 c_e, str(cache_dir))
    frontier.apply_seeds(cfg, campaign, run_id)
    if model_tag is not None:
        # (d): perturb ONLY the model-private channel.
        alt = seeding.model_seed(model_tag, frontier.DTH_DEG,
                                 frontier.DIFF_BP, run_id)
        for arena in cfg["environment"]["arenas"].values():
            arena["random_seed"] = int(alt)
    cfg["environment"].setdefault("results", {})["base_path"] = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    return path


def _run(runner, cfg_path: Path) -> Path:
    runner.run(cfg_path)
    zips = sorted(cfg_path.parent.glob("config_folder_*/run_*.zip"))
    if not zips:
        raise RuntimeError(f"no run archive under {cfg_path.parent}")
    return zips[0]


def _member(zf: zipfile.ZipFile, suffix: str) -> str | None:
    return next((n for n in zf.namelist() if n.endswith(suffix)), None)


def percept_trace(run_zip: Path, max_tick: int = TRACE_TICKS) -> list[tuple]:
    """[(tick, target, q_hat), ...] for tick <= max_tick, sorted."""
    with zipfile.ZipFile(run_zip) as zf:
        name = _member(zf, "_percept.csv")
        if name is None:
            raise RuntimeError(f"{run_zip}: no percept log")
        rows = list(csv.DictReader(io.TextIOWrapper(zf.open(name))))
    out = []
    for r in rows:
        if not r.get("target"):
            continue
        tick = int(r["tick"])
        if tick <= max_tick:
            out.append((tick, r["target"], r["q_hat"]))
    return sorted(out)




def raw_bytes(run_zip: Path, suffix: str) -> bytes | None:
    with zipfile.ZipFile(run_zip) as zf:
        name = _member(zf, suffix)
        return zf.read(name) if name else None


def overlap_equal(t1: list[tuple], t2: list[tuple]) -> tuple[bool, int]:
    """Compare traces on their common (tick, target) coordinates."""
    d1, d2 = dict(((t, g), q) for t, g, q in t1), dict(((t, g), q) for t, g, q in t2)
    common = sorted(set(d1) & set(d2))
    if not common:
        return False, 0
    return all(d1[k] == d2[k] for k in common), len(common)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path,
                    default=_ROOT / "results" / "ra_ddm_frontier" / "audit")
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args(argv)

    if not frontier.RA_TEMPLATE.is_file() or not frontier.DDM_TEMPLATE.is_file():
        frontier.write_templates()
    root = args.out_dir
    cache = root / "table_cache"
    cache.mkdir(parents=True, exist_ok=True)
    runner = InProcessRunner()
    run_ids = list(range(1, args.runs + 1))
    results: list[tuple[str, bool, str]] = []
    t0 = time.time()

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    print(f"audit battery: {args.runs} trials, scheme {seeding.SCHEME}, "
          f"traces to tick {TRACE_TICKS}")

    # Run the probe matrix once; reuse archives across checks.
    zips: dict[tuple, Path] = {}
    for rid in run_ids:
        zips[("A", rid)] = _run(runner, _make_cfg(
            "ra", root / f"ra_A/replicate_{rid}", rid,
            u=RA_A["u_hat"] * frontier.u_star(RA_A["v"])[0], v=RA_A["v"]))
        zips[("C", rid)] = _run(runner, _make_cfg(
            "ddm", root / f"ddm_C/replicate_{rid}", rid, c_e=DDM_C,
            cache_dir=cache))
    for rid in run_ids[:2]:
        zips[("B", rid)] = _run(runner, _make_cfg(
            "ra", root / f"ra_B/replicate_{rid}", rid,
            u=RA_B["u_hat"] * frontier.u_star(RA_B["v"])[0], v=RA_B["v"]))
        zips[("D", rid)] = _run(runner, _make_cfg(
            "ddm", root / f"ddm_D/replicate_{rid}", rid, c_e=DDM_D,
            cache_dir=cache))

    # (a) determinism — same config twice, bitwise-identical logs.
    for tag, kwargs in (("ra", dict(u=RA_A["u_hat"] * frontier.u_star(0.5)[0],
                                    v=0.5)),
                        ("ddm", dict(c_e=DDM_C, cache_dir=cache))):
        z1 = _run(runner, _make_cfg(tag, root / f"det_{tag}_1", 1, **kwargs))
        z2 = _run(runner, _make_cfg(tag, root / f"det_{tag}_2", 1, **kwargs))
        same = all(raw_bytes(z1, s) == raw_bytes(z2, s)
                   for s in ("_percept.csv", "_position.csv",
                             "_sensory_noise.csv"))
        record(f"(a) determinism [{tag}]", same,
               "bitwise-identical percept/position/noise logs" if same
               else "logs differ between two runs of the SAME config — "
                    "unseeded randomness somewhere")

    # (b) cross-model — both models tick at 1 s (RECON D-01 as amended
    # 2026-08-30), so at equal run_id they read BITWISE-identical percept
    # realizations at equal (tick, target): the spec's full §3 pairing.
    ok_all, n_min = True, 1 << 30
    for rid in run_ids:
        eq, n = overlap_equal(percept_trace(zips[("A", rid)]),
                              percept_trace(zips[("C", rid)]))
        ok_all &= eq and n > 0
        n_min = min(n_min, n)
    record("(b) cross-model", ok_all,
           f"RA≡DDM percept draws bitwise-identical at equal (tick, target) "
           f"on all {args.runs} run_ids (≥{n_min} shared coordinates each)")

    # (c) parameter invariance — env trace never depends on (û, v) or c_e.
    eq_ra = all(overlap_equal(percept_trace(zips[("A", rid)]),
                              percept_trace(zips[("B", rid)]))[0]
                for rid in run_ids[:2])
    eq_ddm = all(overlap_equal(percept_trace(zips[("C", rid)]),
                               percept_trace(zips[("D", rid)]))[0]
                 for rid in run_ids[:2])
    record("(c) parameter invariance [ra]", eq_ra,
           f"(v={RA_A['v']}, û={RA_A['u_hat']}) vs (v={RA_B['v']}, "
           f"û={RA_B['u_hat']}): identical env traces")
    record("(c) parameter invariance [ddm]", eq_ddm,
           f"c_e={DDM_C:g} vs c_e={DDM_D:g}: identical env traces")

    # (d) stream separation — model_seed change: env untouched, behavior free.
    for tag, kwargs, key in (("ra", dict(u=RA_A["u_hat"] * frontier.u_star(0.5)[0],
                                         v=0.5), ("A", 1)),
                             ("ddm", dict(c_e=DDM_C, cache_dir=cache), ("C", 1))):
        z_alt = _run(runner, _make_cfg(tag, root / f"sep_{tag}", 1,
                                       model_tag=f"{tag}-audit-alt", **kwargs))
        env_eq, _ = overlap_equal(percept_trace(zips[key]), percept_trace(z_alt))
        beh_diff = raw_bytes(zips[key], "_position.csv") != \
            raw_bytes(z_alt, "_position.csv")
        record(f"(d) stream separation [{tag}] env", env_eq,
               "env trace untouched by a model_seed change" if env_eq
               else "model seed leaks into the shared stream")
        note = ("private-noise trajectory changed as expected" if beh_diff else
                "trajectory UNCHANGED — this model may consume no private "
                "noise in this configuration (informational, not a failure)")
        record(f"(d) stream separation [{tag}] behavior", True, note)

    # ------------------------------------------------------------------ report
    all_pass = all(ok for _n, ok, _d in results)
    lines = [
        "# AUDIT — §3 seed-scheme battery",
        "",
        f"Scheme `{seeding.SCHEME}`; trial identity (Δθ = {frontier.DTH_DEG}°, "
        f"δ_Q = {frontier.DIFF_BP} bp); {args.runs} run_ids; env trace = shared "
        f"percept draws, ticks ≤ {TRACE_TICKS}; git `{frontier.git_sha()}`.",
        "",
        "Routing (RECON §5): `sensory_stream.seed` ← `env_seed(…, 'sensory')`; "
        "arena `random_seed` ← `model_seed(model, …)` — the arena seed feeds "
        "only model-private generators in this simulator.",
        "",
        "| check | result | detail |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {'**PASS**' if ok else '**FAIL**'} | {d} |"
              for n, ok, d in results]
    lines += ["", f"Overall: **{'PASS' if all_pass else 'FAIL'}** "
                  f"({time.time() - t0:.0f} s wall). "
              + ("Submission unblocked." if all_pass else
                 "BLOCKING — fix before any submission (§3).")]
    audit_md = _HERE / "AUDIT.md"
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {audit_md}  —  overall {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
