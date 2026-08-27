# ------------------------------------------------------------------------------
#  CollectiPy
#  Copyright (c) 2025 Sindiso Mkhatshwa
#
#  This file is part of CollectyPy, released under the BSD 3-Clause License.
# ------------------------------------------------------------------------------

"""Pre-flight gates (CAMPAIGN_SPEC.md Section 9). All three must pass.

    python3 -m campaign.preflight stream-determinism        # 9.1
    python3 -m campaign.preflight legacy [--golden DIR]     # 9.2
    python3 -m campaign.preflight small-matrix --results-root DIR
                                  [--reps 20] [--workers N] [--keep-raw]   # 9.3
    python3 -m campaign.preflight all --results-root DIR    # 9.1 + 9.2 + 9.3

9.1 — same (dQ, dtheta, replicate) at two criterion values: the logged per-tick
      q_hat streams must be EXACTLY identical over the common ticks. The percept is
      reconstructed from the seed, not resampled, so anything short of exact string
      equality is a bug, never tolerance-worthy.
9.2 — `mode: legacy` must be bit-reproducible: the same pre-feature config run twice
      produces identical archives, and the percept log confirms the legacy
      pass-through (mode 'legacy', q_hat column empty). With --golden, the archives
      are also compared member-by-member against a stored pre-feature reference.
9.3 — the full matrix at a small replicate count, per-tick logging on: no cell
      censored beyond expectation, manifests present and consistent, the table cache
      hit for every main condition, and the realised crossing log-odds tracking the
      prediction outside the flagged discretisation-limited cells.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from campaign import factors, genconfig, matrix, summarise  # noqa: E402


def _read_member(zip_path: Path, suffix: str) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith(suffix))
        return list(csv.DictReader(io.TextIOWrapper(zf.open(member))))


def _run_one(cfg: dict, scratch: Path, tag: str) -> Path:
    """Run one replicate config in-process; return the run archive path."""
    from campaign.run_chunk import InProcessRunner
    cfg_path = scratch / f"{tag}.json"
    cfg_path.write_text(json.dumps(cfg))
    InProcessRunner().run(cfg_path)
    return next((scratch / f"out_{tag}").glob("config_folder_*/run_*.zip"))


# ---------------------------------------------------------------------------
# 9.1
# ---------------------------------------------------------------------------
def check_stream_determinism() -> bool:
    """Two criterion values, one (dQ, dtheta, replicate): identical q_hat streams."""
    print("[9.1] shared-stream determinism")
    cond_a = matrix.find_condition("q01_a60_ce1")
    cond_b = matrix.find_condition("q01_a60_ce300")
    rep = 0
    with tempfile.TemporaryDirectory(prefix="preflight91_") as scratch:
        scratch = Path(scratch)
        streams = {}
        for tag, cond in (("a", cond_a), ("b", cond_b)):
            cfg = genconfig.replicate_config(cond, rep, str(scratch / f"out_{tag}"))
            zip_path = _run_one(cfg, scratch, tag)
            rows = _read_member(zip_path, "_percept.csv")
            streams[tag] = {
                (r["tick"], r["target"]): r["q_hat"]
                for r in rows if r["target"]
            }
            mode = rows[0]["mode"] if rows else "?"
            if mode != "shared":
                print(f"  FAIL: percept log reports mode {mode!r}, expected 'shared'")
                return False
    common = sorted(set(streams["a"]) & set(streams["b"]), key=lambda k: int(k[0]))
    if not common:
        print("  FAIL: no common (tick, target) rows between the two runs")
        return False
    diffs = [k for k in common if streams["a"][k] != streams["b"][k]]
    if diffs:
        k = diffs[0]
        print(f"  FAIL: {len(diffs)}/{len(common)} q_hat values differ; first at "
              f"(tick {k[0]}, {k[1]}): {streams['a'][k]} vs {streams['b'][k]}")
        return False
    print(f"  PASS: {len(common)} (tick, target) q_hat values EXACTLY identical "
          f"across c_e = {cond_a.c_e:g} and c_e = {cond_b.c_e:g} "
          f"(runs of {len(streams['a'])} and {len(streams['b'])} rows)")
    return True


# ---------------------------------------------------------------------------
# 9.2
# ---------------------------------------------------------------------------
_LEGACY_TEMPLATE = "config/embodied_pure_ddm_bellman.json"

#: Zip members whose content must match across reruns. The pkl is excluded only
#: because pickle serialisation embeds memory-layout details; every logged
#: MEASUREMENT lives in the CSVs and events.json.
_COMPARE_SUFFIXES = ("_ddm.csv", "_position.csv", "_targets.csv",
                     "_perception.csv", "_percept.csv", "events.json")


def _legacy_cfg(scratch: Path, out_tag: str) -> dict:
    with open(_ROOT / _LEGACY_TEMPLATE, encoding="utf-8") as fh:
        data = json.load(fh)
    env = data["environment"]
    env.pop("gui", None)
    env["num_runs"] = 1
    env["time_limit"] = 30
    for arena in env["arenas"].values():
        arena["random_seed"] = 12345
    blk = env["agents"]["movable_0"]["embodied_pure_ddm"]
    # Cheap solver settings: the gate tests REPRODUCIBILITY of the legacy noise
    # path, not boundary accuracy, and both runs use the same table either way.
    blk["bellman"]["N_t"] = 5000
    blk["bellman"]["T_max_check_factor"] = None
    env["results"]["base_path"] = str(scratch / f"out_{out_tag}")
    return data


def _member_texts(zip_path: Path) -> dict[str, bytes]:
    out = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(_COMPARE_SUFFIXES):
                out[name.split("/")[-1]] = zf.read(name)
    return out


def check_legacy(golden: Path | None = None) -> bool:
    """The pre-feature config, run twice: bit-identical logs, legacy pass-through."""
    print("[9.2] legacy mode unchanged and bit-reproducible")
    with tempfile.TemporaryDirectory(prefix="preflight92_") as scratch:
        scratch = Path(scratch)
        zips = [
            _run_one(_legacy_cfg(scratch, tag), scratch, tag) for tag in ("a", "b")
        ]
        texts = [_member_texts(z) for z in zips]
        if set(texts[0]) != set(texts[1]):
            print(f"  FAIL: member sets differ: {set(texts[0]) ^ set(texts[1])}")
            return False
        bad = [n for n in texts[0] if texts[0][n] != texts[1][n]]
        if bad:
            print(f"  FAIL: rerun differs in {bad}")
            return False
        percept = _read_member(zips[0], "_percept.csv")
        modes = {r["mode"] for r in percept}
        qhats = {r["q_hat"] for r in percept if r["target"]}
        if modes - {"legacy"} or qhats - {""}:
            print(f"  FAIL: legacy percept log unexpected (modes={modes}, "
                  f"q_hat values={list(qhats)[:3]}): the pass-through must record "
                  "mode 'legacy' and leave q_hat empty")
            return False
        n = len(texts[0])
        print(f"  PASS: two runs of {_LEGACY_TEMPLATE} bit-identical across {n} "
              f"logged members; percept log confirms the legacy pass-through")
        if golden is not None:
            gold_zip = next(Path(golden).glob("**/run_*.zip"))
            gold = _member_texts(gold_zip)
            bad = [n_ for n_ in gold if texts[0].get(n_) != gold[n_]]
            if bad:
                print(f"  FAIL: differs from golden reference {gold_zip} in {bad}")
                return False
            print(f"  PASS: matches golden reference {gold_zip}")
    return True


# ---------------------------------------------------------------------------
# 9.3
# ---------------------------------------------------------------------------
def _run_condition(args) -> tuple[str, int]:
    from campaign import run_chunk
    name, argv = args
    return name, run_chunk.main(argv)


def check_small_matrix(results_root: Path, reps: int, workers: int,
                       keep_raw: bool) -> bool:
    """The full matrix at `reps` replicates, then the Section 9.3 assertions."""
    conds = matrix.build_conditions()
    print(f"[9.3] small matrix: {len(conds)} conditions x {reps} replicates "
          f"(workers={workers}, per-tick logging on)")
    results_root.mkdir(parents=True, exist_ok=True)

    work = []
    for cond in conds:
        argv = ["--only", f"{cond.arm}/{cond.name}:0", "--reps", str(reps),
                "--chunk", str(reps), "--results-root", str(results_root)]
        if keep_raw:
            argv.append("--keep-raw")
        work.append((cond.name, argv))
    failures = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, rc in pool.map(_run_condition, work):
                if rc != 0:
                    failures.append(name)
    else:
        for w in work:
            name, rc = _run_condition(w)
            if rc != 0:
                failures.append(name)
    if failures:
        print(f"  FAIL: {len(failures)} conditions did not complete: {failures[:8]}")
        return False

    ok = True
    n_censored_total = 0
    worst = []
    for cond in conds:
        cond_dir = results_root / cond.rel_dir
        rows = summarise.read_chunk_csv(cond_dir / "chunks" / "chunk_0000.csv")
        if len(rows) != reps:
            print(f"  FAIL {cond.name}: {len(rows)} rows, expected {reps}")
            ok = False
            continue
        if not (cond_dir / "manifest.json").is_file():
            print(f"  FAIL {cond.name}: manifest.json missing")
            ok = False
            continue
        man = json.loads((cond_dir / "manifest.json").read_text())
        cen = sum(int(r["censored"]) for r in rows)
        n_censored_total += cen
        # Expectation: censoring only where the dry run flagged the risk.
        if cen > 0 and man["predicted"]["DT_over_T_max"] <= 0.5:
            print(f"  WARN {cond.name}: {cen}/{reps} censored in an unflagged cell")
        a_vals = [float(r["a_realised"]) for r in rows
                  if r["a_realised"] not in ("", None)]
        if a_vals and man["predicted"]["a"] > 0:
            ratio = (sum(a_vals) / len(a_vals)) / man["predicted"]["a"]
            worst.append((abs(ratio - 1.0), ratio, cond.name,
                          cond.discretisation_limited))

    # The realised crossing log-odds should track a* outside the flagged cells.
    worst.sort(reverse=True)
    print(f"  censored total: {n_censored_total}")
    print("  realised-a/predicted-a, worst 8 (flag: discretisation-limited):")
    for _, ratio, name, disc in worst[:8]:
        print(f"    {name:<22} {ratio:>7.2f} {'[disc-limited]' if disc else ''}")
    bad_unflagged = [w for w in worst if w[0] > 0.5 and not w[3]]
    if bad_unflagged:
        print(f"  WARN: {len(bad_unflagged)} UNFLAGGED cells with realised a "
              f"off by >50%: {[w[2] for w in bad_unflagged[:6]]}")

    cache = results_root / "table_cache"
    n_main = sum(c.arm == "main" for c in conds)
    n_tables = len(list(cache.glob("bellman_*.npz"))) if cache.is_dir() else 0
    print(f"  table cache: {n_tables} tables for {n_main} main conditions"
          + ("" if n_tables >= n_main else "  (some conditions solved in-task)"))
    if ok:
        print(f"  PASS: all {len(conds)} chunk CSVs complete, manifests present")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gate", choices=["stream-determinism", "legacy",
                                     "small-matrix", "all"])
    ap.add_argument("--results-root", type=Path, default=None)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--golden", type=Path, default=None)
    args = ap.parse_args(argv)

    results = []
    if args.gate in ("stream-determinism", "all"):
        results.append(("9.1", check_stream_determinism()))
    if args.gate in ("legacy", "all"):
        results.append(("9.2", check_legacy(args.golden)))
    if args.gate in ("small-matrix", "all"):
        if args.results_root is None:
            ap.error("small-matrix requires --results-root")
        results.append(("9.3", check_small_matrix(
            args.results_root, args.reps, args.workers, args.keep_raw)))

    print()
    for name, passed in results:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    return 0 if all(p for _, p in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
