"""Compare two versioned eval runs.

Usage:
  python -m scripts.compare_eval_runs v001_2026-11-23_pre_fix v002_2026-11-24_after_fix
  python -m scripts.compare_eval_runs v001_2026-11-23_pre_fix latest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.versioning import list_run_ids, load_manifest, runs_root


def _resolve(reports: Path, run_id: str) -> str:
    if run_id == "latest":
        ids = list_run_ids(reports)
        if not ids:
            raise SystemExit("No runs found under reports/runs/")
        return ids[-1]
    return run_id


def _pick_metric(metrics: dict, split: str) -> tuple[str, dict] | None:
    for fname, summary in metrics.items():
        if summary.get("split") == split:
            return fname, summary
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two reports/runs/* snapshots")
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--reports-dir", type=str, default=str(ROOT / "reports"))
    args = parser.parse_args()

    reports = Path(args.reports_dir)
    before_id = _resolve(reports, args.before)
    after_id = _resolve(reports, args.after)
    before = load_manifest(runs_root(reports) / before_id)
    after = load_manifest(runs_root(reports) / after_id)

    print(f"Before: {before_id}")
    print(f"After:  {after_id}")
    print()

    for split, keys in (
        ("eval", ["exact_match", "token_f1", "contains_gold"]),
        ("redteam", ["pass_rate"]),
    ):
        b = _pick_metric(before.get("metrics") or {}, split)
        a = _pick_metric(after.get("metrics") or {}, split)
        print(f"## {split}")
        if not b and not a:
            print("  (missing in both)")
            print()
            continue
        if not b or not a:
            print(f"  before={b[1] if b else None}")
            print(f"  after ={a[1] if a else None}")
            print()
            continue
        _, bs = b
        _, as_ = a
        for k in keys:
            bv, av = bs.get(k), as_.get(k)
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                delta = av - bv
                sign = "+" if delta >= 0 else ""
                print(f"  {k}: {bv} → {av} ({sign}{delta:.4f})")
            else:
                print(f"  {k}: {bv} → {av}")
        print()

    print("Raw manifests:")
    print(json.dumps({"before": before.get("metrics"), "after": after.get("metrics")}, indent=2))


if __name__ == "__main__":
    main()
