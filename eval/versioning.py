"""Eval run versioning — freeze snapshots so later improvements are comparable."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

RUNS_DIR_NAME = "runs"
MANIFEST_NAME = "MANIFEST.json"
LEADERBOARD_NAME = "LEADERBOARD.md"


def runs_root(reports_dir: Path) -> Path:
    root = reports_dir / RUNS_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_run_ids(reports_dir: Path) -> list[str]:
    root = runs_root(reports_dir)
    ids = [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(ids)


def next_run_id(reports_dir: Path, tag: str = "") -> str:
    """Allocate vNNN_YYYY-MM-DD[_tag]."""
    existing = list_run_ids(reports_dir)
    max_n = 0
    for name in existing:
        m = re.match(r"^v(\d+)", name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    n = max_n + 1
    slug = re.sub(r"[^a-z0-9]+", "_", (tag or "").lower()).strip("_")
    base = f"v{n:03d}_{date.today().isoformat()}"
    return f"{base}_{slug}" if slug else base


def run_dir(reports_dir: Path, run_id: str) -> Path:
    path = runs_root(reports_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Compact metrics for MANIFEST / leaderboard (no predictions dump)."""
    split = report.get("split")
    out: dict[str, Any] = {
        "split": split,
        "n": report.get("n"),
        "elapsed_s": report.get("elapsed_s"),
        "use_rag": report.get("use_rag"),
        "backend": report.get("backend"),
        "model": report.get("model"),
    }
    if split == "eval":
        out.update(
            {
                "exact_match": report.get("exact_match"),
                "token_f1": report.get("token_f1"),
                "contains_gold": report.get("contains_gold"),
            }
        )
    elif split == "redteam":
        out["pass_rate"] = report.get("pass_rate")
    return out


def load_manifest(run_path: Path) -> dict[str, Any]:
    path = run_path / MANIFEST_NAME
    if not path.exists():
        return {"run_id": run_path.name, "artifacts": {}, "metrics": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(run_path: Path, manifest: dict[str, Any]) -> Path:
    path = run_path / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def record_artifact(
    reports_dir: Path,
    run_id: str,
    *,
    filename: str,
    report: dict[str, Any],
    notes: str = "",
) -> Path:
    """Write report JSON into runs/<run_id>/ and update MANIFEST."""
    dest_dir = run_dir(reports_dir, run_id)
    dest = dest_dir / filename
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = load_manifest(dest_dir)
    manifest["run_id"] = run_id
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    if notes and not manifest.get("notes"):
        manifest["notes"] = notes
    artifacts = manifest.setdefault("artifacts", {})
    artifacts[filename] = {
        "written_at": manifest["updated_at"],
        "summary": summarize_report(report),
    }
    metrics = manifest.setdefault("metrics", {})
    metrics[filename] = summarize_report(report)
    save_manifest(dest_dir, manifest)
    return dest


def rebuild_leaderboard(reports_dir: Path) -> Path:
    """Rewrite reports/LEADERBOARD.md from all run manifests."""
    lines = [
        "# Eval leaderboard",
        "",
        "Frozen runs live under `reports/runs/`. Newer rows should beat older ones on the same split.",
        "",
        "## Factoid (`eval` split)",
        "",
        "| Run | RAG | n | Exact | Token F1 | Contains gold | Notes |",
        "|-----|-----|---|-------|----------|---------------|-------|",
    ]
    fact_rows: list[tuple[str, dict[str, Any], str]] = []
    red_rows: list[tuple[str, dict[str, Any], str]] = []

    for run_id in list_run_ids(reports_dir):
        man = load_manifest(runs_root(reports_dir) / run_id)
        note = (man.get("notes") or "").replace("\n", " ")[:60]
        for fname, summary in (man.get("metrics") or {}).items():
            low = fname.lower()
            if "smoke" in low:
                continue  # keep smoke in the folder; omit from headline table
            label = f"{fname}" + (f" — {note}" if note else "")
            if summary.get("split") == "eval":
                fact_rows.append((run_id, summary, label))
            elif summary.get("split") == "redteam":
                red_rows.append((run_id, summary, label))

    if not fact_rows:
        lines.append("| — | — | — | — | — | — | no runs yet |")
    else:
        for run_id, s, note in fact_rows:
            lines.append(
                "| `{run}` | {rag} | {n} | {em} | {f1} | {cg} | {note} |".format(
                    run=run_id,
                    rag=s.get("use_rag"),
                    n=s.get("n"),
                    em=s.get("exact_match"),
                    f1=s.get("token_f1"),
                    cg=s.get("contains_gold"),
                    note=note,
                )
            )

    lines += [
        "",
        "## Red-team",
        "",
        "| Run | n | Pass rate | Notes |",
        "|-----|---|-----------|-------|",
    ]
    if not red_rows:
        lines.append("| — | — | — | no runs yet |")
    else:
        for run_id, s, note in red_rows:
            lines.append(
                "| `{run}` | {n} | {pr} | {note} |".format(
                    run=run_id,
                    n=s.get("n"),
                    pr=s.get("pass_rate"),
                    note=note,
                )
            )

    lines += [
        "",
        "## How to add a new run",
        "",
        "```bash",
        "python -m eval.run_eval --split eval --use-rag 0 --api-url $URL --run-id auto --tag after_rag \\",
        "  --out reports/baseline_sft.json",
        "python -m scripts.compare_eval_runs v001_2026-11-23_pre_fix auto",
        "```",
        "",
        "Use `--run-id auto` to allocate the next `vNNN_...` folder. Pass an explicit id to append into an existing run.",
        "",
    ]
    path = reports_dir / LEADERBOARD_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
