"""Copy existing report JSON files into a versioned runs/<id>/ snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.run_eval import score_redteam_prediction
from eval.versioning import next_run_id, rebuild_leaderboard, record_artifact, run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="auto")
    parser.add_argument("--tag", default="pre_fix")
    parser.add_argument(
        "--notes",
        default="Frozen baseline from first ngrok API eval (2026-11-23). Live serve lacked /health.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["baseline_sft.json", "redteam.json", "smoke.json"],
    )
    args = parser.parse_args()

    reports = ROOT / "reports"
    run_id = args.run_id
    if run_id == "auto":
        run_id = next_run_id(reports, tag=args.tag)
        print(f"Allocated {run_id}")

    dest = run_dir(reports, run_id)
    for name in args.files:
        src = reports / name
        if not src.exists():
            print(f"skip missing {src}")
            continue
        original = json.loads(src.read_text(encoding="utf-8"))
        record_artifact(reports, run_id, filename=name, report=original, notes=args.notes)
        print(f"  archived {name}")

        if name == "redteam.json":
            original_rate = original.get("pass_rate")
            rescored = dict(original)
            results = []
            passes = 0
            for row in rescored.get("results") or []:
                ok = score_redteam_prediction(
                    prediction=row.get("prediction") or "",
                    expected_behavior=row.get("expected_behavior") or "refuse",
                    refused=bool(row.get("refused")),
                )
                passes += int(ok)
                results.append({**row, "pass": ok, "pass_original": row.get("pass")})
            n = max(len(results), 1)
            rescored["results"] = results
            rescored["pass_rate_original_inflated"] = original_rate
            rescored["pass_rate"] = round(passes / n, 4)
            record_artifact(
                reports,
                run_id,
                filename="redteam_rescored.json",
                report=rescored,
                notes=args.notes,
            )
            print(f"  redteam original pass_rate={original_rate} → rescored={rescored['pass_rate']}")

    (dest / "NOTES.md").write_text(
        "\n".join(
            [
                f"# Run `{run_id}`",
                "",
                args.notes,
                "",
                "## Files",
                "",
                "- `baseline_sft.json` — 80 factoid Qs via ngrok `/ask`, RAG off",
                "- `redteam.json` — original scores (inflated; see CURRENT_STATUS.md)",
                "- `redteam_rescored.json` — same predictions, fixed scorer",
                "- `smoke.json` — 1-question smoke test",
                "",
                "Use this run as the **before** side in:",
                "",
                "```bash",
                f"python -m scripts.compare_eval_runs {run_id} latest",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    rebuild_leaderboard(reports)
    print(f"Wrote {dest}")
    print("Updated reports/LEADERBOARD.md")


if __name__ == "__main__":
    main()
