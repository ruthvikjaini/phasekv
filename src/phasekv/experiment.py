"""Experiment runner: benchmark grid + ablations, JSON outputs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from phasekv.scoring import BASELINE_NAMES, run_grid, score_policy


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run(config: dict) -> dict:
    grid_cfg = config["grid"]
    rows = run_grid(
        n_tokens_list=grid_cfg["n_tokens"],
        keep_fractions=grid_cfg["keep_fractions"],
        seeds=grid_cfg["seeds"],
        stride=config.get("stride", 4),
    )
    # Ablations at the headline cell: merge on/off, rescue on/off proxy.
    ablations: dict = {}
    head = config["ablations"]
    for name in ("phase_adaptive",):
        ablations[f"{name}_merge"] = score_policy(
            name,
            head["n_tokens"],
            head["keep_fraction"],
            head["seeds"],
            stride=config.get("stride", 4),
            merge=True,
        )
        ablations[f"{name}_nomerge"] = score_policy(
            name,
            head["n_tokens"],
            head["keep_fraction"],
            head["seeds"],
            stride=config.get("stride", 4),
            merge=False,
        )
    return {"grid": rows, "ablations": ablations, "config": config}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the phasekv benchmark grid.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--out", default="results/experiment.json")
    parser.add_argument("--measured-out", default="results/measured.json")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run(config)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    # measured.json: headline aggregates only, for README cross-footing.
    def cell(policy: str, keep: float) -> dict:
        for row in result["grid"]:
            if row["policy"] == policy and row["keep_fraction"] == keep and row["n_tokens"] == 512:
                return row
        raise KeyError(policy)

    measured = {
        "headline": {
            "n_tokens": 512,
            "keep_fraction": 0.25,
            "policies": {
                name: {
                    "needle_recall": cell(name, 0.25)["needle_recall_mean"],
                    "worst_phase_recall": cell(name, 0.25)["worst_phase_recall_mean"],
                    "mass_retained": cell(name, 0.25)["mass_retained_mean"],
                    "output_rel_error": cell(name, 0.25)["output_rel_error_mean"],
                }
                for name in BASELINE_NAMES + ["phase_adaptive", "phase_adaptive_norescue"]
            },
        },
        "headline_group_recall": {
            name: {
                "group1": cell(name, 0.25).get("needle_recall_group1_mean"),
                "group2": cell(name, 0.25).get("needle_recall_group2_mean"),
            }
            for name in BASELINE_NAMES + ["phase_adaptive", "phase_adaptive_norescue"]
        },
        "ablations": result["ablations"],
        "config": config,
    }
    Path(args.measured_out).write_text(json.dumps(measured, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out} ({len(result['grid'])} grid cells)")
    print(f"wrote {args.measured_out}")


if __name__ == "__main__":
    main()
