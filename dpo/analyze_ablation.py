"""
analyze_ablation.py
===================
Collect and summarize DPO ablation experiment results.

Reads trainer_state.json from each experiment output directory,
extracts training metrics, and produces a comparison table + JSON report.

Usage:
    python analyze_ablation.py \
        --output_base ./output/ablation \
        --results_file ./output/ablation/ablation_results.json
"""

import json
import re
import sys
from pathlib import Path

import fire


def parse_experiment_name(dirname: str) -> dict:
    """Extract temp and beta from directory name like 'dpo_temp1.2_beta0.1'."""
    m = re.match(r"dpo_temp([0-9.]+)_beta([0-9.]+)", dirname)
    if not m:
        return {}
    return {"temperature": float(m.group(1)), "beta": float(m.group(2))}


def load_trainer_state(experiment_dir: Path) -> dict:
    """Load trainer_state.json and extract key metrics."""
    state_file = experiment_dir / "trainer_state.json"
    if not state_file.exists():
        return {}

    state = json.loads(state_file.read_text())
    log_history = state.get("log_history", [])

    train_logs = [e for e in log_history if "loss" in e and "eval_loss" not in e]
    eval_logs = [e for e in log_history if "eval_loss" in e]

    result = {
        "total_steps": state.get("global_step", 0),
        "best_metric": state.get("best_metric"),
    }

    if train_logs:
        result["final_train_loss"] = train_logs[-1].get("loss")
        result["train_losses"] = [(e.get("step", 0), e.get("loss")) for e in train_logs]

    if eval_logs:
        result["final_eval_loss"] = eval_logs[-1].get("eval_loss")
        result["best_eval_loss"] = min(e.get("eval_loss", float("inf")) for e in eval_logs)
        result["eval_losses"] = [(e.get("step", 0), e.get("eval_loss")) for e in eval_logs]

        final_eval = eval_logs[-1]
        for key in ("eval_rewards/chosen", "eval_rewards/rejected",
                     "eval_rewards/margins", "eval_rewards/accuracies",
                     "eval_logps/chosen", "eval_logps/rejected"):
            if key in final_eval:
                result[key.replace("/", "_")] = final_eval[key]

    return result


def count_data_pairs(data_dir: Path) -> dict:
    """Count preference pairs in each generated dataset."""
    counts = {}
    for f in sorted(data_dir.glob("dpo_pairs_*.json")):
        text = f.read_text().strip()
        n = len(text.splitlines()) if not text.startswith("[") else len(json.loads(text))
        counts[f.name] = n
    return counts


def main(
    output_base: str = "./output/ablation",
    results_file: str = "./output/ablation/ablation_results.json",
):
    base = Path(output_base)
    if not base.exists():
        sys.exit(f"Output directory not found: {output_base}")

    experiments = []
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        params = parse_experiment_name(exp_dir.name)
        if not params:
            continue

        metrics = load_trainer_state(exp_dir)
        if not metrics:
            print(f"  Warning: no trainer_state.json in {exp_dir.name}, skipping")
            continue

        experiments.append({
            "name": exp_dir.name,
            **params,
            **metrics,
        })

    if not experiments:
        print("No completed experiments found.")
        return

    # ── Data pair counts ──────────────────────────────────────────────────────
    data_dir = base.parent / "dpo" / "train_set" / "ablation"
    if not data_dir.exists():
        data_dir = Path("./dpo/train_set/ablation")
    data_counts = count_data_pairs(data_dir) if data_dir.exists() else {}

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  DPO ABLATION RESULTS SUMMARY")
    print("=" * 85)
    print(f"{'Experiment':<30} {'Temp':>5} {'Beta':>6} {'Train Loss':>11} "
          f"{'Eval Loss':>10} {'Best Eval':>10}")
    print("-" * 85)

    for exp in sorted(experiments, key=lambda e: (e["temperature"], e["beta"])):
        print(f"{exp['name']:<30} {exp['temperature']:>5.1f} {exp['beta']:>6.2f} "
              f"{exp.get('final_train_loss', 'N/A'):>11} "
              f"{exp.get('final_eval_loss', 'N/A'):>10} "
              f"{exp.get('best_eval_loss', 'N/A'):>10}")

    # ── Reward margins (if available) ─────────────────────────────────────────
    has_margins = any("eval_rewards_margins" in e for e in experiments)
    if has_margins:
        print("\n" + "-" * 85)
        print(f"{'Experiment':<30} {'Margin':>8} {'Accuracy':>10} "
              f"{'Chosen R':>10} {'Rejected R':>10}")
        print("-" * 85)
        for exp in sorted(experiments, key=lambda e: (e["temperature"], e["beta"])):
            print(f"{exp['name']:<30} "
                  f"{exp.get('eval_rewards_margins', 'N/A'):>8} "
                  f"{exp.get('eval_rewards_accuracies', 'N/A'):>10} "
                  f"{exp.get('eval_rewards_chosen', 'N/A'):>10} "
                  f"{exp.get('eval_rewards_rejected', 'N/A'):>10}")

    print("=" * 85)

    # ── Best configuration ────────────────────────────────────────────────────
    valid = [e for e in experiments if e.get("best_eval_loss") is not None]
    if valid:
        best = min(valid, key=lambda e: e["best_eval_loss"])
        print(f"\n  Best by eval loss: {best['name']}")
        print(f"    temperature={best['temperature']}, beta={best['beta']}, "
              f"best_eval_loss={best['best_eval_loss']}")

    margin_valid = [e for e in experiments if e.get("eval_rewards_margins") is not None]
    if margin_valid:
        best_margin = max(margin_valid, key=lambda e: e["eval_rewards_margins"])
        print(f"  Best by reward margin: {best_margin['name']}")
        print(f"    temperature={best_margin['temperature']}, beta={best_margin['beta']}, "
              f"margin={best_margin['eval_rewards_margins']}")

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "experiments": experiments,
        "data_counts": data_counts,
    }
    if valid:
        report["best_by_eval_loss"] = {
            "name": best["name"],
            "temperature": best["temperature"],
            "beta": best["beta"],
            "best_eval_loss": best["best_eval_loss"],
        }
    if margin_valid:
        report["best_by_margin"] = {
            "name": best_margin["name"],
            "temperature": best_margin["temperature"],
            "beta": best_margin["beta"],
            "margin": best_margin["eval_rewards_margins"],
        }

    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    Path(results_file).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Full report saved → {results_file}\n")


if __name__ == "__main__":
    fire.Fire(main)
