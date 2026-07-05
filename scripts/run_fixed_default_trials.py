#!/usr/bin/env python3
"""Rebuttal Experiment 1: 5 independent generation trials at the FIXED 55/45 default.

Addresses x6Ro-W5 ("multiple independent generation, confidence intervals, or
significance tests") for the deployable setting, complementing Table 8 which
only reports trials at per-dataset optimal weights.

Per trial: generate FRESH descriptions -> evaluate at 55/45 AND (optionally)
the per-dataset optimal weight on the SAME descriptions. Descriptions are
SAVED per trial (unlike run_variance.py) for full auditability.

Place this file in scripts/ of the NETRA repo. Example:

    python scripts/run_fixed_default_trials.py --dataset flowers102 \
        --clip-model ViT-L/14 --n-trials 5 --extra-weights 0.0/1.0

Outputs: experiments/rebuttal_trials/<dataset>_fixed_trials.json
"""

import argparse
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run import build_task_runner, build_task_spec
from scripts.run_weight_ablation import generate_descriptions, build_prompts_with_weights

logger = logging.getLogger(__name__)

T_CRIT_975 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def one_sample_t(deltas):
    """One-sample t-test of mean(delta) > 0 over trials. Returns (t, p_two_sided, ci95)."""
    n = len(deltas)
    mean = statistics.mean(deltas)
    if n < 2:
        return None, None, (None, None)
    sd = statistics.stdev(deltas)
    if sd == 0:
        return float("inf"), 0.0, (mean, mean)
    se = sd / math.sqrt(n)
    t = mean / se
    tc = T_CRIT_975.get(n - 1, 1.96)
    ci = (mean - tc * se, mean + tc * se)
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df=n - 1)
    except Exception:
        p = None  # report t only; p<0.05 iff |t| > tc
    return t, p, ci


def parse_weights(s):
    """'0.55/0.45' -> (0.55, 0.45)"""
    a, b = s.split("/")
    return float(a), float(b)


def main():
    parser = argparse.ArgumentParser(description="5-trial variance at fixed 55/45")
    parser.add_argument("--task", choices=["classification"], default="classification")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--val-size", type=int, default=10000)
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--llm-api-key", type=str)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--fixed-weights", type=str, default="0.55/0.45",
                        help="Primary (deployable) config, default 0.55/0.45")
    parser.add_argument("--extra-weights", type=str, default="",
                        help="Comma-separated extra configs evaluated on the SAME "
                             "descriptions, e.g. '0.7/0.3,0.0/1.0' (per-dataset optimal)")
    parser.add_argument("--output-dir", type=str, default="experiments/rebuttal_trials")
    parser.add_argument("--verbose", "-v", action="store_true")
    # Dummy args for build_task_spec compatibility
    parser.add_argument("--sam-checkpoint", type=str)
    parser.add_argument("--sam-model-type", type=str, default="vit_b")
    parser.add_argument("--gdino-config", type=str)
    parser.add_argument("--gdino-checkpoint", type=str)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    configs = [("fixed", parse_weights(args.fixed_weights))]
    if args.extra_weights.strip():
        for i, w in enumerate(args.extra_weights.split(",")):
            configs.append((f"extra_{w.strip()}", parse_weights(w.strip())))

    task_spec = build_task_spec(args)
    task_runner = build_task_runner(args, task_spec)

    print(f"\n{'='*70}")
    print(f"  FIXED-DEFAULT TRIALS — {args.n_trials} trials, {args.clip_model} on {args.dataset}")
    print(f"  Configs: {[f'{n}:{a:.0%}/{b:.0%}' for n,(a,b) in configs]}")
    print(f"{'='*70}")

    # Templates-only baseline (deterministic, once)
    print("\nTemplates-only baseline...")
    baseline = task_runner.evaluate(
        build_prompts_with_weights(task_spec.class_names, {}, 1.0, 0.0), task_spec
    ).primary_metric
    print(f"  baseline = {baseline:.4f}")

    trials, total_cost = [], 0.0
    for trial in range(args.n_trials):
        t0 = time.time()
        descriptions, desc_cost = generate_descriptions(task_spec, args.llm, args.llm_provider)
        cost = desc_cost.get("total_cost_usd", 0) if isinstance(desc_cost, dict) else 0
        total_cost += cost

        row = {"trial": trial, "cost_usd": cost, "descriptions": descriptions, "results": {}}
        for name, (a, b) in configs:
            prompts = build_prompts_with_weights(task_spec.class_names, descriptions, a, b)
            acc = task_runner.evaluate(prompts, task_spec).primary_metric
            row["results"][name] = {"alpha": a, "beta": b, "accuracy": acc,
                                    "delta_vs_templates": acc - baseline}
        row["duration_s"] = time.time() - t0
        trials.append(row)

        printable = "  ".join(f"{n}={row['results'][n]['accuracy']:.4f}"
                              f"({row['results'][n]['delta_vs_templates']:+.4f})"
                              for n, _ in configs)
        print(f"  trial {trial+1}/{args.n_trials}: {printable}  [{row['duration_s']:.0f}s]")

    # ── Summary per config ────────────────────────────────────────────
    summary = {}
    print(f"\n{'='*70}\n  SUMMARY (n={args.n_trials}, baseline={baseline:.4f})\n{'='*70}")
    for name, (a, b) in configs:
        accs = [t["results"][name]["accuracy"] for t in trials]
        deltas = [t["results"][name]["delta_vs_templates"] for t in trials]
        t_stat, p, ci = one_sample_t(deltas)
        s = {
            "alpha": a, "beta": b,
            "mean_accuracy": statistics.mean(accs),
            "std_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "mean_delta": statistics.mean(deltas),
            "min_delta": min(deltas),
            "all_beat_baseline": min(accs) > baseline,
            "n_beat_baseline": sum(x > baseline for x in accs),
            "t_stat": t_stat, "p_two_sided": p,
            "ci95_delta": ci,
        }
        summary[name] = s
        pstr = f"p={p:.4f}" if p is not None else f"(p<0.05 iff |t|>{T_CRIT_975.get(args.n_trials-1,1.96)})"
        print(f"  {name:>16} ({a:.0%}/{b:.0%}): {s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f}"
              f"  Δ={s['mean_delta']:+.4f}  beat {s['n_beat_baseline']}/{args.n_trials}"
              f"  t={t_stat:.2f} {pstr}  CI95=[{ci[0]:+.4f},{ci[1]:+.4f}]")

    out = Path(args.output_dir) / f"{args.dataset}_fixed_trials.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "dataset": args.dataset, "clip_model": args.clip_model,
            "val_size": args.val_size, "llm": args.llm, "n_trials": args.n_trials,
            "baseline_accuracy": baseline, "summary": summary,
            "total_cost_usd": total_cost, "trials": trials,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out}  (includes per-trial descriptions for audit)")


if __name__ == "__main__":
    main()
