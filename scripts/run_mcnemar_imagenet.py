#!/usr/bin/env python3
"""Rebuttal Experiment 2: McNemar paired tests on ImageNet-1K / ImageNet-V2.

Tests NETRA vs templates on paired per-image predictions (same test set),
answering x6Ro-W5's request for significance tests on the small large-scale
margins. Reuses run_imagenet.py's exact pipeline and CACHED descriptions
(zero LLM cost). Reproduced accuracies double as a sanity check vs Table 3.

Place in scripts/ of the NETRA repo. Examples:

    python scripts/run_mcnemar_imagenet.py --dataset imagenet \
        --data-dir /path/to/imagenet/val --clip-model ViT-L/14 --llm gpt-4o

    python scripts/run_mcnemar_imagenet.py --dataset imagenet-v2 --llm gpt-4o

Outputs: experiments/rebuttal_trials/mcnemar_<dataset>.json
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_imagenet import (  # noqa: E402
    load_imagenet, load_imagenet_v2, load_clip_model,
    encode_images_with_preprocess, encode_text_prompts,
    generate_descriptions, build_template_prompts, build_netra_prompts,
)

logger = logging.getLogger(__name__)


def mcnemar(correct_a, correct_b):
    """McNemar test on paired correctness vectors (a=baseline, b=method).

    Returns dict with discordant counts and two-sided p.
    b01: a correct, b wrong.  b10: a wrong, b correct.
    """
    a = correct_a.bool()
    b = correct_b.bool()
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "p_two_sided": 1.0, "method": "degenerate"}
    # Exact binomial if feasible, else continuity-corrected chi-square/normal
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue
        method = "exact_binomial"
    except Exception:
        z = (abs(b01 - b10) - 1) / math.sqrt(n)
        p = math.erfc(z / math.sqrt(2))
        method = "normal_approx_cc"
    return {"b01": b01, "b10": b10, "n_discordant": n,
            "p_two_sided": float(p), "method": method}


def preds_for(model, tokenizer, ppc, classnames, image_features, device, batch_classes=64):
    """Encode class embeddings (batched over classes to bound memory) and return argmax preds."""
    embs = encode_text_prompts(model, tokenizer, ppc, classnames, device)
    with torch.no_grad():
        sims = image_features.to(device) @ embs.T
        return sims.argmax(dim=1).cpu()


def main():
    parser = argparse.ArgumentParser(description="McNemar paired tests on ImageNet")
    parser.add_argument("--dataset", required=True, choices=["imagenet", "imagenet-v2"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--clip-model", type=str, default="ViT-L/14")
    parser.add_argument("--llm", type=str, default="gpt-4o")
    parser.add_argument("--llm-provider", type=str, default="openai")
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--configs", type=str, default="0.85/0.15,0.55/0.45",
                        help="NETRA alpha/beta configs to test vs templates")
    parser.add_argument("--include-cuple", action="store_true",
                        help="Also run NETRA-vs-CuPL+e paired test (informational)")
    parser.add_argument("--output-dir", type=str, default="experiments/rebuttal_trials")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    print(f"\n{'='*70}\n  McNEMAR — {args.dataset} | {args.clip_model} | {args.llm}\n{'='*70}\n")

    # ── Data + CLIP + image features (identical to run_imagenet.py) ──
    if args.dataset == "imagenet":
        if not args.data_dir:
            raise ValueError("ImageNet requires --data-dir (val ImageFolder or 'huggingface')")
        dataset, classnames, indices = load_imagenet(args.data_dir, args.val_size)
    else:
        dataset, classnames, indices = load_imagenet_v2(args.data_dir, args.val_size)

    model, preprocess, tokenizer = load_clip_model(args.clip_model, args.device)
    print("Encoding images (one-time)...")
    t0 = time.time()
    image_features, labels = encode_images_with_preprocess(
        model, preprocess, dataset, indices, args.device, args.batch_size)
    print(f"  {image_features.shape[0]} images in {time.time()-t0:.0f}s\n")

    # ── Cached descriptions (no LLM cost if cache present) ───────────
    descriptions, cost = generate_descriptions(classnames, args.llm, args.llm_provider)

    # ── Paired predictions ────────────────────────────────────────────
    results = {"dataset": args.dataset, "clip_model": args.clip_model,
               "llm": args.llm, "n_images": int(image_features.shape[0]),
               "accuracies": {}, "mcnemar_vs_templates": {}}

    print("Templates (80-ensemble)...")
    preds_tpl = preds_for(model, tokenizer, build_template_prompts(classnames),
                          classnames, image_features, args.device)
    correct_tpl = preds_tpl == labels
    acc_tpl = correct_tpl.float().mean().item()
    results["accuracies"]["templates"] = acc_tpl
    print(f"  acc = {acc_tpl:.4f}  (sanity check vs Table 3)")

    variants = {}
    for cfg in args.configs.split(","):
        a, b = (float(x) for x in cfg.strip().split("/"))
        key = f"netra_{cfg.strip()}"
        print(f"NETRA {a:.0%}/{b:.0%}...")
        preds = preds_for(model, tokenizer,
                          build_netra_prompts(classnames, descriptions, a, b),
                          classnames, image_features, args.device)
        correct = preds == labels
        acc = correct.float().mean().item()
        variants[key] = correct
        results["accuracies"][key] = acc
        m = mcnemar(correct_tpl, correct)
        results["mcnemar_vs_templates"][key] = m
        direction = "gain" if acc > acc_tpl else "loss/tie"
        print(f"  acc = {acc:.4f} (Δ={acc-acc_tpl:+.4f}, {direction})  "
              f"discordant: tpl-only-right={m['b01']}, netra-only-right={m['b10']}  "
              f"p={m['p_two_sided']:.2e} ({m['method']})")

    if args.include_cuple:
        from scripts.run_imagenet import build_cupl_prompts
        print("CuPL+e (uniform)...")
        preds_c = preds_for(model, tokenizer, build_cupl_prompts(classnames, descriptions),
                            classnames, image_features, args.device)
        correct_c = preds_c == labels
        results["accuracies"]["cuple"] = correct_c.float().mean().item()
        results["mcnemar_netra_vs_cuple"] = {
            k: mcnemar(correct_c, v) for k, v in variants.items()}
        print(f"  acc = {results['accuracies']['cuple']:.4f}")

    out = Path(args.output_dir) / f"mcnemar_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
