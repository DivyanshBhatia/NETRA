#!/usr/bin/env python3
"""Rebuttal Experiment 2: McNemar paired tests on ImageNet-1K / ImageNet-V2.

v2 — self-contained HuggingFace loading that downloads ONLY the ~6.7 GB
validation parquet shards (the repo's load_imagenet_huggingface triggers a
~150 GB all-splits download). No patch to run_imagenet.py needed.

Place in scripts/ of the NETRA repo (replacing the previous version).

    python scripts/run_mcnemar_imagenet.py --dataset imagenet \
        --data-dir huggingface --clip-model ViT-L/14 --llm gpt-4o

    # or with a local ImageFolder val:
    python scripts/run_mcnemar_imagenet.py --dataset imagenet \
        --data-dir /path/to/imagenet/val --clip-model ViT-L/14 --llm gpt-4o

Sanity gate (with the ORIGINAL cached descriptions in data/descriptions/):
templates acc ~= 0.7286, netra_0.85/0.15 ~= 0.7333 (Table 3).

Outputs: experiments/rebuttal_trials/mcnemar_<dataset>.json
"""

import argparse
import io
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_imagenet import (  # noqa: E402
    load_imagenet, load_imagenet_v2, load_clip_model,
    encode_images_with_preprocess, encode_text_prompts,
    generate_descriptions, build_template_prompts, build_netra_prompts,
    download_imagenet_classnames,
)

logger = logging.getLogger(__name__)


# ── Val-only HuggingFace loader (~6.7 GB, not 150 GB) ─────────────────────
def load_imagenet_val_only_hf(val_size=None):
    """Load ONLY ImageNet-1K validation shards from HuggingFace.

    Uses a data_files glob so the `datasets` library never prepares the
    train/test splits. Requires `hf auth login` + accepted license at
    https://huggingface.co/datasets/ILSVRC/imagenet-1k
    """
    from datasets import load_dataset
    from PIL import Image as PILImage

    print("  Loading ImageNet val (val-only parquet shards) from HuggingFace...")
    ds = load_dataset(
        "parquet",
        data_files={"validation":
                    "hf://datasets/ILSVRC/imagenet-1k/data/val-*.parquet"},
        split="validation",
    )

    classnames = download_imagenet_classnames()
    if classnames is None:
        try:  # parquet metadata usually preserves the ClassLabel feature
            classnames = [ds.features["label"].int2str(i).split(",")[0].strip()
                          for i in range(1000)]
        except Exception as e:
            raise RuntimeError(
                "Could not obtain ImageNet classnames (helper download failed "
                "and parquet lacks label names). Check network access.") from e

    if val_size and val_size < len(ds):
        indices = np.random.RandomState(42).permutation(len(ds))[:val_size]
    else:
        indices = np.arange(len(ds))

    def _to_pil(img):
        if isinstance(img, PILImage.Image):
            return img.convert("RGB")
        if isinstance(img, dict):  # {'bytes': ..., 'path': ...} raw form
            if img.get("bytes") is not None:
                return PILImage.open(io.BytesIO(img["bytes"])).convert("RGB")
            if img.get("path"):
                return PILImage.open(img["path"]).convert("RGB")
        if isinstance(img, (str, Path)):
            return PILImage.open(img).convert("RGB")
        raise TypeError(f"Unrecognized image payload type: {type(img)}")

    class ValOnlyWrapper:
        """Minimal ImageFolder-like interface: dataset[idx] -> (PIL, label)."""
        def __init__(self, hf_ds):
            self.ds = hf_ds
            self.classes = classnames

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, idx):
            item = self.ds[int(idx)]
            return _to_pil(item["image"]), int(item["label"])

    print(f"  Loaded ImageNet val: {len(ds)} images, {len(classnames)} classes")
    return ValOnlyWrapper(ds), classnames, indices


# ── McNemar ───────────────────────────────────────────────────────────────
def mcnemar(correct_a, correct_b):
    """McNemar test on paired correctness vectors (a=baseline, b=method).

    b01: a correct, b wrong.  b10: a wrong, b correct.
    """
    a = correct_a.bool()
    b = correct_b.bool()
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    n = b01 + b10
    if n == 0:
        return {"b01": 0, "b10": 0, "n_discordant": 0,
                "p_two_sided": 1.0, "method": "degenerate"}
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


def preds_for(model, tokenizer, ppc, classnames, image_features, device):
    """Encode class embeddings and return argmax predictions."""
    embs = encode_text_prompts(model, tokenizer, ppc, classnames, device)
    with torch.no_grad():
        sims = image_features.to(device) @ embs.T
        return sims.argmax(dim=1).cpu()


def main():
    parser = argparse.ArgumentParser(description="McNemar paired tests on ImageNet")
    parser.add_argument("--dataset", required=True, choices=["imagenet", "imagenet-v2"])
    parser.add_argument("--data-dir", type=str, default="huggingface",
                        help="'huggingface' (val-only download) or local ImageFolder val dir")
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

    # ── Data ──────────────────────────────────────────────────────────
    if args.dataset == "imagenet":
        if args.data_dir in (None, "huggingface"):
            dataset, classnames, indices = load_imagenet_val_only_hf(args.val_size)
        else:
            dataset, classnames, indices = load_imagenet(args.data_dir, args.val_size)
    else:
        dataset, classnames, indices = load_imagenet_v2(args.data_dir, args.val_size)

    # ── Descriptions cache check (BEFORE paying for encoding) ────────
    cache = Path(__file__).parent.parent / "data" / "descriptions" / \
        f"descriptions_1000classes_{args.llm}.json"
    if not cache.exists():
        print(f"\n  *** WARNING: {cache} not found — descriptions will be "
              f"REGENERATED (~$1, needs API key) and will DIFFER from the "
              f"paper's Table 3 run. The sanity gate will not match. ***\n")

    model, preprocess, tokenizer = load_clip_model(args.clip_model, args.device)
    print("Encoding images (one-time)...")
    t0 = time.time()
    image_features, labels = encode_images_with_preprocess(
        model, preprocess, dataset, indices, args.device, args.batch_size)
    print(f"  {image_features.shape[0]} images in {time.time()-t0:.0f}s\n")

    descriptions, _cost = generate_descriptions(classnames, args.llm, args.llm_provider)

    # ── Paired predictions ────────────────────────────────────────────
    results = {"dataset": args.dataset, "clip_model": args.clip_model,
               "llm": args.llm, "n_images": int(image_features.shape[0]),
               "descriptions_cache_present": cache.exists(),
               "accuracies": {}, "mcnemar_vs_templates": {}}

    print("Templates (80-ensemble)...")
    preds_tpl = preds_for(model, tokenizer, build_template_prompts(classnames),
                          classnames, image_features, args.device)
    correct_tpl = preds_tpl == labels
    acc_tpl = correct_tpl.float().mean().item()
    results["accuracies"]["templates"] = acc_tpl
    print(f"  acc = {acc_tpl:.4f}  (sanity gate: ~0.7286 on full IN-1K val)")

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
        print(f"  acc = {acc:.4f} (Δ={acc-acc_tpl:+.4f})  "
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
