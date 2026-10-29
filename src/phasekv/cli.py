"""Command-line interface: demo, audit one context, and benchmark."""

from __future__ import annotations

import argparse
import json

from phasekv.adaptive import PhaseAdaptivePolicy, merge_downgrade
from phasekv.attention import make_context
from phasekv.cache import cache_bytes
from phasekv.policies import POLICIES, apply_policy
from phasekv.probes import evaluate
from phasekv.scoring import BASELINE_NAMES, run_grid


def cmd_demo(args: argparse.Namespace) -> int:
    context = make_context(seed=args.seed, n_tokens=args.n_tokens)
    print(f"context: {context.n_heads} heads, {context.n_tokens} tokens, "
          f"{int(context.needle_mask.sum())} needles, keep={args.keep}")
    for name in BASELINE_NAMES + ["phase_adaptive"]:
        if name == "phase_adaptive":
            result = PhaseAdaptivePolicy(stride=args.stride).apply(context, args.keep)
            mask = result.keep_mask
            extra = f" restores={result.n_restores}"
        else:
            mask = apply_policy(name, context, args.keep, seed=args.seed)
            extra = ""
        metrics = evaluate(context, mask, stride=args.stride)
        print(f"{name:20s} recall={metrics['needle_recall']:.3f} "
              f"worst-phase={metrics['worst_phase_recall']:.3f} "
              f"err={metrics['output_rel_error']:.3f} "
              f"bytes={metrics['cache_bytes']}{extra}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    context = make_context(seed=args.seed, n_tokens=args.n_tokens)
    if args.policy == "phase_adaptive":
        result = PhaseAdaptivePolicy(stride=args.stride).apply(context, args.keep)
        mask = result.keep_mask
    else:
        mask = apply_policy(args.policy, context, args.keep, seed=args.seed)
    merged = None
    if args.merge:
        merged, _ = merge_downgrade(context, mask)
    metrics = evaluate(context, mask, stride=args.stride, merged_values=merged)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    rows = run_grid(
        n_tokens_list=[args.n_tokens],
        keep_fractions=[args.keep],
        seeds=list(range(args.seeds)),
        stride=args.stride,
    )
    print(f"{'policy':20s} {'recall':>7s} {'worst':>7s} {'mass':>7s} {'err':>7s}")
    for row in rows:
        print(f"{row['policy']:20s} {row['needle_recall_mean']:7.3f} "
              f"{row['worst_phase_recall_mean']:7.3f} {row['mass_retained_mean']:7.3f} "
              f"{row['output_rel_error_mean']:7.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="phasekv", description="Phase-aware KV-cache compression.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="Compare all policies on one context.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-tokens", type=int, default=512)
    p.add_argument("--keep", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=4)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("audit", help="Score one policy on one context as JSON.")
    p.add_argument("--policy", choices=BASELINE_NAMES + ["phase_adaptive"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-tokens", type=int, default=512)
    p.add_argument("--keep", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--merge", action="store_true")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("benchmark", help="Run the benchmark grid for one cell.")
    p.add_argument("--n-tokens", type=int, default=512)
    p.add_argument("--keep", type=float, default=0.25)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--stride", type=int, default=4)
    p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
