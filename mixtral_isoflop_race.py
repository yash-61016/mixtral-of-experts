"""
Mixtral of Experts - MoE vs Dense Iso-Compute Training Race
===========================================================
Paper: Mixtral of Experts (arXiv: 2401.04088)
What this implements: A FLOP-budgeted training race between a dense Transformer
                      and a sparse MoE Transformer to compare validation
                      perplexity at equal cumulative compute.
Hardware: RTX 3090, 24GB VRAM (also runs on CPU with longer runtime)
Time to run: ~20 minutes for smoke config, ~2-3 hours for full config

Dependencies (pinned):
    torch==2.1.0          # pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    matplotlib==3.7.1     # pip install matplotlib==3.7.1
    numpy==1.24.3         # pip install numpy==1.24.3
    datasets==2.19.0      # optional, for TinyStories/WikiText-2 loading

Run:
    python mixtral-of-experts/mixtral_isoflop_race.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "theme"))
from deep_sea_theme import PALETTE, EXTENDED, add_source_label, annotate_points, apply_theme

from moe_core import (
    FLOPCounter,
    DenseTransformerLM,
    ModelConfig,
    MoETransformerLM,
    build_train_val_datasets,
    causal_lm_loss,
    estimate_step_flops,
    load_text_corpus,
    seed_everything,
)

apply_theme()


@dataclass
class RaceConfig:
    """Configuration for dense vs MoE FLOP-budget race."""

    dataset: str = "tinystories"
    max_chars: int = 220_000
    seq_len: int = 96
    batch_size: int = 16
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 3
    dense_d_ff: int = 768
    n_experts: int = 8
    k: int = 2
    eval_interval_flops: int = int(4e10)
    total_flops: int = int(6e11)
    max_steps: int = 6_000
    val_batches: int = 30
    lr: float = 3e-4
    seed: int = 7

    @property
    def d_ff_per_expert(self) -> int:
        return self.dense_d_ff // self.n_experts


@dataclass
class LossCurve:
    """Series recorded at FLOP checkpoints."""

    name: str
    cumulative_flops: List[float] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_perplexity: List[float] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    tokens_seen: List[int] = field(default_factory=list)
    wall_seconds: List[float] = field(default_factory=list)


def parse_args() -> RaceConfig:
    """Parse CLI arguments into RaceConfig."""
    parser = argparse.ArgumentParser(description="Dense vs MoE race at equal FLOP budget")
    parser.add_argument("--dataset", type=str, default="tinystories", choices=["tinystories", "wikitext2"])
    parser.add_argument("--max-chars", type=int, default=220_000)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dense-d-ff", type=int, default=768)
    parser.add_argument("--n-experts", type=int, default=8)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--eval-interval-flops", type=float, default=4e10)
    parser.add_argument("--total-flops", type=float, default=6e11)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--val-batches", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return RaceConfig(
        dataset=args.dataset,
        max_chars=args.max_chars,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dense_d_ff=args.dense_d_ff,
        n_experts=args.n_experts,
        k=args.k,
        eval_interval_flops=int(args.eval_interval_flops),
        total_flops=int(args.total_flops),
        max_steps=args.max_steps,
        val_batches=args.val_batches,
        lr=args.lr,
        seed=args.seed,
    )


def detect_device() -> torch.device:
    """Print hardware details and return compute device."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {vram_gb:.1f} GB")
        if vram_gb < 8:
            print(f"Warning: low VRAM ({vram_gb:.1f} GB). Reduce batch size or seq_len if OOM occurs.")
    else:
        dev = torch.device("cpu")
        print("No GPU found. Running on CPU; runtime will be slower.")
    print(f"Device: {dev}\n")
    return dev


def evaluate_model(
    model: torch.nn.Module,
    val_data,
    batch_size: int,
    val_batches: int,
    generator: torch.Generator,
) -> Dict[str, float]:
    """Compute mean val loss and perplexity over a fixed number of batches."""
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for _ in range(val_batches):
            x, y = val_data.sample_batch(batch_size, generator=generator)
            out = model(x)
            loss = causal_lm_loss(out["logits"], y)
            losses.append(loss.item())
    mean_loss = float(sum(losses) / len(losses))
    return {"val_loss": mean_loss, "val_ppl": float(math.exp(mean_loss))}


def train_to_flop_budget(
    name: str,
    model: torch.nn.Module,
    train_data,
    val_data,
    cfg: RaceConfig,
    optimizer: torch.optim.Optimizer,
    step_flops: int,
    data_seed: int,
) -> LossCurve:
    """Train one model until cumulative FLOP budget is reached."""
    counter = FLOPCounter()
    curve = LossCurve(name=name)
    next_eval = cfg.eval_interval_flops
    start = time.perf_counter()
    train_gen = torch.Generator().manual_seed(data_seed)
    val_gen = torch.Generator().manual_seed(data_seed + 10_000)

    for step in range(1, cfg.max_steps + 1):
        if counter.total >= cfg.total_flops:
            break

        model.train()
        x, y = train_data.sample_batch(cfg.batch_size, generator=train_gen)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = causal_lm_loss(out["logits"], y)
        loss.backward()
        optimizer.step()

        counter.add(step_flops)

        if counter.total >= next_eval or counter.total >= cfg.total_flops:
            metrics = evaluate_model(
                model=model,
                val_data=val_data,
                batch_size=cfg.batch_size,
                val_batches=cfg.val_batches,
                generator=val_gen,
            )
            elapsed = time.perf_counter() - start
            curve.cumulative_flops.append(float(counter.total))
            curve.train_loss.append(float(loss.item()))
            curve.val_perplexity.append(metrics["val_ppl"])
            curve.steps.append(step)
            curve.tokens_seen.append(step * cfg.batch_size * cfg.seq_len)
            curve.wall_seconds.append(float(elapsed))

            print(
                f"[{name:5}] step={step:4d} | flops={counter.total:>13,d} "
                f"| train_loss={loss.item():.4f} | val_ppl={metrics['val_ppl']:.2f}"
            )
            next_eval += cfg.eval_interval_flops

    return curve


def write_curve_csv(out_path: Path, dense_curve: LossCurve, moe_curve: LossCurve) -> None:
    """Persist checkpoint metrics for later analysis."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "cumulative_flops",
            "step",
            "tokens_seen",
            "train_loss",
            "val_perplexity",
            "wall_seconds",
        ])

        for curve in [dense_curve, moe_curve]:
            for i in range(len(curve.cumulative_flops)):
                writer.writerow(
                    [
                        curve.name,
                        int(curve.cumulative_flops[i]),
                        curve.steps[i],
                        curve.tokens_seen[i],
                        f"{curve.train_loss[i]:.6f}",
                        f"{curve.val_perplexity[i]:.6f}",
                        f"{curve.wall_seconds[i]:.3f}",
                    ]
                )


def plot_isoflop_curves(dense_curve: LossCurve, moe_curve: LossCurve, out_dir: Path) -> None:
    """Create loss and perplexity curves versus cumulative FLOPs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x_dense = [v / 1e11 for v in dense_curve.cumulative_flops]
    x_moe = [v / 1e11 for v in moe_curve.cumulative_flops]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        x_dense,
        dense_curve.train_loss,
        "o-",
        color=PALETTE["steel"],
        label="Dense train loss",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    ax.plot(
        x_moe,
        moe_curve.train_loss,
        "s--",
        color=PALETTE["slate"],
        label="MoE train loss",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    ax.set_title("Dense vs MoE: train loss by cumulative FLOPs")
    ax.set_xlabel("Cumulative FLOPs (1e11)")
    ax.set_ylabel("Train loss")
    ax.legend(loc="best")
    add_source_label(ax)
    fig.savefig(out_dir / "isoflop_train_loss.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        x_dense,
        dense_curve.val_perplexity,
        "o-",
        color=PALETTE["steel"],
        label="Dense val perplexity",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    ax.plot(
        x_moe,
        moe_curve.val_perplexity,
        "s--",
        color=EXTENDED["success"],
        label="MoE val perplexity",
        markeredgecolor="white",
        markeredgewidth=1.2,
    )
    annotate_points(ax, x_dense, dense_curve.val_perplexity, fmt="{:.1f}", color=PALETTE["steel"])
    annotate_points(ax, x_moe, moe_curve.val_perplexity, fmt="{:.1f}", color=EXTENDED["success"])
    ax.set_title("Dense vs MoE: val perplexity by cumulative FLOPs")
    ax.set_xlabel("Cumulative FLOPs (1e11)")
    ax.set_ylabel("Validation perplexity")
    ax.legend(loc="best")
    add_source_label(ax)
    fig.savefig(out_dir / "isoflop_val_perplexity.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def print_final_table(dense_curve: LossCurve, moe_curve: LossCurve) -> None:
    """Print final FLOP-matched endpoint comparison."""
    if not dense_curve.cumulative_flops or not moe_curve.cumulative_flops:
        print("No checkpoints recorded. Increase flop budget or reduce eval interval.")
        return

    d_ppl = dense_curve.val_perplexity[-1]
    m_ppl = moe_curve.val_perplexity[-1]
    diff = d_ppl - m_ppl
    rel = (diff / d_ppl) * 100.0 if d_ppl > 0 else 0.0

    print("\n" + "-" * 74)
    print(f"{'Model':<12} {'Final FLOPs':>15} {'Final train loss':>18} {'Final val ppl':>16}")
    print("-" * 74)
    print(
        f"{'Dense':<12} {int(dense_curve.cumulative_flops[-1]):>15,d} "
        f"{dense_curve.train_loss[-1]:>18.4f} {dense_curve.val_perplexity[-1]:>16.2f}"
    )
    print(
        f"{'MoE':<12} {int(moe_curve.cumulative_flops[-1]):>15,d} "
        f"{moe_curve.train_loss[-1]:>18.4f} {moe_curve.val_perplexity[-1]:>16.2f}"
    )
    print("-" * 74)
    print(f"Perplexity delta (Dense - MoE): {diff:.2f} ({rel:.2f}%)")
    print("-" * 74)


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("=" * 76)
    print("Script 1: MoE vs Dense iso-compute training race")
    print("=" * 76)
    print(
        f"dataset={cfg.dataset} | seq_len={cfg.seq_len} | batch={cfg.batch_size} | "
        f"d_model={cfg.d_model} | layers={cfg.n_layers} | heads={cfg.n_heads}"
    )
    print(
        f"dense_d_ff={cfg.dense_d_ff} | moe_experts={cfg.n_experts} | k={cfg.k} "
        f"| d_ff_per_expert={cfg.d_ff_per_expert}"
    )
    print(f"total_flops={cfg.total_flops:,} | eval_interval_flops={cfg.eval_interval_flops:,}\n")

    device = detect_device()

    text = load_text_corpus(repo_root=repo_root, dataset=cfg.dataset, max_chars=cfg.max_chars)
    tokenizer, train_data, val_data = build_train_val_datasets(
        text=text,
        seq_len=cfg.seq_len,
        device=device,
    )

    base_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        seq_len=cfg.seq_len,
    )

    dense = DenseTransformerLM(base_cfg, d_ff=cfg.dense_d_ff).to(device)
    moe = MoETransformerLM(
        base_cfg,
        d_ff_per_expert=cfg.d_ff_per_expert,
        n_experts=cfg.n_experts,
        k=cfg.k,
    ).to(device)

    dense_opt = torch.optim.AdamW(dense.parameters(), lr=cfg.lr)
    moe_opt = torch.optim.AdamW(moe.parameters(), lr=cfg.lr)

    dense_step_flops = estimate_step_flops(
        batch=cfg.batch_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dense_d_ff=cfg.dense_d_ff,
    )
    moe_step_flops = estimate_step_flops(
        batch=cfg.batch_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        moe_d_ff_per_expert=cfg.d_ff_per_expert,
        moe_n_experts=cfg.n_experts,
        moe_k=cfg.k,
    )

    print(
        f"Estimated step FLOPs: dense={dense_step_flops:,} | moe={moe_step_flops:,} "
        f"| ratio={dense_step_flops / max(1, moe_step_flops):.2f}x\n"
    )

    t0 = time.perf_counter()
    dense_curve = train_to_flop_budget(
        name="Dense",
        model=dense,
        train_data=train_data,
        val_data=val_data,
        cfg=cfg,
        optimizer=dense_opt,
        step_flops=dense_step_flops,
        data_seed=cfg.seed + 777,
    )

    moe_curve = train_to_flop_budget(
        name="MoE",
        model=moe,
        train_data=train_data,
        val_data=val_data,
        cfg=cfg,
        optimizer=moe_opt,
        step_flops=moe_step_flops,
        data_seed=cfg.seed + 777,
    )

    write_curve_csv(out_dir / "isoflop_race_metrics.csv", dense_curve, moe_curve)
    plot_isoflop_curves(dense_curve, moe_curve, out_dir)
    print_final_table(dense_curve, moe_curve)

    elapsed = time.perf_counter() - t0
    print(f"\nSaved outputs to: {out_dir}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
