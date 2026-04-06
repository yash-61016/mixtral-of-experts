"""
Mixtral of Experts - Router Load-Balance Visualizer
===================================================
Paper: Mixtral of Experts (arXiv: 2401.04088)
What this implements: A paired MoE training experiment that compares router
                      behavior with and without auxiliary load-balancing loss,
                      then visualizes expert entropy and usage heatmaps.
Hardware: RTX 3090, 24GB VRAM (also runs on CPU with longer runtime)
Time to run: ~15 minutes for smoke config, ~1-2 hours for full config

Dependencies (pinned):
    torch==2.1.0          # pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    matplotlib==3.7.1     # pip install matplotlib==3.7.1
    numpy==1.24.3         # pip install numpy==1.24.3
    datasets==2.19.0      # optional, for TinyStories/WikiText-2 loading

Run:
    python mixtral-of-experts/mixtral_router_balance.py
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
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "theme"))
from deep_sea_theme import PALETTE, EXTENDED, add_source_label, annotate_points, apply_theme

from moe_core import (
    ExpertUsageTracker,
    ModelConfig,
    MoETransformerLM,
    auxiliary_load_balancing_loss,
    build_train_val_datasets,
    causal_lm_loss,
    load_text_corpus,
    seed_everything,
)

apply_theme()


@dataclass
class RouterRunSummary:
    """Recorded metrics for one routing run."""

    name: str
    steps: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_perplexity: List[float] = field(default_factory=list)
    aux_loss: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    top1_share: List[float] = field(default_factory=list)
    heatmap_rows: List[np.ndarray] = field(default_factory=list)


@dataclass
class BalanceConfig:
    """Configuration for router balance experiment."""

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
    steps: int = 2_000
    checkpoint_every: int = 40
    val_batches: int = 20
    lr: float = 3e-4
    seed: int = 11
    aux_alpha: float = 1e-2

    @property
    def d_ff_per_expert(self) -> int:
        return self.dense_d_ff // self.n_experts


def parse_args() -> BalanceConfig:
    """Parse CLI args for router balance script."""
    parser = argparse.ArgumentParser(description="Router collapse vs aux-loss balancing experiment")
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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--aux-alpha", type=float, default=1e-2)
    args = parser.parse_args()
    return BalanceConfig(
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
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        val_batches=args.val_batches,
        lr=args.lr,
        seed=args.seed,
        aux_alpha=args.aux_alpha,
    )


def detect_device() -> torch.device:
    """Print hardware details and return selected torch device."""
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


def evaluate_ppl(
    model: torch.nn.Module,
    val_data,
    batch_size: int,
    val_batches: int,
    generator: torch.Generator,
) -> float:
    """Compute validation perplexity at current checkpoint."""
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for _ in range(val_batches):
            x, y = val_data.sample_batch(batch_size, generator=generator)
            out = model(x)
            loss = causal_lm_loss(out["logits"], y)
            losses.append(loss.item())
    val_loss = float(sum(losses) / len(losses))
    return float(math.exp(val_loss))


def visualise_expert_distribution_heatmap(step_checkpoints: List[np.ndarray], out_path: Path, title: str) -> None:
    """Render expert usage heatmap with x=expert and y=checkpoint."""
    matrix = np.stack(step_checkpoints, axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(matrix, cmap="deep_sea_light", aspect="auto", vmin=0.0, vmax=max(1e-6, matrix.max()))
    ax.set_title(title)
    ax.set_xlabel("Expert id")
    ax.set_ylabel("Checkpoint index")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Selection frequency")
    add_source_label(ax)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def train_router_run(
    run_name: str,
    cfg: BalanceConfig,
    device: torch.device,
    train_data,
    val_data,
    vocab_size: int,
    alpha: float,
    data_seed: int,
) -> RouterRunSummary:
    """Train one MoE run and record routing diagnostics."""
    base_cfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        seq_len=cfg.seq_len,
    )
    model = MoETransformerLM(
        base_cfg,
        d_ff_per_expert=cfg.d_ff_per_expert,
        n_experts=cfg.n_experts,
        k=cfg.k,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    tracker = ExpertUsageTracker(cfg.n_experts)
    summary = RouterRunSummary(name=run_name)

    for step in range(1, cfg.steps + 1):
        model.train()
        train_gen = torch.Generator().manual_seed(data_seed + step)
        x, y = train_data.sample_batch(cfg.batch_size, generator=train_gen)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)

        lm_loss = causal_lm_loss(out["logits"], y)
        aux = auxiliary_load_balancing_loss(
            router_probs=out["router_probs"],
            expert_indices=out["expert_indices"],
            n_experts=cfg.n_experts,
        )
        loss = lm_loss + alpha * aux
        loss.backward()
        optimizer.step()

        tracker.update(out["expert_indices"])

        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            freq_t = tracker.window_frequencies(reset=True)
            freq = freq_t.cpu().numpy()
            if freq_t.sum().item() > 0:
                p = freq_t.clamp(min=1e-9)
                entropy = float((-(p * torch.log(p))).sum().item())
            else:
                entropy = 0.0

            val_gen = torch.Generator().manual_seed(data_seed + 100_000 + step)
            ppl = evaluate_ppl(model, val_data, cfg.batch_size, cfg.val_batches, generator=val_gen)
            top1 = float(freq_t.max().item()) if freq_t.numel() else 0.0

            summary.steps.append(step)
            summary.train_loss.append(float(lm_loss.item()))
            summary.val_perplexity.append(ppl)
            summary.aux_loss.append(float(aux.item()))
            summary.entropy.append(entropy)
            summary.top1_share.append(top1)
            summary.heatmap_rows.append(freq)

            print(
                f"[{run_name:8}] step={step:4d} | train_loss={lm_loss.item():.4f} "
                f"| val_ppl={ppl:.2f} | aux={aux.item():.4f} | entropy={entropy:.3f} | top1={top1:.3f}"
            )

    return summary


def plot_entropy_over_training(runs: Dict[str, RouterRunSummary], out_dir: Path) -> None:
    """Plot entropy trajectories for no-aux and with-aux runs."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = ["no_aux", "with_aux"]
    colors = {
        "no_aux": PALETTE["steel"],
        "with_aux": EXTENDED["success"],
    }
    styles = {
        "no_aux": "o-",
        "with_aux": "s--",
    }
    for name in order:
        run = runs[name]
        ax.plot(
            run.steps,
            run.entropy,
            styles[name],
            color=colors[name],
            label=name,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        annotate_points(ax, run.steps, run.entropy, fmt="{:.2f}", color=colors[name])

    ax.set_title("Router entropy over training: no aux vs with aux")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Expert selection entropy")
    ax.legend(loc="best")
    add_source_label(ax)
    fig.savefig(out_dir / "router_entropy_over_training.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(runs: Dict[str, RouterRunSummary], out_path: Path) -> None:
    """Write checkpoint diagnostics to CSV."""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "step", "train_loss", "val_perplexity", "aux_loss", "entropy", "top1_share"])
        for name, summary in runs.items():
            for i in range(len(summary.steps)):
                writer.writerow(
                    [
                        name,
                        summary.steps[i],
                        f"{summary.train_loss[i]:.6f}",
                        f"{summary.val_perplexity[i]:.6f}",
                        f"{summary.aux_loss[i]:.6f}",
                        f"{summary.entropy[i]:.6f}",
                        f"{summary.top1_share[i]:.6f}",
                    ]
                )


def print_summary_table(runs: Dict[str, RouterRunSummary]) -> None:
    """Print final run comparison for entropy and collapse metrics."""
    no_aux = runs["no_aux"]
    with_aux = runs["with_aux"]

    print("\n" + "-" * 106)
    print(
        f"{'Run':<12} {'Final entropy':>16} {'Final top1 share':>20} "
        f"{'Final train loss':>20} {'Final val ppl':>16}"
    )
    print("-" * 106)
    print(
        f"{'no_aux':<12} {no_aux.entropy[-1]:>16.4f} {no_aux.top1_share[-1]:>20.4f} "
        f"{no_aux.train_loss[-1]:>20.4f} {no_aux.val_perplexity[-1]:>16.2f}"
    )
    print(
        f"{'with_aux':<12} {with_aux.entropy[-1]:>16.4f} {with_aux.top1_share[-1]:>20.4f} "
        f"{with_aux.train_loss[-1]:>20.4f} {with_aux.val_perplexity[-1]:>16.2f}"
    )
    print("-" * 106)
    delta = with_aux.entropy[-1] - no_aux.entropy[-1]
    print(f"Entropy delta (with_aux - no_aux): {delta:.4f}")
    print("-" * 106)


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    repo_root = Path(__file__).resolve().parent.parent
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print("Script 2: Router load-balance visualizer with auxiliary loss")
    print("=" * 78)
    print(
        f"dataset={cfg.dataset} | seq_len={cfg.seq_len} | batch={cfg.batch_size} | "
        f"d_model={cfg.d_model} | layers={cfg.n_layers} | experts={cfg.n_experts} | k={cfg.k}"
    )
    print(
        f"steps={cfg.steps} | checkpoint_every={cfg.checkpoint_every} "
        f"| aux_alpha={cfg.aux_alpha:.4f}\n"
    )

    device = detect_device()
    text = load_text_corpus(repo_root=repo_root, dataset=cfg.dataset, max_chars=cfg.max_chars)
    tokenizer, train_data, val_data = build_train_val_datasets(
        text=text,
        seq_len=cfg.seq_len,
        device=device,
    )

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}\n")

    t0 = time.perf_counter()
    runs: Dict[str, RouterRunSummary] = {}
    runs["no_aux"] = train_router_run(
        run_name="no_aux",
        cfg=cfg,
        device=device,
        train_data=train_data,
        val_data=val_data,
        vocab_size=tokenizer.vocab_size,
        alpha=0.0,
        data_seed=cfg.seed + 1234,
    )
    runs["with_aux"] = train_router_run(
        run_name="with_aux",
        cfg=cfg,
        device=device,
        train_data=train_data,
        val_data=val_data,
        vocab_size=tokenizer.vocab_size,
        alpha=cfg.aux_alpha,
        data_seed=cfg.seed + 1234,
    )

    plot_entropy_over_training(runs, out_dir)
    visualise_expert_distribution_heatmap(
        runs["no_aux"].heatmap_rows,
        out_dir / "expert_usage_heatmap_no_aux.png",
        "Expert selection heatmap: no auxiliary loss",
    )
    visualise_expert_distribution_heatmap(
        runs["with_aux"].heatmap_rows,
        out_dir / "expert_usage_heatmap_with_aux.png",
        "Expert selection heatmap: with auxiliary loss",
    )
    write_csv(runs, out_dir / "router_balance_metrics.csv")
    print_summary_table(runs)

    elapsed = time.perf_counter() - t0
    print(f"\nSaved outputs to: {out_dir}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
