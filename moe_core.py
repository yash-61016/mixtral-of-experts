"""Core utilities for Mixtral-style sparse MoE training artifacts.

This module holds reusable components shared by both artifact scripts:
- FLOP accounting helpers
- Router auxiliary load-balancing loss
- Expert usage tracking and entropy diagnostics
- Minimal causal LM blocks for dense and MoE transformers
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for reproducible experiment traces."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CharTokenizer:
    """Character-level tokenizer for small LM experiments.

    Character tokenization keeps dependencies light while still enabling
    meaningful train/validation perplexity comparisons.
    """

    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class SequenceDataset:
    """Streaming contiguous token batches for causal LM training."""

    def __init__(self, token_ids: torch.Tensor, seq_len: int, device: torch.device) -> None:
        if token_ids.numel() <= seq_len + 1:
            raise ValueError("Need more tokens than seq_len + 1")
        self.tokens = token_ids
        self.seq_len = seq_len
        self.device = device

    def sample_batch(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_start = self.tokens.numel() - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,), generator=generator)
        x = torch.stack([self.tokens[s : s + self.seq_len] for s in starts])
        y = torch.stack([self.tokens[s + 1 : s + 1 + self.seq_len] for s in starts])
        return x.to(self.device), y.to(self.device)


def load_text_corpus(repo_root: Path, dataset: str, max_chars: int) -> str:
    """Load corpus text.

    First tries Hugging Face datasets when available, then falls back to local
    repo markdown/text files so scripts remain runnable offline.
    """

    dataset = dataset.lower().strip()
    text = ""
    try:
        from datasets import load_dataset  # type: ignore

        if dataset == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split="train[:1%]")
            text = "\n".join(item["text"] for item in ds if item.get("text"))
        elif dataset == "wikitext2":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n".join(item["text"] for item in ds if item.get("text"))
    except Exception:
        text = ""

    if not text:
        fallback_files = [
            repo_root / "attention-is-all-you-need" / "attention-blog.md",
            repo_root / "mixtral-of-experts" / "notes" / "Mixtral of Experts.md",
            repo_root / "llama-2" / "llama-blog.md",
            repo_root / "Applied LLM Mechanics.md",
        ]
        chunks: List[str] = []
        for path in fallback_files:
            if path.exists():
                chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        text = "\n\n".join(chunks)

    if len(text) < 1000:
        raise RuntimeError("Corpus too small. Add more local text or install datasets package.")

    return text[:max_chars]


def build_train_val_datasets(
    text: str,
    seq_len: int,
    device: torch.device,
    train_fraction: float = 0.9,
) -> Tuple[CharTokenizer, SequenceDataset, SequenceDataset]:
    """Tokenize corpus and split into train/val contiguous streams."""
    tokenizer = CharTokenizer(text)
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(ids.numel() * train_fraction)
    train_ids = ids[:split]
    val_ids = ids[split:]
    return (
        tokenizer,
        SequenceDataset(train_ids, seq_len=seq_len, device=device),
        SequenceDataset(val_ids, seq_len=seq_len, device=device),
    )


@dataclass
class ModelConfig:
    """Configuration shared by dense and MoE language models."""

    vocab_size: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    seq_len: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    """Minimal masked self-attention block for autoregressive language modeling."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.seq_len, cfg.seq_len)).view(1, 1, cfg.seq_len, cfg.seq_len)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.out_proj(out)


class DenseFFN(nn.Module):
    """Dense SwiGLU feed-forward baseline for apples-to-apples MoE comparison."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(gated)


class SwiGLUExpert(nn.Module):
    """SwiGLU expert used inside MoE FFN blocks."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(gated)


class MoEFeedForward(nn.Module):
    """Top-k sparse MoE feed-forward block with explicit token dispatch."""

    def __init__(self, d_model: int, d_ff_per_expert: int, n_experts: int = 8, k: int = 2) -> None:
        super().__init__()
        if k > n_experts:
            raise ValueError("k must be <= n_experts")
        self.n_experts = n_experts
        self.k = k
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff_per_expert) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, d_model = x.shape
        flat_x = x.reshape(-1, d_model)
        logits = self.router(flat_x)
        router_probs = F.softmax(logits, dim=-1)

        topk_vals, topk_idx = torch.topk(logits, k=self.k, dim=-1)
        topk_probs = F.softmax(topk_vals, dim=-1)

        out = torch.zeros_like(flat_x)
        for expert_id, expert in enumerate(self.experts):
            for slot in range(self.k):
                token_mask = topk_idx[:, slot] == expert_id
                if token_mask.any():
                    selected = flat_x[token_mask]
                    expert_out = expert(selected)
                    weights = topk_probs[token_mask, slot].unsqueeze(-1)
                    out[token_mask] += expert_out * weights

        return out.view(bsz, seq_len, d_model), router_probs.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, self.k)


class DenseTransformerBlock(nn.Module):
    """Transformer block with dense FFN."""

    def __init__(self, cfg: ModelConfig, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ffn = DenseFFN(cfg.d_model, d_ff)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x, None, None


class MoETransformerBlock(nn.Module):
    """Transformer block with sparse MoE FFN."""

    def __init__(self, cfg: ModelConfig, d_ff_per_expert: int, n_experts: int = 8, k: int = 2) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.moe = MoEFeedForward(cfg.d_model, d_ff_per_expert=d_ff_per_expert, n_experts=n_experts, k=k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.ln1(x))
        moe_out, probs, indices = self.moe(self.ln2(x))
        x = x + moe_out
        return x, probs, indices


class DenseTransformerLM(nn.Module):
    """Dense autoregressive language model for iso-compute comparisons."""

    def __init__(self, cfg: ModelConfig, d_ff: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([DenseTransformerBlock(cfg, d_ff=d_ff) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        h = self.tok(x) + self.pos(pos).unsqueeze(0)

        for block in self.blocks:
            h, _, _ = block(h)

        logits = self.head(self.ln_f(h))
        return {"logits": logits}


class MoETransformerLM(nn.Module):
    """Sparse MoE autoregressive language model for iso-compute comparisons."""

    def __init__(self, cfg: ModelConfig, d_ff_per_expert: int, n_experts: int = 8, k: int = 2) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_experts = n_experts
        self.k = k
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                MoETransformerBlock(
                    cfg,
                    d_ff_per_expert=d_ff_per_expert,
                    n_experts=n_experts,
                    k=k,
                )
                for _ in range(cfg.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        h = self.tok(x) + self.pos(pos).unsqueeze(0)

        router_probs: List[torch.Tensor] = []
        topk_idx: List[torch.Tensor] = []
        for block in self.blocks:
            h, probs, indices = block(h)
            router_probs.append(probs)
            topk_idx.append(indices)

        logits = self.head(self.ln_f(h))
        return {
            "logits": logits,
            "router_probs": torch.stack(router_probs, dim=1),
            "expert_indices": torch.stack(topk_idx, dim=1),
        }


def causal_lm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over flattened token positions."""
    vocab = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))


def estimate_attention_flops(batch: int, seq_len: int, d_model: int, n_layers: int) -> int:
    """Approximate attention FLOPs for one forward pass.

    Formula uses two terms per layer:
    - qkv and output projections: 8 * B * T * d_model^2
    - attention score and weighted value: 4 * B * T^2 * d_model
    """
    proj = 8 * batch * seq_len * d_model * d_model
    att = 4 * batch * seq_len * seq_len * d_model
    return n_layers * (proj + att)


def estimate_dense_ffn_flops(batch: int, seq_len: int, d_model: int, d_ff: int) -> int:
    """Approximate dense FFN FLOPs for one forward pass.

    Dense SwiGLU uses three projections (gate, up, down).
    Multiply+add accounting yields factor 6.
    """
    return 6 * batch * seq_len * d_model * d_ff


def estimate_moe_ffn_active_flops(
    batch: int,
    seq_len: int,
    d_model: int,
    d_ff_per_expert: int,
    k: int,
) -> int:
    """Approximate active MoE FFN FLOPs for one forward pass.

    Each token executes k experts. Each SwiGLU expert executes three projection
    matmuls (gate, up, down). Each matmul uses a factor 2 for multiply+add,
    yielding a factor 6.
    """
    return 6 * batch * seq_len * d_model * d_ff_per_expert * k


def estimate_router_flops(batch: int, seq_len: int, d_model: int, n_experts: int) -> int:
    """Approximate router FLOPs for one MoE layer forward pass.

    Includes linear projection to router logits and a small softmax/top-k term.
    """
    projection = 2 * batch * seq_len * d_model * n_experts
    softmax_topk = 5 * batch * seq_len * n_experts
    return projection + softmax_topk


def estimate_step_flops(
    batch: int,
    seq_len: int,
    d_model: int,
    n_layers: int,
    dense_d_ff: Optional[int] = None,
    moe_d_ff_per_expert: Optional[int] = None,
    moe_n_experts: Optional[int] = None,
    moe_k: int = 2,
    backward_multiplier: float = 3.0,
) -> int:
    """Estimate total training-step FLOPs including backward pass.

    backward_multiplier approximates forward + backward + optimizer costs.
    """
    att = estimate_attention_flops(batch=batch, seq_len=seq_len, d_model=d_model, n_layers=n_layers)

    if dense_d_ff is not None:
        per_layer_ffn = estimate_dense_ffn_flops(batch, seq_len, d_model, dense_d_ff)
        per_layer_router = 0
    elif moe_d_ff_per_expert is not None:
        if moe_n_experts is None:
            raise ValueError("Provide moe_n_experts when using moe_d_ff_per_expert")
        per_layer_ffn = estimate_moe_ffn_active_flops(
            batch=batch,
            seq_len=seq_len,
            d_model=d_model,
            d_ff_per_expert=moe_d_ff_per_expert,
            k=moe_k,
        )
        per_layer_router = estimate_router_flops(
            batch=batch,
            seq_len=seq_len,
            d_model=d_model,
            n_experts=moe_n_experts,
        )
    else:
        raise ValueError("Provide dense_d_ff or moe_d_ff_per_expert")

    forward = att + n_layers * (per_layer_ffn + per_layer_router)
    return int(forward * backward_multiplier)


class FLOPCounter:
    """Monotonic cumulative FLOP counter for FLOP-budgeted training loops."""

    def __init__(self) -> None:
        self.total = 0

    def add(self, flops: int) -> None:
        if flops < 0:
            raise ValueError("FLOPs to add must be non-negative")
        self.total += int(flops)


class ExpertUsageTracker:
    """Tracks per-expert routing frequency and entropy over time."""

    def __init__(self, n_experts: int) -> None:
        self.n_experts = n_experts
        self.counts = torch.zeros(n_experts, dtype=torch.float64)
        self.window_counts = torch.zeros(n_experts, dtype=torch.float64)
        self.history: List[torch.Tensor] = []

    def update(self, expert_indices: torch.Tensor) -> None:
        flat = expert_indices.reshape(-1).to(device="cpu", dtype=torch.long)
        step_counts = torch.bincount(flat, minlength=self.n_experts).to(torch.float64)
        self.counts += step_counts
        self.window_counts += step_counts

    def _normalized(self, counts: torch.Tensor) -> torch.Tensor:
        total = counts.sum()
        if total <= 0:
            return torch.zeros(self.n_experts, dtype=torch.float32)
        return (counts / total).to(torch.float32)

    def frequencies(self) -> torch.Tensor:
        return self._normalized(self.counts)

    def window_frequencies(self, reset: bool = False) -> torch.Tensor:
        freqs = self._normalized(self.window_counts)
        if reset:
            self.window_counts.zero_()
        return freqs

    def entropy(self) -> torch.Tensor:
        p = self.frequencies().clamp(min=1e-9)
        if p.sum() <= 0:
            return torch.tensor(0.0)
        return -(p * torch.log(p)).sum()

    def window_entropy(self, reset: bool = False) -> torch.Tensor:
        p = self.window_frequencies(reset=reset)
        if p.sum() <= 0:
            return torch.tensor(0.0)
        p = p.clamp(min=1e-9)
        return -(p * torch.log(p)).sum()

    def reset_window(self) -> None:
        self.window_counts.zero_()

    def snapshot(self) -> torch.Tensor:
        freqs = self.frequencies().clone()
        self.history.append(freqs)
        return freqs


def auxiliary_load_balancing_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Switch-style top-1 auxiliary loss encouraging balanced expert usage.

    router_probs: [B, T, E] or [B, L, T, E]
    expert_indices: [B, T, K] or [B, L, T, K]
    """
    probs = router_probs.reshape(-1, n_experts)
    if expert_indices.shape[-1] > 1:
        top1_idx = expert_indices[..., 0].reshape(-1)
    else:
        top1_idx = expert_indices.reshape(-1)

    dispatch = F.one_hot(top1_idx.to(torch.long), num_classes=n_experts).to(probs.dtype)

    f = dispatch.mean(dim=0)
    p = probs.mean(dim=0)
    return n_experts * torch.sum(f * p)
