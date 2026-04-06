# Mixtral of Experts - Artifacts

From-scratch implementation and experiment artifact for [Mixtral of Experts](https://arxiv.org/abs/2401.04088).

Accompanies the write-up: [Mixtral of Experts: Top-2 Routing Gives 47B Capacity at 13B Active Compute](mixtral-blog.md)

---

## Files

| File | Description |
|------|-------------|
| `mixtral_isoflop_race.py` | Trains dense and sparse MoE Transformers to the same cumulative FLOP budget, then compares train loss and validation perplexity checkpoint-by-checkpoint. |
| `mixtral_router_balance.py` | Runs paired MoE training (`no_aux` vs `with_aux`) and tracks router entropy, top-1 concentration, and per-expert usage heatmaps over training. |
| `moe_core.py` | Shared components: minimal dense/MoE LM blocks, SwiGLU FFNs, top-k routing, FLOP estimators, auxiliary load-balancing loss, and usage diagnostics. |
| `test_moe_core.py` | Unit tests for FLOP formulas, auxiliary loss behavior, usage tracking, and deterministic dataset sampling. |

## Hardware

Target hardware is RTX 3090 (24GB VRAM). Both scripts also run on CPU (slower).

- `mixtral_isoflop_race.py` smoke config runs in ~20 minutes on a 3090
- `mixtral_router_balance.py` smoke config runs in ~15 minutes on a 3090

## Setup

```bash
# CUDA 12.1 build of PyTorch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# plotting + numerics
pip install matplotlib==3.7.1 numpy==1.24.3

# optional: only needed for direct TinyStories/WikiText-2 loading
pip install datasets==2.19.0
```

## Run

```bash
# Script 1: FLOP-matched dense vs MoE race
python mixtral_isoflop_race.py \
  --dataset tinystories \
  --total-flops 2e11 \
  --eval-interval-flops 2e10 \
  --max-steps 300

# Script 2: router balance no-aux vs with-aux
python mixtral_router_balance.py \
  --dataset tinystories \
  --steps 300 \
  --checkpoint-every 25 \
  --aux-alpha 1e-2
```

## Outputs

Both scripts write outputs to `outputs/`.

**`mixtral_isoflop_race.py`**
- `isoflop_train_loss.png` - train loss vs cumulative FLOPs
- `isoflop_val_perplexity.png` - validation perplexity vs cumulative FLOPs
- `isoflop_race_metrics.csv` - checkpoint metrics (flops, step, tokens, losses, wall time)

**`mixtral_router_balance.py`**
- `router_entropy_over_training.png` - expert selection entropy across checkpoints
- `expert_usage_heatmap_no_aux.png` - per-expert routing distribution without auxiliary loss
- `expert_usage_heatmap_with_aux.png` - per-expert routing distribution with auxiliary loss
- `router_balance_metrics.csv` - checkpoint metrics (train loss, val perplexity, aux loss, entropy, top1 share)

---

## Key Finding

At nearly equal cumulative training FLOPs, sparse MoE achieves lower validation perplexity than dense in this artifact setup.

Dense endpoint: 25.31 val ppl at 2.079e11 FLOPs  
MoE endpoint: 20.98 val ppl at 2.049e11 FLOPs  
Delta: -4.324 perplexity (17.09% lower) for MoE

```text
KEY FINDING: compute-matched dense vs sparse MoE
  Dense val perplexity : 25.308
  MoE   val perplexity : 20.984
  Delta (dense-moe)    : 4.324 (17.09%)
```

Additional router-balance observation from the paired run:
- with auxiliary loss, entropy rises (2.0596 vs 2.0276)
- top-1 concentration drops (0.1759 vs 0.1822)
- validation perplexity is similar in this smoke-scale setting
