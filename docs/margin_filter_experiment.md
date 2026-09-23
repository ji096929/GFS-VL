# Margin Filter Experiment — GFS-VL Pseudo-label Selection

**Date**: 2026-09-22 ~ 2026-09-23  
**Machine**: NVIDIA A10 23GB (DSW instance)  
**Code**: `pointcept/engines/train.py` — `_compute_text_margin` + `pseudo_label_selection` margin branch

## Background

In GFS-VL pseudo-label selection, a **margin filter** was added: if a novel class's mean feature
is too close to base class text embeddings ("confidently wrong"), the pseudo-label is discarded.

```
margin = cos(proto, text_novel_self) - max_{b in base} cos(proto, text_b)
if margin < margin_thresh: discard
```

**Hypothesis**: Filtering "confidently wrong" pseudo-labels should improve novel mIoU.

## Setup

| Parameter | Value |
|-----------|-------|
| Config | `semseg-pt-v3m1-0-gfsregistrain_k1.py` |
| batch_size | 2 (reduced from 12 for A10 23GB VRAM) |
| SphereCrop.point_max | 65536 (reduced from 102400 for VRAM) |
| epoch | 20 |
| ps_thresh | 0.6 (unchanged) |
| save_freq | 3 (per-regis checkpointing) |
| num_worker | 6 |

> **Note**: batch_size=2 and point_max=65536 deviate from the original paper's settings
> (batch=12, point_max=102400). Absolute mIoU values are NOT directly comparable to
> the official benchmarks. Cross-comparison between our groups is valid.

## Experiment Matrix

| Exp Name | use_margin_filter | margin_thresh | regis_train_list | Status |
|----------|-------------------|---------------|------------------|--------|
| exp_no_margin_s10 | **False** | — | regis1 (seed=10) | **Done** |
| exp_m0023_s10 | True | 0.023 | regis1 (seed=10) | **Done** |
| exp_m0023_s20 | True | 0.023 | regis2 (seed=20) | **Done** |
| exp_m0122_s10 | True | 0.122 | regis1 (seed=10) | Training |
| exp_m0122_s20 | True | 0.122 | regis2 (seed=20) | Queued |

## Results

### Summary (val set)

| Exp | mIoU | BASE | NOVEL | HM |
|-----|------|------|-------|----|
| **no_margin (regis1)** | **0.3585** | 0.6556 | **0.2793** | **0.3917** |
| m0023 (regis1) | 0.3470 | 0.6880 | 0.2560 | 0.3732 |
| m0023 (regis2) | 0.3405 | 0.6641 | 0.2543 | 0.3677 |
| m0122 (regis1) | *pending* | | | |
| m0122 (regis2) | *pending* | | | |

### Per-class Analysis: m0023_s10 vs no_margin (same regis1)

**Aggregate**: Base mean Δ = **+3.2pp**, Novel mean Δ = **−2.3pp**

#### Base classes (margin helps)
| Class | ctrl | m0023 | Δ |
|-------|------|-------|---|
| chair | 0.6389 | 0.7749 | **+0.136** |
| curtain | 0.6498 | 0.7290 | +0.079 |
| window | 0.5859 | 0.6591 | +0.073 |
| refrigerator | 0.5948 | 0.6424 | +0.048 |
| bed | 0.7476 | 0.7861 | +0.039 |

#### Novel classes hurt most by margin
| Class | ctrl | m0023 | Δ | Reason |
|-------|------|-------|---|--------|
| **armchair** | **0.547** | **0.016** | **−0.531** | semantically close to base `chair` |
| nightstand | 0.472 | 0.294 | −0.178 | close to `cabinet`/`bed` |
| radiator | 0.174 | 0.001 | −0.173 | close to `wall` |
| shelf | 0.314 | 0.144 | −0.170 | close to `bookshelf` |
| book | 0.336 | 0.209 | −0.127 | close to `bookshelf` |
| office chair | 0.201 | 0.107 | −0.094 | close to `chair` |
| toilet | 0.541 | 0.449 | −0.092 | close to `bathtub` |

#### Novel classes helped by margin
| Class | ctrl | m0023 | Δ | Reason |
|-------|------|-------|---|--------|
| kitchen counter | 0.309 | 0.427 | **+0.118** | removed `cabinet` false positives |
| coffee table | 0.583 | 0.692 | +0.109 | removed `table` false positives |
| shower curtain | 0.330 | 0.434 | +0.104 | removed `curtain` false positives |
| bathroom vanity | 0.274 | 0.362 | +0.088 | removed `cabinet` false positives |
| ceiling | 0.709 | 0.796 | +0.087 | cleaner pseudo-labels |

### Pseudo-label Filter Statistics (m0023)

Training logs show `margin_rate ≈ 44.6%` — **nearly half of all novel pseudo-labels were discarded**.

```
[filter] n=199413 ps_drop=18394 margin_drop=88979 keep=92040 margin_rate=44.62%
```

For m0122 (stricter threshold): `margin_rate ≈ 84.7%`.

## Conclusions

### Per task rubric
| ΔmIoU-N (experiment − control) | Verdict |
|--------------------------------|---------|
| ≥ +1.5 | ✅ effective |
| +0.5 ~ 1.5 | ⚠️ marginal |
| **< +0.5 or negative** | **❌ reject** |

**m0023 result: ΔmIoU-N = 0.256 − 0.279 = −2.3pp → REJECT**

### Root cause
The margin filter operates at **class level** (drops ALL pseudo-labels of a class in a scene).
For novel classes that are **semantically close to base classes** (armchair→chair, shelf→bookshelf,
nightstand→cabinet), their features naturally have low margin and get **entirely wiped out**.

The filter correctly removes false positives from "base-like" predictions on distant novel classes
(kitchen counter, coffee table), but the collateral damage on proximal novel classes is too severe.

### Recommendation
**Point-level filtering** instead of class-level: drop individual low-margin points within a class
rather than discarding the entire class prototype. This would preserve hard novel examples while
still removing noisy pseudo-labels.

## Environment Notes

- CUDA 13.0 + PyTorch 2.13 + Python 3.12 (non-standard for this codebase)
- Pre-built wheels saved to `/mnt/workspace/gfs_wheels/` for fast restore after instance restart
- `restore.sh` installs from prebuilt wheels (~2 min vs ~30 min recompile)
- Torch compat patches: `weights_only=False` in `torch.load`, removed `MultiStepLR(verbose=)`,
  added `CheckpointLoader` hook for resume support
