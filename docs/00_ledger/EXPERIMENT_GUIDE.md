# 实验数据导览（入口）

> **本文档只是指路牌。项目全貌看 `OVERVIEW.md`。**

---

## 该看哪份

| 你想知道 | 打开 |
|---|---|
| **项目做了什么、结论是什么、时间线** | **`OVERVIEW.md`** |
| 定稿的 12 格（两数据集 × 两 shot × 三结构） | `OVERVIEW.md` §三，或 `AUDIT_2026-11-27.md` §12 |
| 哪些数字作废了、为什么 | `AUDIT_2026-11-27.md` |
| 某个实验跑没跑过、多少分 | `LEDGER.md`（228 个，自动生成） |
| 目录名里每个词什么意思 | `NAMING.md` |
| 每个 run 的真实结构 | `STRUCTURE_MAP.md` |
| 服务器上什么东西能删 | `ASSETS.md` |
| 跨层方法为什么有效 | `../01_cross_layer/XL1_GRADIENT_HIGHWAY_2026-11-24.md` |
| 伪标签筛选有哪些方案 | `../03_pseudo_label/PL_FILTER_WAVE2_2026-11-26.md` |
| 哪些方向已经放弃 | `../_archive/README.md` |

## 三条铁律

1. **结构判定看权重，不看 config**——`config.py` 被覆盖过 34 次，已从 `train.log` 恢复。要绝对可靠用 `tools/gen_cells.py`。
2. **同一张表里离线只和离线比、在线只和在线比**——混比会算出 +3.6 这种假涨幅（真实约 +2.2）。
3. **只认 s42 / s7 / s13 三 seed**——随机 seed 单点方差可达 3.7 分。

## 历史版本

本文件原内容是「111 个实验的导览」（2026-11-22），已被 `OVERVIEW.md` 取代。
需要旧版可 `git log -- docs/00_ledger/EXPERIMENT_GUIDE.md` 查看。
