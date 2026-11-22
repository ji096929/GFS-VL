"""OCSB 第一步的共用底座：种子、episode 构造、特征提取、原型、相关性聚合。

这里的所有函数都与 `COSeg.forward` 的后半段逐行对应：

    getFeatures -> 前景/背景原型 -> 余弦相关性 -> lin1 -> HCA 聚合
    -> class_reduce -> kpconv -> cls -> interpolation 回原始分辨率

抽出来的原因：`upper_bound.py`、`check.py` 都要用同一套前向，
各写一份必然漂移。

两个必须知道的坑：

1. `voxelize(mode=0)` 会用 `np.random.randint` 从每个体素里**随机挑一个
   代表点**（`COSeg/util/voxelize.py:91`）。不锁种子时，换一个进程点云就
   变一份，同一个 episode 的 Baseline IoU 能差到 18pp——比要测的增益大
   一个量级。所以每个 episode 开始前必须 `seed_everything(10000 + g_ep)`，
   而且 g_ep 要用**全局**编号，不能是分片内的局部编号。

2. 送进 `lin1` 的原型数必须恒为 K（每个类别 K 个）。多个 U 块聚合时靠
   "每块各采 K 个再按块平均"实现，绝不能用 `K // n_blocks` 去砍份额，
   否则 `lin1` 的输入通道数随块数变化，语义直接坏掉。
"""
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as Fs
from scipy.spatial import cKDTree

sys.path.insert(0, '/data01/jh/code/COSeg')
sys.path.insert(0, '/data01/jh/code')
from ring4_isolated import prepare_block  # noqa: E402,F401
from lib.pointops2.functions import pointops  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
NAMES_PATH = '/data01/jh/code/COSeg/datasets/ScanNet/meta/scannet_classnames.txt'
META_PATH = os.path.join(CACHE, 'val_meta_all.json')
# 10 个 novel 类在 scannet_classnames 里的下标，顺序决定逐类表的行序
NEW_IDS = [10, 6, 7, 19, 3, 13, 12, 5, 14, 2]
K = 100          # 每个类别采多少个原型（COSeg 的 n_subprototypes）
SEED_BASE = 10000  # episode 种子基数，与模型构造种子 0 分开


def seed_everything(seed):
    """锁死所有随机源。

    重点是 `np.random.seed`：体素化选点走的是 numpy 的全局 RNG，
    只设 torch 的种子完全没用。cudnn 的 deterministic 是第二道保险，
    防止换卡换算法带来 1e-6 级抖动。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model():
    """构造模型：先锁 0 号种子再 build，保证构造期的随机初始化也确定。

    `diag_bc.build_model()` 内部是 COSeg(cfg) -> 加载 backbone -> 加载
    sc0_1w1s.pth -> eval()。三者都是 strict 的，338 个 key 全部覆盖，
    所以这里锁种子只影响 forward 的可复现性，不影响权重本身。
    """
    import diag_bc as D
    seed_everything(0)
    model, cfg = D.build_model()
    return model, cfg


def build_episodes(per_class, shard=0, shards=1):
    """按「类内序号优先、类别循环在内」的顺序构造 episode 列表。

    顺序是 post 脚本推算全局编号的依据：
        g_ep = shard + shards * ei
    某类样本不足 per_class 时会跳过，所以总数不一定是 10 * per_class
    （val_560 实际是 539 个）。
    """
    metas = json.load(open(META_PATH))
    by_class = defaultdict(list)
    for m in metas:
        by_class[m['class']].append(m)
    eps = []
    for i in range(per_class):
        for c in NEW_IDS:
            if i < len(by_class[c]):
                eps.append(by_class[c][i])
    if shards > 1:
        eps = eps[shard::shards]
    return eps


def load_class_names():
    return [l.strip() for l in open(NAMES_PATH)]


@torch.no_grad()
def feats_one(model, x, y, dev):
    """单块点云过 backbone，拿降采样特征、坐标、下采样标签。

    返回的 `yl` 是**降采样后**的真值标签（N_down），后面所有按点取原型的
    掩码都必须和它同尺寸；`xyz`/`off` 保留原始分辨率，供最后的插值用。
    """
    off = torch.tensor([len(x)], dtype=torch.int32)
    fd, cd, ol, yl, _, _ = model.getFeatures(
        x.contiguous().to(dev), off.to(dev), y.to(dev), None)
    return dict(fd=fd, cd=cd, ol=ol, yl=yl,
                xyz=x[:, :3].contiguous().to(dev), off=off.to(dev))


@torch.no_grad()
def build_proto(model, f, mask, k=K):
    """从一块点云里按掩码采 k 个原型。

    点数不足或掩码为空时返回全零块，保持返回形状恒为 (k, feat_dim)，
    这样 `lin1` 的输入通道数在任何情况下都不变。
    """
    mask = mask.bool()
    if int(mask.sum()) == 0:
        return torch.zeros(k, model.feat_dim, device=f['fd'].device)
    return model.getMutiplePrototypes(f['cd'][mask], f['fd'][mask], k)


@torch.no_grad()
def predict_agg(model, qf, sparse_list, lam, dev, mode='mix', margin=0.0,
                proto_weights=None):
    """复刻 COSeg.forward 的后半段，支持把多个原型集的**相关性**加权融合。

    参数
    ----
    sparse_list : [S 原型, U 原型_1, ..., U 原型_m]，每个都是 (2K, C)
    lam         : U 原型的权重。0 或只有一项时完全退化成 Baseline。
    mode        : 'mix'       双边线性混合 (1-λ)·S + λ·U
                  'one_sided' 单向遗漏修补 S + λ·relu(U − S − margin)
                              U 只能把相关性往上抬，不能否决 S 已认出的点
    margin      : one_sided 模式下 U 要超过 S 多少个单位才生效
    proto_weights : (2K,) 逐原型权重，直接乘在 U 侧的相关性上。用来保留
                  BridgeScorer 的连续可信度，而不是把选中的点一律当 1.0。

    融合发生在余弦相关性上而不是原型上：
        corr = (1-λ) * corr_S + λ * mean_m corr_U_m
    因为 `lin1` 是线性的且 (1-λ)+λ=1，所以它与"先各自过 lin1 再平均"
    严格等价，偏置项正好抵消。

    返回
    ----
    pred_raw : (2, N_raw) 插值回原始分辨率的 logits
    out      : (N_down, 2) 降采样分辨率上的 logits，用来算 U 的初始前景概率
    """
    q_fd, q_cd = qf['fd'], qf['cd']

    # base 引导：查询点对 10 个 base 类原型的最大余弦相似度，喂给 HCA 层
    base_sim = Fs.cosine_similarity(
        q_fd[:, None, :], model.base_prototypes[None, :, :], dim=2)
    base_g = base_sim.max(dim=1, keepdim=True)[0]

    corrs = [Fs.cosine_similarity(q_fd[:, None, :], sp[None, :, :], dim=2)
             for sp in sparse_list]
    if len(corrs) == 1 or lam <= 0:
        corr = corrs[0]
    else:
        cu = torch.stack(corrs[1:]).mean(0)
        if proto_weights is not None:
            cu = cu * proto_weights[None, :].to(cu.device).to(cu.dtype)
        if mode == 'one_sided':
            corr = corrs[0] + lam * Fs.relu(cu - corrs[0] - margin)
        else:
            corr = (1.0 - lam) * corrs[0] + lam * cu

    # (N_down, 2K) -> (N_down, 2, K)；2 == n_way+1（背景 + 前景）
    corr = model.lin1(corr.view(corr.shape[0], 2, -1)).permute(2, 1, 0).unsqueeze(0)
    for layer in model.agglayers:
        corr = layer(corr, base_g)
    corr = corr.squeeze(0).permute(2, 1, 0).contiguous()
    corr = model.class_reduce(corr).squeeze(1)

    # kpconv：在降采样点云上做局部聚合，sigma=2.0 与原生 forward 一致
    coord = q_cd.contiguous()
    batch = torch.zeros(corr.shape[0], dtype=torch.int64, device=coord.device)
    radius = 2.5 * model.args.grid_size * 2.0
    off = torch.tensor([coord.shape[0]], dtype=torch.int32, device=coord.device)
    nbr = pointops.ball_query(radius, model.args.max_num_neighbors,
                              coord, coord, off, off)[0].long()
    corr = model.kpconv(corr, coord, batch, nbr.clone())
    out = model.cls(corr)

    # 插值回原始分辨率：源点用降采样坐标，目标点用原始 xyz
    pred_raw = pointops.interpolation(
        q_cd, qf['xyz'], out.contiguous(), qf['ol'], qf['off']
    ).transpose(0, 1).unsqueeze(0)[0]
    return pred_raw, out


@torch.no_grad()
def build_u_sparse(model, u_feats, mode, fg_hi=0.7, bg_lo=0.2, k=K):
    """按三种口径从 U 块构建 (背景 K 个 + 前景 K 个) 原型。

    mode
    ----
    'naive'      前景 = prob > fg_hi，背景 = prob < bg_lo（含假阳性，可用方法）
    'oracle_b'   前景 = prob > fg_hi 且 真值前景（纯桥梁 B_pos）
    'oracle_bc'  前景 = 全部真值前景（桥梁 + 遗漏区 C_pos，理论上界）

    背景一律只取真值背景，模糊区（prob 介于两者之间）不参与任何原型——
    这条是修 Bug3 时定下的：把漏检的真前景塞进背景库会直接破坏背景原型。

    多个 U 块时，每块各采 K 个再按块平均，保证输出恒为 (2K, C)。
    返回 (sparse, 前景点数总和)。
    """
    fgs, bgs = [], []
    n_fg = 0
    for uf in u_feats:
        prob = uf['prob'].to(uf['fd'].device)
        lab = uf['yl'].to(uf['fd'].device)      # 降采样后的真值前景(0/1)
        if mode == 'naive':
            m_fg = prob > fg_hi
            m_bg = prob < bg_lo
        elif mode == 'oracle_b':
            m_fg = (prob > fg_hi) & (lab > 0.5)
            m_bg = lab <= 0.5
        else:                                    # oracle_bc
            m_fg = lab > 0.5
            m_bg = lab <= 0.5
        n_fg += int(m_fg.sum())
        fgs.append(build_proto(model, uf, m_fg, k))
        bgs.append(build_proto(model, uf, m_bg, k))
    return torch.cat([torch.stack(bgs).mean(0), torch.stack(fgs).mean(0)], 0), n_fg


@torch.no_grad()
def iou_counts(pred, gt):
    """返回 (tp, fp, fn, iou)。pred 是 (2, N) 的 logits 或 (N,) 的 0/1。

    统一在这里算，避免各脚本对 argmax 维度/阈值的理解不一致。
    """
    p = pred.argmax(0) if pred.dim() == 2 else pred
    p = p.cpu()
    gt = gt.cpu()
    tp = int(((p == 1) & (gt == 1)).sum())
    fp = int(((p == 1) & (gt == 0)).sum())
    fn = int(((p == 0) & (gt == 1)).sum())
    return tp, fp, fn, tp / max(tp + fp + fn, 1)


def episode_tensors(block, cls):
    """把一个 block 拆成 (点云 6 通道输入, 二值前景标签)。"""
    x = torch.cat([block['coord'], block['feat']], 1)
    y = (block['label'] == cls).long()
    return x, y


def seg_downsample(xyz, seg, cd):
    """把原始分辨率的超点 id 映射到降采样点：取最近原始点的 seg。

    `getFeatures` 只返回降采样点，而超点 id 只在原始分辨率上有，所以要用
    KDTree 做一次最近邻映射。同一个超点内的降采样点会共享同一个 id。
    """
    if xyz.shape[0] == cd.shape[0]:
        return seg
    tree = cKDTree(xyz.detach().cpu().numpy())
    _, idx = tree.query(cd.detach().cpu().numpy())
    return torch.from_numpy(np.asarray(seg.cpu().numpy())[idx]).to(cd.device)
