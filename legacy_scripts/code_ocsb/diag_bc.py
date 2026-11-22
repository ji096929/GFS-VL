"""Step-1 diagnostic for 3D-OCSB premise.

Frozen COSeg, 1-way 1-shot ScanNet val episodes. No training.

For each episode:
  - S, Q from the same novel class, different scenes
  - U = M blocks from scenes disjoint from S/Q (labels never used for sampling)
  - forward U as query -> P0 per point
  - aggregate to superpoints (official .segs.json ids, remapped per block)
  - record B / C statistics

Outputs a JSONL with per-episode records + a summary JSON.
"""
import sys
import os
import json
import glob
import argparse
import pickle

sys.path.insert(0, '/data01/jh/code')
import fake_tp  # noqa: E402
sys.path.insert(0, '/data01/jh/code/COSeg')

import numpy as np
import torch
import torch.nn.functional as F

# Repro must come before any CUDA work. Without this, cuDNN picks different
# convolution algorithms in different processes/GPUs, which shifts the
# no-bridge baseline IoU of the *same episode* by up to 6pp -- enough to swamp
# every +1pp effect we are trying to measure.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from scipy.spatial import cKDTree

from util.config import load_cfg_from_cfg_file, merge_cfg_from_list
from util.data_util import voxelize
from model.coseg import COSeg
from util.common_util import load_pretrain_checkpoint

CFG_PATH = '/data01/jh/code/COSeg/config/scannetv2_COSeg_fs.yaml'
META = '/data01/jh/code/COSeg/datasets/ScanNet/meta'
NOVEL_FOLD0 = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair',
               'counter', 'curtain', 'desk', 'door', 'floor']


def load_class_names():
    names = [x.strip() for x in open(os.path.join(META, 'scannet_classnames.txt'))]
    return names


def data_prepare_seg(coord, feat, label, seg, voxel_size, voxel_max):
    coord = coord.astype(np.float32).copy()
    feat = feat.astype(np.float32)
    label = label.astype(np.int64)
    seg = seg.astype(np.int64)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label, seg = (
            coord[uniq_idx], feat[uniq_idx], label[uniq_idx], seg[uniq_idx]
        )
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = label.shape[0] // 2
        crop_idx = np.argsort(
            np.sum(np.square(coord - coord[init_idx]), 1)
        )[:voxel_max]
        coord, feat, label, seg = (
            coord[crop_idx], feat[crop_idx], label[crop_idx], seg[crop_idx]
        )
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= (coord_min + coord_max) / 2.0
    return (
        torch.FloatTensor(coord),
        torch.FloatTensor(feat) / 255.0,
        torch.LongTensor(label),
        torch.LongTensor(seg),
    )


def build_model(device='cuda:0'):
    cfg = load_cfg_from_cfg_file(CFG_PATH)
    cfg = merge_cfg_from_list(cfg, [
        'data_root', '/data01/jh/data/ScanNet/blocks_bs1_s1/data',
        'pretrain_backbone', '/data01/jh/weights/sc_s0pre',
        'weight', '/data01/jh/weights/sc0_1w1s.pth',
        'cvfold', '0', 'n_way', '1', 'k_shot', '1',
        'train_gpu', '[0]', 'multiprocessing_distributed', 'False',
    ])
    model = COSeg(cfg).cuda()
    model = load_pretrain_checkpoint(model, cfg.pretrain_backbone, 0)
    ckpt = torch.load(cfg.weight, map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    if any(k.startswith('module.') for k in sd):
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    return model, cfg


ORTH_CACHE = '/data01/jh/data/orth_cache'
_ORTH_MEM = {}


def scan_blocks(data_dir):
    """Index every block: path, scene name, present labels."""
    blocks = []
    for f in sorted(glob.glob(os.path.join(data_dir, '*.npy'))):
        d = np.load(f, mmap_mode='r')
        labels = np.asarray(d[:, 6]).astype(int)
        present = set(np.unique(labels).tolist())
        name = os.path.basename(f)[:-4]
        scene = name.split('_block_')[0]
        blocks.append({'path': f, 'name': name, 'scene': scene,
                       'classes': present, 'n': int(d.shape[0])})
    return blocks


def agg_majority(values, idx, n_regions):
    """Majority vote of integer values per region (vectorized)."""
    values = np.asarray(values).astype(np.int64)
    idx = np.asarray(idx).astype(np.int64)
    n_cls = int(values.max()) + 1
    comb = idx * n_cls + values
    counts = np.bincount(comb, minlength=n_regions * n_cls).reshape(
        n_regions, n_cls
    )
    return counts.argmax(1)


def agg_mean(t, idx, n_regions):
    """Mean of a tensor per region. Handles both (N,) and (N, C) tensors."""
    squeeze = t.dim() == 1
    if squeeze:
        t = t.unsqueeze(1)
    out = torch.zeros(n_regions, t.shape[1], dtype=t.dtype, device=t.device)
    cnt = torch.zeros(n_regions, 1, dtype=t.dtype, device=t.device)
    out.index_add_(0, idx, t)
    cnt.index_add_(
        0, idx, torch.ones(t.shape[0], 1, dtype=t.dtype, device=t.device)
    )
    res = out / cnt.clamp(min=1)
    return res.squeeze(1) if squeeze else res


def run_episode(model, cfg, s_info, q_info, u_infos, sampled_class, taus,
                proto_mode='fps', n_per=200):
    device = torch.device('cuda:0')

    def load(info):
        d = np.load(info['path'])
        return data_prepare_seg(
            d[:, 0:3], d[:, 3:6], d[:, 6], d[:, 7],
            cfg.voxel_size, cfg.voxel_max,
        )

    s_coord, s_feat, s_label, s_seg = load(s_info)
    q_coord, q_feat, q_label, q_seg = load(q_info)
    u_list = [load(i) for i in u_infos]

    s_bin = (s_label == sampled_class).long()
    q_map = (q_label == sampled_class).long()

    # concatenate U blocks into one batch
    u_coord = torch.cat([x[0] for x in u_list], 0)
    u_feat = torch.cat([x[1] for x in u_list], 0)
    u_label = torch.cat([x[2] for x in u_list], 0)
    u_seg_raw = torch.cat([x[3] for x in u_list], 0)
    u_offsets = torch.tensor([x[0].shape[0] for x in u_list]).cumsum(0).long()

    # remap seg ids to be globally unique across U blocks
    u_seg = u_seg_raw.clone()
    offset = 0
    for i, x in enumerate(u_list):
        n = x[0].shape[0]
        u_seg[offset:offset + n] = u_seg_raw[offset:offset + n] + i * 10 ** 7
        offset += n
    uniq, u_seg = torch.unique(u_seg, return_inverse=True)
    n_regions = len(uniq)

    support_x = torch.cat([s_coord, s_feat], 1).to(device)
    query_x = torch.cat([q_coord, q_feat], 1).to(device)
    u_x = torch.cat([u_coord, u_feat], 1).to(device)
    support_offset = torch.tensor([s_coord.shape[0]], dtype=torch.int32).to(device)
    u_offset = torch.tensor([u_coord.shape[0]], dtype=torch.int32).to(device)

    with torch.no_grad():
        # U as query: get per-point logits over (bg, novel)
        final_pred, _ = model(
            support_offset, support_x, s_bin.to(device),
            u_offset, u_x, torch.zeros_like(u_label).to(device),
            100, None, None, np.array([sampled_class]),
        )
        p = F.softmax(final_pred[0], dim=0)  # [2, N_u]
        p_fg = p[1]  # [N_u]

        # backbone features for similarity
        u_feat_down, u_coord_down, _, _, u_base_pred, _ = model.getFeatures(
            u_x, u_offset, torch.zeros_like(u_label).to(device), None
        )
        s_feat_down, s_coord_down, _, _, _, _ = model.getFeatures(
            support_x, support_offset, s_bin.to(device), None
        )

    # map support foreground to downsampled points via nearest neighbour
    tree_s = cKDTree(s_coord.cpu().numpy())
    _, s_nn = tree_s.query(s_coord_down.cpu().numpy())
    s_bin_down = s_bin.cpu().numpy()[s_nn]
    fg_idx = np.where(s_bin_down == 1)[0]
    if len(fg_idx) == 0:
        print('  [skip] no fg in support after downsample', flush=True)
        return None
    a_single = F.normalize(
        s_feat_down[fg_idx].mean(0, keepdim=True), dim=1
    )  # [1, C]

    # map downsampled U points back to voxel-level seg ids
    tree_u = cKDTree(u_coord.cpu().numpy())
    _, u_nn = tree_u.query(u_coord_down.cpu().numpy())
    u_seg_down = u_seg.cpu().numpy()[u_nn]

    # aggregate probability and features to superpoints
    # p_fg is per original point (final_pred is interpolated back to u_x),
    # while u_feat_down is per downsampled point. Use the matching seg mapping
    # for each of them.
    seg_full = u_seg.to(device)              # [N_u]
    seg_t = torch.LongTensor(u_seg_down).to(device)  # [N_down]
    q_region = agg_mean(p_fg, seg_full, n_regions)   # [L]
    feat_region = agg_mean(u_feat_down, seg_t, n_regions)
    feat_region = F.normalize(feat_region, dim=1)

    # superpoint GT by majority vote over original points
    gt_region = agg_majority(u_label.cpu().numpy(), u_seg.cpu().numpy(), n_regions)
    gt_mask = torch.LongTensor((gt_region == sampled_class).astype(np.int64)).to(device)

    # soft label purity: fraction of points in each superpoint with label == c
    purity_region = agg_mean(
        (u_label == sampled_class).float().to(device), seg_full, n_regions
    )

    c_hat = (q_region >= 0.5).long()  # 1-way: fg vs bg
    fg_region = gt_mask == 1
    c_region = c_hat == 0
    c_pos = c_region & fg_region
    b_mask = c_hat == 1
    b_pos = b_mask & fg_region
    b_neg = b_mask & (~fg_region)

    sim_a = (feat_region @ a_single.t()).squeeze(1)  # [L]

    rec = {
        'class': int(sampled_class),
        'n_regions': int(n_regions),
        'n_fg_regions': int(fg_region.sum().item()),
        'n_B': int(b_mask.sum().item()),
        'n_B_pos': int(b_pos.sum().item()),
        'n_B_neg': int(b_neg.sum().item()),
        'n_C': int(c_region.sum().item()),
        'n_C_pos': int(c_pos.sum().item()),
        'u_with_class': int(
            sum(1 for info in u_infos if sampled_class in info['classes'])
        ),
        # per-bridge data for Top-K analysis
        'B_q': q_region[b_mask].tolist(),
        'B_gt': fg_region[b_mask].long().tolist(),
        # global superpoint id (block-offset encoded) for cross-episode
        # stability analysis
        'B_seg': uniq[b_mask].long().tolist(),
        # block name per bridge, so the same block can be matched across
        # episodes (the block index inside U changes per episode)
        'B_block': [
            u_infos[int(s) // (10 ** 7)]['name']
            for s in uniq[b_mask].long().tolist()
        ],
        'taus': {},
    }
    b_idx = b_mask.nonzero().flatten()
    pos_local = b_pos[b_idx]
    neg_local = b_neg[b_idx]

    # ---- orthogonal signal 1: intra-U background adhesion ----
    # A false bridge (e.g. one table superpoint that looks like a counter) is
    # adjacent to other superpoints of the SAME object in the same U block,
    # which the model confidently calls background. A true bridge is not.
    try:
        block_id = (uniq.long() // (10 ** 7)).to(device)
        bg_region = (q_region < 0.2) & (~fg_region)
        eta_intra = torch.full((len(b_idx),), float('nan'), device=device)
        local_rel = torch.full((len(b_idx),), float('nan'), device=device)
        for k, r in enumerate(b_idx):
            same = (block_id == block_id[r]) & bg_region
            if same.sum() > 0:
                sims = feat_region[r] @ feat_region[same].t()
                eta_intra[k] = sims.max()
                # relative value: max [Sim(B,D) - Sim(A,D)]_+
                sims_a = (a_single @ feat_region[same].t()).squeeze(0)
                local_rel[k] = (sims - sims_a).clamp(min=0).max()
        rec['B_pos_eta_intra'] = eta_intra[pos_local].tolist()
        rec['B_neg_eta_intra'] = eta_intra[neg_local].tolist()
        rec['B_pos_local_rel'] = local_rel[pos_local].tolist()
        rec['B_neg_local_rel'] = local_rel[neg_local].tolist()
    except Exception as e:
        rec['intra_err'] = repr(e)

    # ---- orthogonal signal 2: Delta spread on C (peak-to-mean) ----
    # A true bridge activates only a few true same-class omissions (sharp
    # peak, low mean); a false bridge fires broadly over many candidates.
    # tau scan per Task 2 spec: {0.05, 0.1, 0.2}
    try:
        c_idx = c_region.nonzero().flatten()
        if len(b_idx) > 0 and len(c_idx) > 1:
            s0 = sim_a[c_idx]
            sim_bc_mat = feat_region[b_idx] @ feat_region[c_idx].t()
            qb = q_region[b_idx]
            for tau_tag, tau in (('t005', 0.05), ('t01', 0.1), ('t02', 0.2)):
                spread = torch.zeros(len(b_idx), device=device)
                d_rows = []
                for k in range(len(b_idx)):
                    sB = torch.log(
                        (torch.exp(s0) + qb[k]
                         * torch.exp(sim_bc_mat[k] / tau)) / (1 + qb[k])
                    )
                    d = sB - s0
                    d_rows.append(d)
                    top2 = d.topk(2).values.mean()
                    mean_pos = d.clamp(min=0).mean()
                    spread[k] = top2 - mean_pos
                rec['B_pos_spread_' + tau_tag] = spread[pos_local].tolist()
                rec['B_neg_spread_' + tau_tag] = spread[neg_local].tolist()
                if tau_tag == 't01':
                    rec['B_pos_spread'] = rec['B_pos_spread_t01']
                    rec['B_neg_spread'] = rec['B_neg_spread_t01']
                    # delta_promiscuity: fraction of C with delta > 0
                    dm = torch.stack(d_rows)
                    prom = (dm > 0).float().mean(dim=1)
                    rec['B_pos_promiscuity'] = prom[pos_local].tolist()
                    rec['B_neg_promiscuity'] = prom[neg_local].tolist()
                    # mean of relu(delta) on C
                    mp = dm.clamp(min=0).mean(dim=1)
                    rec['B_pos_delta_mean'] = mp[pos_local].tolist()
                    rec['B_neg_delta_mean'] = mp[neg_local].tolist()
    except Exception as e:
        rec['spread_err'] = repr(e)

    # ---- orthogonal signal C: COSeg base classifier risk ----
    try:
        base_prob_region = agg_mean(u_base_pred.max(dim=1)[0], seg_t, n_regions)
        rec['B_pos_base_prob'] = base_prob_region[b_idx][pos_local].tolist()
        rec['B_neg_base_prob'] = base_prob_region[b_idx][neg_local].tolist()
    except Exception as e:
        rec['baseprob_err'] = repr(e)

    # ---- orthogonal signal D: size + point-level consistency ----
    try:
        size_region = torch.bincount(seg_full, minlength=n_regions).float()
        q_std = torch.zeros(n_regions, device=device)
        q_p90 = torch.zeros(n_regions, device=device)
        for r in range(n_regions):
            v = p_fg[seg_full == r]
            if len(v) > 1:
                q_std[r] = v.std()
                q_p90[r] = v.quantile(0.9)
        for name, vec in (('size', size_region), ('q_std', q_std),
                          ('q_p90', q_p90)):
            rec['B_pos_' + name] = vec[b_idx][pos_local].tolist()
            rec['B_neg_' + name] = vec[b_idx][neg_local].tolist()
        rec['_q90_vec'] = q_p90.tolist()
        # point-feature variance inside each superpoint
        feat_var = torch.zeros(n_regions, device=device)
        for r in range(n_regions):
            v = u_feat_down[seg_t == r]
            if len(v) > 1:
                feat_var[r] = v.std(dim=0).mean()
        rec['B_pos_feat_var'] = feat_var[b_idx][pos_local].tolist()
        rec['B_neg_feat_var'] = feat_var[b_idx][neg_local].tolist()
    except Exception as e:
        rec['size_err'] = repr(e)

    # ---- orthogonal signal A2: pool background adhesion ----
    try:
        pool_bg = (q_region < 0.15) & (~fg_region)
        if pool_bg.sum() > 0 and len(b_idx) > 0:
            sim_pool = feat_region[b_idx] @ feat_region[pool_bg].t()
            k = min(3, sim_pool.shape[1])
            top3 = sim_pool.topk(k, dim=1).values.mean(dim=1)
            rec['B_pos_sim_pool_bg'] = top3[pos_local].tolist()
            rec['B_neg_sim_pool_bg'] = top3[neg_local].tolist()
            sim_pool_a = (a_single @ feat_region[pool_bg].t()).squeeze(0)
            top3_a = sim_pool_a.topk(min(3, sim_pool_a.shape[0])).values.mean()
            rel = top3 - top3_a
            rec['B_pos_rel_pool_bg'] = rel[pos_local].tolist()
            rec['B_neg_rel_pool_bg'] = rel[neg_local].tolist()

            # Eq.8-style risk with D_pool as the negative bank
            s0_pool = sim_a[pool_bg]
            qb = q_region[b_idx]
            risk = torch.zeros(len(b_idx), device=device)
            for kk in range(len(b_idx)):
                sB = torch.log(
                    (torch.exp(s0_pool) + qb[kk]
                     * torch.exp(sim_pool[kk] / 0.1)) / (1 + qb[kk])
                )
                d = (sB - s0_pool).clamp(min=0)
                risk[kk] = torch.logsumexp(d, dim=0) - float(np.log(max(len(d), 1)))
            rec['B_pos_delta_bg_risk'] = risk[pos_local].tolist()
            rec['B_neg_delta_bg_risk'] = risk[neg_local].tolist()
    except Exception as e:
        rec['pool_err'] = repr(e)

    # ---- upgraded negative-control signals (per-bridge, for AUC) ----
    # signal 1: does B look MORE like a base class than A does?
    # signal 3: multi-prototype consistency minus base affinity
    try:
        with torch.no_grad():
            base_p = model.base_prototypes
            valid = base_p.norm(dim=1) > 0
            sim_b_base = None
            if valid.sum() > 0:
                base_p_n = F.normalize(base_p[valid], dim=1)
                sim_b_base = feat_region @ base_p_n.t()  # [L, N_base]
                sim_a_base = a_single @ base_p_n.t()      # [1, N_base]
                delta_base = (sim_b_base - sim_a_base).max(dim=1)[0]
                rec['B_pos_delta_base'] = delta_base[b_pos].tolist()
                rec['B_neg_delta_base'] = delta_base[b_neg].tolist()
                rec['B_pos_base_abs'] = sim_b_base.max(dim=1)[0][b_pos].tolist()
                rec['B_neg_base_abs'] = sim_b_base.max(dim=1)[0][b_neg].tolist()

            protos = model.getPrototypes(
                [s_coord_down], [s_feat_down],
                [torch.BoolTensor(s_bin_down).to(device)], k=100
            )
            protos = F.normalize(protos, dim=1)
            sim_b_multi = (feat_region @ protos.t()).max(dim=1)[0]
            rec['B_pos_multi_max'] = sim_b_multi[b_pos].tolist()
            rec['B_neg_multi_max'] = sim_b_multi[b_neg].tolist()
            # unsupervised score for end-to-end (no GT), same order as b_mask
            rec['_e2e_score'] = sim_b_multi[b_mask].tolist()
            # combo score (Task 4: use the best unsupervised combination)
            try:
                if '_q90_vec' in rec:
                    qv = torch.FloatTensor(rec['_q90_vec']).to(device)
                    z = lambda v: (v - v.mean()) / (v.std() + 1e-8)
                    combo = 0.3 * z(sim_b_multi) + 0.7 * z(qv)
                    rec['_e2e_score_combo'] = combo[b_mask].tolist()
            except Exception as e:
                rec['combo_err'] = repr(e)

            # multi-prototype A version of the spread signal
            try:
                c_idx_m = c_region.nonzero().flatten()
                if len(b_idx) > 0 and len(c_idx_m) > 1:
                    s0m = sim_b_multi[c_idx_m]  # max_k Sim(C, a_k)
                    sim_bc_m = feat_region[b_idx] @ feat_region[c_idx_m].t()
                    qb_m = q_region[b_idx]
                    sp_m = torch.zeros(len(b_idx), device=device)
                    for km in range(len(b_idx)):
                        sBm = torch.log(
                            (torch.exp(s0m) + qb_m[km]
                             * torch.exp(sim_bc_m[km] / 0.1)) / (1 + qb_m[km])
                        )
                        dm = sBm - s0m
                        sp_m[km] = (dm.topk(2).values.mean()
                                    - dm.clamp(min=0).mean())
                    rec['B_pos_spread_multi'] = sp_m[pos_local].tolist()
                    rec['B_neg_spread_multi'] = sp_m[neg_local].tolist()
            except Exception as e:
                rec['spread_multi_err'] = repr(e)
            if sim_b_base is not None:
                margin = sim_b_multi - sim_b_base.max(dim=1)[0]
                rec['B_pos_margin'] = margin[b_pos].tolist()
                rec['B_neg_margin'] = margin[b_neg].tolist()
    except Exception as e:
        rec['upgrade_err'] = repr(e)

    # bridge advantage on C_pos: max sim over B_pos vs sim(A, C_pos)
    if b_pos.sum() > 0 and c_pos.sum() > 0:
        b_feat = feat_region[b_pos]
        c_feat = feat_region[c_pos]
        sim_bc = (c_feat @ b_feat.t()).max(dim=1)[0]  # [n_C_pos]
        sim_ac = sim_a[c_pos]
        rec['adv_single'] = float((sim_bc > sim_ac).float().mean().item())
        rec['mean_sim_bc'] = float(sim_bc.mean().item())
        rec['mean_sim_ac'] = float(sim_ac.mean().item())

        # false-bridge threat: do wrong bridges also attract the missed fg?
        if b_neg.sum() > 0:
            bn_feat = feat_region[b_neg]
            sim_bnc = (c_feat @ bn_feat.t()).max(dim=1)[0]
            rec['adv_neg_on_cpos'] = float(
                (sim_bnc > sim_ac).float().mean().item()
            )
            rec['mean_sim_bnc'] = float(sim_bnc.mean().item())

        # support negative controls D: max sim of bridges to support non-target
        try:
            s_seg_down_np = s_seg.cpu().numpy()[s_nn]
            neg_idx = np.where(s_bin_down == 0)[0]
            if len(neg_idx) > 0 and b_mask.sum() > 0:
                seg_np = s_seg_down_np[neg_idx]
                uniq_s, inv_s = np.unique(seg_np, return_inverse=True)
                d_feat = agg_mean(
                    s_feat_down[torch.LongTensor(neg_idx).to(device)],
                    torch.LongTensor(inv_s).to(device), len(uniq_s),
                )
                d_feat = F.normalize(d_feat, dim=1)
                sim_bd_all = (feat_region @ d_feat.t()).max(dim=1)[0]
                # per-bridge support_D similarity, for the CSR eta comparison
                rec['_support_D_sim'] = sim_bd_all[b_mask].tolist()
                rec['mean_sim_Bpos_D'] = (
                    float(sim_bd_all[b_pos].mean().item())
                    if b_pos.sum() > 0 else None
                )
                rec['mean_sim_Bneg_D'] = (
                    float(sim_bd_all[b_neg].mean().item())
                    if b_neg.sum() > 0 else None
                )
        except Exception as e:
            rec['neg_ctrl_err'] = repr(e)

        # multi-prototype baseline: COSeg's 100 sub-prototypes
        try:
            with torch.no_grad():
                protos = model.getPrototypes(
                    [s_coord_down], [s_feat_down],
                    [torch.BoolTensor(s_bin_down).to(device)], k=100
                )
            protos = F.normalize(protos, dim=1)
            sim_a_multi = (feat_region @ protos.t()).max(dim=1)[0]
            sim_ac_multi = sim_a_multi[c_pos]
            rec['adv_multi'] = float(
                (sim_bc > sim_ac_multi).float().mean().item()
            )
            rec['mean_sim_ac_multi'] = float(sim_ac_multi.mean().item())
        except Exception as e:
            rec['adv_multi'] = None
            rec['multi_err'] = repr(e)
    else:
        rec['adv_single'] = None

    for tau in taus:
        b = (q_region >= tau)
        n_b = int(b.sum().item())
        rec['taus'][str(tau)] = {
            'n_B': n_b,
            'n_B_correct': int((b & fg_region).sum().item()),
            'precision': float(
                (b & fg_region).sum().item() / max(n_b, 1)
            ),
            'soft_precision': float(purity_region[b].mean().item())
            if n_b > 0 else 0.0,
            'precision_30': float((purity_region[b] > 0.3).float().mean().item())
            if n_b > 0 else 0.0,
            'omission': float(
                c_pos.sum().item() / max(fg_region.sum().item(), 1)
            ),
        }

    # ---- Oracle bridge injection into the QUERY prediction (logit level) ----
    # Does adding the ground-truth bridge set B_pos to the COSeg query logits
    # actually improve query IoU? This is the fusion-interface sanity check.
    try:
        q_x = torch.cat([q_coord, q_feat], 1).to(device)
        q_offset = torch.tensor([q_coord.shape[0]], dtype=torch.int32).to(device)
        with torch.no_grad():
            q_pred, _ = model(
                support_offset, support_x, s_bin.to(device),
                q_offset, q_x, q_map.to(device),
                100, None, None, np.array([sampled_class]),
            )
            base_full = q_pred[0]      # [2, N_q] -- both classes, needed for
                                       # margin-space calibration
            base_logit = base_full[1]  # [N_q]
            q_map_d = q_map.to(device)
            tp0 = ((base_logit > 0) & (q_map_d == 1)).sum().item()
            fp0 = ((base_logit > 0) & (q_map_d == 0)).sum().item()
            fn0 = ((base_logit <= 0) & (q_map_d == 1)).sum().item()
            rec['oracle_base_iou'] = tp0 / max(tp0 + fp0 + fn0, 1)
            rec['oracle_base_tp'] = tp0
            rec['oracle_base_fp'] = fp0
            rec['oracle_base_fn'] = fn0
            rec['oracle_base_prec'] = tp0 / max(tp0 + fp0, 1)
            rec['oracle_base_rec'] = tp0 / max(tp0 + fn0, 1)
            # unsupervised episode-level descriptors of the direct edge S->Q.
            # These are the only quantities available at test time for deciding
            # whether the support set is already confident enough to skip the
            # bridge; the baseline IoU itself needs GT.
            rec['S_fg_ratio'] = float((base_logit > 0).float().mean())
            rec['S_p95'] = float(base_logit.quantile(0.95))
            rec['S_p90'] = float(base_logit.quantile(0.90))
            rec['S_mean'] = float(base_logit.mean())
            rec['S_std'] = float(base_logit.std())
            _fg = base_logit[base_logit > 0]
            rec['S_conf_fg'] = float(_fg.mean()) if _fg.numel() else 0.0

            def _detail(logit, tag):
                """Same IoU as _iou, plus TP/FP/FN so a drop can be split
                into a recall loss vs a precision loss."""
                pred = (logit > 0).long()
                tp = ((pred == 1) & (q_map_d == 1)).sum().item()
                fp = ((pred == 1) & (q_map_d == 0)).sum().item()
                fn = ((pred == 0) & (q_map_d == 1)).sum().item()
                rec[tag + '_tp'] = tp
                rec[tag + '_fp'] = fp
                rec[tag + '_fn'] = fn
                rec[tag + '_prec'] = tp / max(tp + fp, 1)
                rec[tag + '_rec'] = tp / max(tp + fn, 1)
                return tp / max(tp + fp + fn, 1)

            if b_pos.sum() > 0:
                q_feat_down, q_coord_down, _, _, _, _ = model.getFeatures(
                    q_x, q_offset, q_map_d, None
                )
                bp_feat = feat_region[b_pos]
                sim_bq = (F.normalize(q_feat_down, dim=1) @ bp_feat.t()).max(
                    dim=1
                )[0]
                # Task 1.5: real query-correlation gate. For every candidate B
                # (pos or neg), max similarity to any query point feature.
                try:
                    qf = F.normalize(q_feat_down, dim=1)
                    sim_all_b = (feat_region[b_mask] @ qf.t()).max(dim=1)[0]
                    rec['B_query_rel'] = sim_all_b.tolist()
                except Exception as e:
                    rec['query_rel_err'] = repr(e)
                tree_q = cKDTree(q_coord.cpu().numpy())
                _, q_nn = tree_q.query(q_coord_down.cpu().numpy())
                # repeated nearest-neighbour indices: take the max, not the last
                sim_np = np.zeros(q_coord.shape[0], dtype=np.float32)
                np.maximum.at(sim_np, q_nn, sim_bq.cpu().numpy())
                sim_full = torch.FloatTensor(sim_np).to(device)
                for lam in (0.5, 1.0, 2.0):
                    pred = (base_logit + lam * sim_full > 0).long()
                    tp = ((pred == 1) & (q_map_d == 1)).sum().item()
                    fp = ((pred == 1) & (q_map_d == 0)).sum().item()
                    fn = ((pred == 0) & (q_map_d == 1)).sum().item()
                    rec['oracle_iou_lam%.1f' % lam] = tp / max(tp + fp + fn, 1)

            if b_pos.sum() > 0:
                # gather all points that belong to a B_pos superpoint
                bp_mask = torch.zeros(u_x.shape[0], dtype=torch.bool,
                                      device=device)
                for r in b_pos.nonzero().flatten():
                    bp_mask |= (seg_full == r)
                bp_x = u_x[bp_mask]

                def _iou(logit):
                    pred = (logit > 0).long()
                    tp = ((pred == 1) & (q_map_d == 1)).sum().item()
                    fp = ((pred == 1) & (q_map_d == 0)).sum().item()
                    fn = ((pred == 0) & (q_map_d == 1)).sum().item()
                    return tp / max(tp + fp + fn, 1)

                # 2A: point-pool fusion, support fg + B_pos points
                s_fg_x = support_x[s_bin.to(device) == 1]
                mixed_x = torch.cat([s_fg_x, bp_x], 0)
                mixed_y = torch.ones(mixed_x.shape[0], dtype=torch.long,
                                     device=device)
                mixed_off = torch.tensor([mixed_x.shape[0]],
                                         dtype=torch.int32).to(device)
                lg_2a, _ = model(
                    mixed_off, mixed_x, mixed_y, q_offset, q_x, q_map_d,
                    100, None, None, np.array([sampled_class]),
                )
                rec['oracle_iou_2a'] = _iou(lg_2a[0, 1])

                # 2C: two-pass HCA, fuse calibrated logits
                def _run_bridge(pts):
                    if pts.shape[0] == 0:
                        return None
                    y = torch.ones(pts.shape[0], dtype=torch.long,
                                   device=device)
                    off = torch.tensor([pts.shape[0]],
                                       dtype=torch.int32).to(device)
                    lg, _ = model(
                        off, pts, y, q_offset, q_x, q_map_d,
                        100, None, None, np.array([sampled_class]),
                    )
                    return lg[0, 1]

                def _sel_pts(sel, balanced, nper=None):
                    """Turn selected region ids into a point set. With
                    balanced=True each superpoint contributes the same number
                    of points, so FPS cannot be dominated by one huge region
                    (a big wall/table bridge would otherwise eat most of the
                    100 prototype slots)."""
                    if not balanced:
                        m = torch.zeros(u_x.shape[0], dtype=torch.bool,
                                        device=device)
                        for r in sel:
                            m |= (seg_full == r)
                        return u_x[m]
                    nper = n_per if nper is None else nper
                    # sort once by region id, then slice each region's
                    # contiguous run, instead of one full boolean scan per
                    # region (K=20 meant 20 full passes over u_x)
                    order = torch.argsort(seg_full)
                    seg_sorted = seg_full[order]
                    counts = torch.bincount(seg_sorted).tolist()
                    starts = [0] * (len(counts) + 1)
                    for i, cnt in enumerate(counts):
                        starts[i + 1] = starts[i] + cnt
                    parts = []
                    for r in sel:
                        ri = int(r)
                        lo, hi = starts[ri], starts[ri + 1]
                        if hi <= lo:
                            continue
                        idx = order[lo:hi]
                        n = idx.numel()
                        if n > nper:
                            step = n / float(nper)
                            pick = (torch.arange(nper, device=idx.device)
                                    .float() * step).long().clamp(max=n - 1)
                            idx = idx[pick]
                        parts.append(u_x[idx])
                    if not parts:
                        return u_x[:0]
                    return torch.cat(parts, 0)

                def _gather(mask):
                    m = torch.zeros(u_x.shape[0], dtype=torch.bool,
                                    device=device)
                    for r in mask.nonzero().flatten():
                        m |= (seg_full == r)
                    return u_x[m]

                lg_b = _run_bridge(bp_x)
                if lg_b is not None:
                    for lam in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
                        fused = (1 - lam) * base_logit + lam * lg_b
                        if lam == 0.5:
                            rec['oracle_iou_2c_%.1f' % lam] = _detail(
                                fused, 'oracle_2c05')
                        else:
                            rec['oracle_iou_2c_%.1f' % lam] = _iou(fused)

                    # Task 6: asymmetric / omission-conditioned fusion.
                    # Symmetric blending pulls down query points that S already
                    # gets right, which is why ~38% of episodes lose IoU even
                    # with perfectly pure B. Two zero-parameter operators:
                    #   omit  -- keep base where S is a confident foreground
                    #   boost -- base + lam*ReLU(B - tau), can never demote
                    for tau_s in (0.0, 0.5, 1.0):
                        conf_s = base_logit > tau_s
                        for lam in (0.3, 0.5):
                            f_o = torch.where(
                                conf_s, base_logit,
                                (1 - lam) * base_logit + lam * lg_b
                            )
                            if tau_s == 0.5 and lam == 0.3:
                                rec['oracle_iou_omit%g_lam%.1f'
                                    % (tau_s, lam)] = _detail(
                                        f_o, 'oracle_omit05_03')
                            else:
                                rec['oracle_iou_omit%g_lam%.1f'
                                    % (tau_s, lam)] = _iou(f_o)
                    for tau_b in (0.0, 0.5, 1.0, 1.5):
                        for lam in (0.3, 0.5):
                            f_p = base_logit + lam * torch.clamp(
                                lg_b - tau_b, min=0
                            )
                            rec['oracle_iou_boost%g_lam%.1f'
                                % (tau_b, lam)] = _iou(f_p)

                # 2C method (a): superpoint centroids (one pseudo-point per
                # superpoint) -> getPrototypes FPS pads them to 100 slots
                try:
                    cents = []
                    for r in b_pos.nonzero().flatten():
                        cents.append(u_x[seg_full == r].mean(0))
                    if len(cents) > 0:
                        bp_cx = torch.stack(cents)
                        lg_cx = _run_bridge(bp_cx)
                        if lg_cx is not None:
                            for lam in (0.2, 0.3, 0.5):
                                rec['oracle_iou_2c_centroid_%.1f' % lam] = _iou(
                                    (1 - lam) * base_logit + lam * lg_cx
                                )
                except Exception as e:
                    rec['centroid_err'] = repr(e)

                # Oracle B_pos + bridged C_pos  (M_c = B union C, Eq.14)
                c_pos_mask = c_region & fg_region
                bcp_mask = b_pos.clone()
                if c_pos_mask.sum() > 0:
                    sim_bc_all = (
                        feat_region[c_pos_mask] @ feat_region[b_pos].t()
                    ).max(dim=1)[0]
                    sim_ac_all = sim_a[c_pos_mask]
                    bridged_idx = c_pos_mask.nonzero().flatten()[
                        sim_bc_all > sim_ac_all
                    ]
                    if len(bridged_idx) > 0:
                        bcp_mask[bridged_idx] = True
                        rec['n_cpos_bridged'] = int(len(bridged_idx))
                lg_bc = _run_bridge(_gather(bcp_mask))
                if lg_bc is not None:
                    for lam in (0.2, 0.3):
                        rec['oracle_iou_2c_bc_%.1f' % lam] = _iou(
                            (1 - lam) * base_logit + lam * lg_bc
                        )

                # Oracle all U foreground (absolute ceiling of the pool)
                lg_fg = _run_bridge(_gather(fg_region))
                if lg_fg is not None:
                    for lam in (0.2, 0.3):
                        rec['oracle_iou_2c_allfg_%.1f' % lam] = _iou(
                            (1 - lam) * base_logit + lam * lg_fg
                        )

                # ---- Task 6: reverse-support validation (RSV) ----
                # Use the candidate bridge set B as the support set and predict
                # the labelled support block S back. Y_S is training data, so
                # this gate never touches U or Q labels -- it is legitimate at
                # test time. A bridge that cannot recognise the very support
                # foreground it is supposed to extend is not trustworthy.
                try:
                    def _rsv(pts):
                        if pts.shape[0] == 0:
                            return None
                        y_r = torch.ones(pts.shape[0], dtype=torch.long,
                                         device=device)
                        off_r = torch.tensor([pts.shape[0]],
                                             dtype=torch.int32).to(device)
                        lgr, _ = model(
                            off_r, pts, y_r, support_offset, support_x,
                            s_bin.to(device), 100, None, None,
                            np.array([sampled_class]),
                        )
                        return lgr[0, 1]

                    def _rsv_stats(lg_r, tag):
                        if lg_r is None or lg_r.numel() != s_bin.numel():
                            return
                        pred_r = (lg_r > 0).long()
                        s_lab = s_bin.to(device)
                        tp = ((pred_r == 1) & (s_lab == 1)).sum().item()
                        fp = ((pred_r == 1) & (s_lab == 0)).sum().item()
                        fn = ((pred_r == 0) & (s_lab == 1)).sum().item()
                        rec['rsv_iou_' + tag] = tp / max(tp + fp + fn, 1)
                        rec['rsv_fpr_' + tag] = fp / max(
                            (s_lab == 0).sum().item(), 1)
                        rec['rsv_tp_' + tag] = tp
                        rec['rsv_fp_' + tag] = fp
                        rec['rsv_fn_' + tag] = fn
                        rec['rsv_npos_' + tag] = int((s_lab == 1).sum().item())

                    _rsv_stats(_rsv(bp_x), 'bpos')
                    _rsv_stats(_rsv(_gather(b_mask)), 'ball')
                except Exception as e:
                    rec['rsv_err'] = repr(e)

            # ---- Task 4: end-to-end with unsupervised B selection ----
            try:
                if b_mask.sum() > 0:
                    score = torch.FloatTensor(
                        rec.get('_e2e_score_combo', rec.get('_e2e_score', []))
                    ).to(device)
                    if score.numel() == b_mask.sum().item():
                        # B_pos has median 21 per episode on the same-scene
                        # runs, so K=2/3 injected only a seventh of what the
                        # oracle used. Precision is flat up to K=10, so widen
                        # the sweep instead of keeping the tiny K.
                        if proto_mode == 'both':
                            Klist = (5, 20)
                            modes = ('fps', 'balanced')
                        else:
                            Klist = (3, 5, 10, 15, 20)
                            modes = (proto_mode,)
                        for K in Klist:
                            kk = min(K, score.numel())
                            topk = torch.topk(score, kk).indices
                            sel = b_mask.nonzero().flatten()[topk]
                            nq = max(20, 1000 // kk)
                            for pm in modes:
                                suf = '' if len(modes) == 1 else '_' + pm
                                pts = _sel_pts(sel, pm == 'balanced', nq)
                                rec['e2e_top%d_npts%s' % (K, suf)] = int(
                                    pts.shape[0])
                                lg = _run_bridge(pts)
                                if lg is None:
                                    continue
                                for lam in (0.2, 0.3):
                                    fused = (1 - lam) * base_logit + lam * lg
                                    rec['e2e_top%d_lam%.1f%s'
                                        % (K, lam, suf)] = _iou(fused)
                                if K == 5 and pm == 'fps':
                                    # recovery edge B->Q vs direct edge S->Q
                                    rec['B_fg_ratio'] = float(
                                        (lg > 0).float().mean())
                                    rec['B_p95'] = float(lg.quantile(0.95))
                                    _bfg = lg[lg > 0]
                                    rec['B_conf_fg'] = float(
                                        _bfg.mean()) if _bfg.numel() else 0.0
                                    pa = (lg > 0)
                                    pb = (base_logit > 0)
                                    inter = (pa & pb).sum().item()
                                    union = (pa | pb).sum().item()
                                    rec['agree_SB'] = inter / max(union, 1)
                                    rec['B_fg_gain'] = float(
                                        (lg > 0).float().mean()
                                        - (base_logit > 0).float().mean())
                                if K in (15, 20):
                                    conf_s = base_logit > 0.5
                                    rec['e2e_top%d_omit_lam0.3%s'
                                        % (K, suf)] = _iou(
                                        torch.where(
                                            conf_s, base_logit,
                                            (1 - 0.3) * base_logit + 0.3 * lg))
                                    rec['e2e_top%d_boost_lam0.3%s'
                                        % (K, suf)] = _iou(
                                        base_logit + 0.3
                                        * torch.clamp(lg, min=0))

                    # No selection at all: every candidate bridge. Tells us
                    # whether selecting is worth anything versus just using
                    # the whole pool.
                    try:
                        m_all = torch.zeros(u_x.shape[0], dtype=torch.bool,
                                            device=device)
                        for r in b_mask.nonzero().flatten():
                            m_all |= (seg_full == r)
                        lg_allb = _run_bridge(u_x[m_all])
                        if lg_allb is not None:
                            for lam in (0.1, 0.2, 0.3):
                                rec['e2e_all_lam%.1f' % lam] = _iou(
                                    (1 - lam) * base_logit + lam * lg_allb
                                )
                    except Exception as e:
                        rec['e2e_all_err'] = repr(e)

                    # Naive Top-K by q_c(R) only (Task 4 item 2)
                    qc_score = q_region[b_mask]
                    if qc_score.numel() == b_mask.sum().item():
                        for K in (3, 5, 10, 15, 20):
                            kk = min(K, qc_score.numel())
                            topk = torch.topk(qc_score, kk).indices
                            sel = b_mask.nonzero().flatten()[topk]
                            nq = max(20, 1000 // kk)
                            pts_q = _sel_pts(sel, proto_mode == 'balanced', nq)
                            lgq = _run_bridge(pts_q)
                            if lgq is not None:
                                for lam in (0.2, 0.3):
                                    fused_q = ((1 - lam) * base_logit
                                               + lam * lgq)
                                    rec['e2e_qc_top%d_lam%.1f'
                                        % (K, lam)] = _iou(fused_q)
                                if K in (15, 20):
                                    conf_s = base_logit > 0.5
                                    rec['e2e_qc_top%d_omit_lam0.3' % K] = _iou(
                                        torch.where(
                                            conf_s, base_logit,
                                            (1 - 0.3) * base_logit + 0.3 * lgq)
                                    )
                                    rec['e2e_qc_top%d_boost_lam0.3' % K] = _iou(
                                        base_logit + 0.3 * torch.clamp(lgq, min=0)
                                    )
            except Exception as e:
                rec['e2e_err'] = repr(e)

            # ---- Task 11: orthogonal physical-appearance referee ----
            # F0 was trained on this data, so any signal derived from its own
            # features is self-confirming (that is the 0.61 AUC ceiling).
            # Raw RGB plus relative height live outside F0 and therefore give
            # an independent opinion on whether a bridge looks like the
            # support foreground.
            try:
                if b_mask.sum() > 0:
                    # do the whole aggregation on CPU in one shot: mixing
                    # devices here kept tripping index_add_
                    u_rgb_c = torch.clamp(u_feat.cpu(), 0, 1).float()
                    uz_c = u_coord[:, 2].float().cpu()
                    uz_c = (uz_c - uz_c.min()) / (uz_c.max() - uz_c.min()
                                                  + 1e-6)
                    seg_c = seg_full.cpu().long()
                    cnt_c = torch.bincount(seg_c, minlength=n_regions).float()
                    denom = cnt_c.clamp(min=1).unsqueeze(1)
                    rgb_r = (torch.zeros(n_regions, 3).index_add_(
                        0, seg_c, u_rgb_c) / denom).to(device)
                    z_r = (torch.zeros(n_regions, 1).index_add_(
                        0, seg_c, uz_c.unsqueeze(1)) / denom).to(device)

                    srgb = torch.clamp(s_feat.to(device), 0, 1)
                    sz = s_coord.to(device)[:, 2]
                    sz_n = (sz - sz.min()) / (sz.max() - sz.min() + 1e-6)
                    # point-level support foreground prototype: avoids the
                    # superpoint aggregation on the support side, which mixed
                    # CPU/CUDA indices
                    fg_pts = (s_label.to(device) == sampled_class)
                    if int(fg_pts.sum()) > 0:
                        pA = torch.cat([srgb[fg_pts].mean(0),
                                        sz_n[fg_pts].mean(0).unsqueeze(0)], 0)
                        fU = torch.cat([rgb_r, z_r], 1)          # [R,4]
                        fU_n = F.normalize(fU, dim=1)
                        pA_n = F.normalize(pA.unsqueeze(0), dim=1)
                        sim_phys = (fU_n @ pA_n.t()).squeeze(1)  # [R]
                        bg_u = (q_region < 0.15)
                        if int(bg_u.sum()) > 0:
                            fD = fU_n[bg_u]
                            sim_BD = fU_n[b_mask] @ fD.t()
                            sim_AD = pA_n @ fD.t()
                            rel = torch.clamp(sim_BD - sim_AD, min=0)
                            _T = 0.1
                            eta = _T * (torch.logsumexp(rel / _T, dim=1)
                                        - np.log(max(rel.shape[1], 1)))
                        else:
                            eta = torch.zeros(int(b_mask.sum()),
                                              device=device)
                        rec['phys_s_pos'] = sim_phys[b_mask].tolist()
                        rec['phys_eta'] = eta.tolist()
                        rec['phys_V'] = (sim_phys[b_mask] - eta).tolist()
                        rec['phys_qc'] = q_region[b_mask].tolist()
            except Exception as e:
                rec['phys_err'] = repr(e)

            # ---- Task 13: orthogonal consensus gate (Agree / Gain) ----
            try:
                if b_mask.sum() > 0 and ORTH_CACHE:
                    qnm = q_info['name']
                    if qnm not in _ORTH_MEM:
                        pth = os.path.join(ORTH_CACHE, qnm + '.npz')
                        if os.path.exists(pth):
                            z = np.load(pth)
                            _ORTH_MEM[qnm] = (torch.from_numpy(
                                z['feat'].astype(np.float32)),
                                torch.from_numpy(z['seg']))
                    snm2 = s_info['name']
                    if snm2 not in _ORTH_MEM:
                        pth = os.path.join(ORTH_CACHE, snm2 + '.npz')
                        if os.path.exists(pth):
                            z = np.load(pth)
                            _ORTH_MEM[snm2] = (torch.from_numpy(
                                z['feat'].astype(np.float32)),
                                torch.from_numpy(z['seg']))
                    rec['g13_qc'] = int(qnm in _ORTH_MEM)
                    rec['g13_sc'] = int(snm2 in _ORTH_MEM)
                    if qnm in _ORTH_MEM and snm2 in _ORTH_MEM:
                        qf_r, qs_r = _ORTH_MEM[qnm]
                        sf_r, ss_r = _ORTH_MEM[snm2]
                        # point-level CLIP via the superpoint lookup
                        qseg = q_seg.long().cpu()
                        qs_r = qs_r.cpu()
                        idx = torch.searchsorted(
                            qs_r.contiguous(), qseg.contiguous()).long()
                        idx = torch.clamp(idx, 0, max(len(qs_r) - 1, 0))
                        valid = qs_r[idx] == qseg
                        fpts = qf_r[idx]  # [Nq, 768]

                        # support prototype and background set in CLIP space
                        sseg_l = s_seg.long().cpu()
                        ss_r = ss_r.cpu()
                        # s_seg carries raw ScanNet segment ids (up to ~17k),
                        # so map them to contiguous indices first
                        s_uniq, s_inv = torch.unique(sseg_l,
                                                     return_inverse=True)
                        n_sr2 = int(len(s_uniq))
                        sreg2 = agg_majority(s_label.numpy(),
                                             s_inv.numpy(), n_sr2)
                        kidx = torch.searchsorted(
                            ss_r.contiguous(), s_uniq.contiguous()).long()
                        kidx = torch.clamp(kidx, 0, max(len(ss_r) - 1, 0))
                        kvalid = ss_r[kidx] == s_uniq
                        fg_r = [k for k in range(n_sr2)
                                if sreg2[k] == sampled_class and
                                bool(kvalid[k])]
                        bg_r = [k for k in range(n_sr2)
                                if sreg2[k] != sampled_class and
                                bool(kvalid[k])]
                        rec['g13_fg'] = len(fg_r)
                        rec['g13_bg'] = len(bg_r)
                        if fg_r:
                            pA13 = F.normalize(
                                sf_r[[int(kidx[k]) for k in fg_r]].mean(
                                    0, keepdim=True), dim=1)
                            D13 = (F.normalize(sf_r[[int(kidx[k])
                                                     for k in bg_r]], dim=1)
                                   if bg_r else None)
                            fpn = F.normalize(fpts, dim=1)
                            mS = (base_logit > 0).cpu() & valid
                            if int(mS.sum()) > 0:
                                vS = (fpn[mS] @ pA13.t()).mean()
                                agree = vS
                                if D13 is not None:
                                    agree = agree - (fpn[mS] @ D13.t()).max()
                                rec['agree_orth'] = float(agree)
                            # bridge path on Q (K=5 by q_c)
                            qc13 = q_region[b_mask]
                            kk13 = min(5, int(qc13.numel()))
                            t13 = torch.topk(qc13, kk13).indices
                            sel13 = b_mask.nonzero().flatten()[t13]
                            m13 = torch.zeros(u_x.shape[0], dtype=torch.bool,
                                              device=device)
                            for r in sel13:
                                m13 |= (seg_full == r)
                            lg13 = _run_bridge(u_x[m13])
                            if lg13 is not None:
                                mB = (lg13 > 0).cpu() & valid
                                if int(mB.sum()) > 0 and int(mS.sum()) > 0:
                                    vB = (fpn[mB] @ pA13.t()).mean()
                                    rec['gain_orth'] = float(vB - vS)
                                    rec['agree_orth_B'] = float(vB)
            except Exception as e:
                import traceback
                rec['gate13_err'] = repr(e)
                rec['gate13_tb'] = traceback.format_exc()[-400:]

            # ---- Task 12: orthogonal CLIP referee + centering + dual-expert arbitration ----
            try:
                if b_mask.sum() > 0 and ORTH_CACHE:
                    # per-region CLIP feature for U, looked up through the
                    # remapped superpoint ids (segment i carries +i*1e7)
                    fU = torch.zeros(n_regions, 768)
                    uq = uniq  # remapped ids (u_seg was overwritten by the
                    # inverse index, so uniq is the only place the original
                    # segment ids survive)
                    got = 0
                    for j in range(len(uq)):
                        v = int(uq[j])
                        bi = v // 10 ** 7
                        sid = v % 10 ** 7
                        if bi >= len(u_infos):
                            continue
                        nm = u_infos[bi]['name']
                        if nm not in _ORTH_MEM:
                            pth = os.path.join(ORTH_CACHE, nm + '.npz')
                            if os.path.exists(pth):
                                z = np.load(pth)
                                _ORTH_MEM[nm] = (torch.from_numpy(
                                    z['feat'].astype(np.float32)),
                                    torch.from_numpy(z['seg']))
                        if nm not in _ORTH_MEM:
                            continue
                        fb, sb = _ORTH_MEM[nm]
                        pos = int(torch.searchsorted(sb, torch.tensor(sid)))
                        if pos < len(sb) and int(sb[pos]) == sid:
                            fU[j] = fb[pos]
                            got += 1
                    rec['clip_got'] = got
                    rec['clip_nreg'] = int(n_regions)
                    rec['clip_nu'] = len(u_infos)
                    if got > 0:
                        # WARM-style episode centering: subtract the pool mean
                        # to remove common background bias, then L2 normalise
                        mu = fU.mean(0, keepdim=True)
                        fUc = F.normalize(fU - mu, dim=1)
                        fUraw = F.normalize(fU, dim=1)

                        # support foreground prototype in CLIP space
                        sfg = (s_label.to(device) == sampled_class)
                        if int(sfg.sum()) > 0:
                            # S features are not cached; approximate with the
                            # CLIP features of the support block itself
                            snm = s_info['name']
                            if snm not in _ORTH_MEM:
                                pth = os.path.join(ORTH_CACHE, snm + '.npz')
                                if os.path.exists(pth):
                                    z = np.load(pth)
                                    _ORTH_MEM[snm] = (torch.from_numpy(
                                        z['feat'].astype(np.float32)),
                                        torch.from_numpy(z['seg']))
                            if snm in _ORTH_MEM:
                                fb, sb = _ORTH_MEM[snm]
                                sseg_d = s_seg.to(device)
                                slag = s_label.to(device)
                                # region-level majority label for support
                                n_sr = int(sseg_d.max()) + 1
                                sreg = agg_majority(
                                    slag.cpu().numpy(),
                                    sseg_d.cpu().numpy(), n_sr)
                                keep = [k for k in range(n_sr)
                                        if sreg[k] == sampled_class]
                                if keep:
                                    prot = []
                                    for k in keep:
                                        pos = int(torch.searchsorted(
                                            sb, torch.tensor(k)))
                                        if pos < len(sb) and int(sb[pos]) == k:
                                            prot.append(fb[pos])
                                    if prot:
                                        pA = torch.stack(prot).mean(0)
                                        pA = F.normalize(pA.unsqueeze(0),
                                                         dim=1)
                                        for tagname, fX in (('c', fUc),
                                                            ('r', fUraw)):
                                            sp = (fX[b_mask] @ pA.t()).squeeze(1)
                                            bgm = (q_region < 0.15)
                                            if int(bgm.sum()) > 0:
                                                fD = fX[bgm]
                                                sBD = fX[b_mask] @ fD.t()
                                                sAD = pA @ fD.t()
                                                rel = torch.clamp(sBD - sAD,
                                                                  min=0)
                                                _T = 0.1
                                                eta = _T * (
                                                    torch.logsumexp(
                                                        rel / _T, dim=1)
                                                    - np.log(
                                                        max(rel.shape[1], 1)))
                                            else:
                                                eta = torch.zeros(
                                                    int(b_mask.sum()),
                                                    device=device)
                                            rec['clip_s_pos_' + tagname] =                                                 sp.tolist()
                                            rec['clip_eta_' + tagname] =                                                 eta.tolist()
                                            rec['clip_V_' + tagname] =                                                 (sp - eta).tolist()
                                        rec['clip_qc'] = q_region[b_mask].tolist()
            except Exception as e:
                rec['clip_err'] = repr(e)

            # ---- Task 14: orthogonal veto as the gate (end-to-end) ----
            try:
                if rec.get('clip_V_c') and b_mask.sum() > 0:
                    Vv = torch.tensor(rec['clip_V_c'], device=device)
                    bb = b_mask.nonzero().flatten()
                    order = torch.argsort(Vv, descending=True)
                    for k in (1, 2, 3, 5):
                        kk = min(k, order.numel())
                        idx = bb[order[:kk]]
                        m = torch.zeros(u_x.shape[0], dtype=torch.bool,
                                        device=device)
                        for r in idx:
                            m |= (seg_full == r)
                        lgv = _run_bridge(u_x[m])
                        if lgv is None:
                            continue
                        for lam in (0.2, 0.3):
                            rec['veto_top%d_lam%.1f' % (k, lam)] = _iou(
                                (1 - lam) * base_logit + lam * lgv)
                    # strict veto: keep only V>0, else close the bridge
                    pos = order[Vv[order] > 0]
                    if pos.numel() > 0:
                        idx = bb[pos]
                        m = torch.zeros(u_x.shape[0], dtype=torch.bool,
                                        device=device)
                        for r in idx:
                            m |= (seg_full == r)
                        lgv = _run_bridge(u_x[m])
                        if lgv is not None:
                            for lam in (0.2, 0.3):
                                rec['veto_pos_lam%.1f' % lam] = _iou(
                                    (1 - lam) * base_logit + lam * lgv)
                            rec['veto_pos_nB'] = int(pos.numel())
                    else:
                        rec['veto_pos_nB'] = 0
                        for lam in (0.2, 0.3):
                            rec['veto_pos_lam%.1f' % lam] = rec[
                                'oracle_base_iou']
            except Exception as e:
                rec['veto_err'] = repr(e)

            # ---- Task 10: query-level logit calibration + soft gate ----
            # Task 9 showed the unsupervised features correlate only ~0.35
            # with the outcome. Instead of threshold search, correct the
            # known failure mode directly: when the bridge path floods the
            # query with foreground, its global margin mean is inflated.
            try:
                if b_mask.sum() > 0:
                    score_qc = q_region[b_mask]
                    kk = min(5, int(score_qc.numel()))
                    if kk > 0:
                        topk = torch.topk(score_qc, kk).indices
                        sel = b_mask.nonzero().flatten()[topk]
                        m_sel = torch.zeros(u_x.shape[0], dtype=torch.bool,
                                            device=device)
                        for r in sel:
                            m_sel |= (seg_full == r)
                        pts_c = u_x[m_sel]
                        y_c = torch.ones(pts_c.shape[0], dtype=torch.long,
                                         device=device)
                        off_c = torch.tensor([pts_c.shape[0]],
                                             dtype=torch.int32).to(device)
                        with torch.no_grad():
                            lg2, _ = model(
                                off_c, pts_c, y_c, q_offset, q_x, q_map_d,
                                100, None, None, np.array([sampled_class]),
                            )
                        m_S = base_full[1] - base_full[0]
                        m_B = lg2[0][1] - lg2[0][0]

                        def _iou_m(m):
                            pred = (m > 0).long()
                            tp = ((pred == 1) & (q_map_d == 1)).sum().item()
                            fp = ((pred == 1) & (q_map_d == 0)).sum().item()
                            fn = ((pred == 0) & (q_map_d == 1)).sum().item()
                            return tp / max(tp + fp + fn, 1)

                        # A: remove the positive global shift only
                        shift = torch.clamp(m_B.mean() - m_S.mean(), min=0)
                        m_Ba = m_B - shift
                        # B: full z-score distribution alignment
                        m_Bb = ((m_B - m_B.mean()) / (m_B.std() + 1e-5)
                                * m_S.std() + m_S.mean())
                        for lam in (0.2, 0.3):
                            rec['calA_top5_lam%.1f' % lam] = _iou_m(
                                (1 - lam) * m_S + lam * m_Ba)
                            rec['calB_top5_lam%.1f' % lam] = _iou_m(
                                (1 - lam) * m_S + lam * m_Bb)
                        rec['cal_shift'] = float(shift)
                        rec['cal_meanS'] = float(m_S.mean())
                        rec['cal_meanB'] = float(m_B.mean())

                        # soft gate: no threshold search at all
                        s_conf = float(rec.get('S_conf_fg', 0.0))
                        b_fg = float((m_B > 0).float().mean())
                        for l0 in (0.3, 0.5):
                            w = (max(0.0, min(1.0, 1.0 - s_conf / 1.5))
                                 * max(0.0, min(1.0, 1.0 - b_fg / 0.5)))
                            lam_ep = l0 * w
                            rec['soft%d_top5' % int(l0 * 10)] = _iou_m(
                                (1 - lam_ep) * m_S + lam_ep * m_B)
                        # fixed guard rail, no tuning
                        for lam in (0.2, 0.3):
                            if s_conf < 0.75 and b_fg < 0.35:
                                rec['guard_top5_lam%.1f' % lam] = _iou_m(
                                    (1 - lam) * m_S + lam * m_B)
                            else:
                                rec['guard_top5_lam%.1f'
                                    % lam] = rec['oracle_base_iou']
            except Exception as e:
                rec['cal_err'] = repr(e)

            # ---- Task 3: CSR recovery on C ----
            try:
                c_idx_csr = c_region.nonzero().flatten()
                if len(c_idx_csr) > 0 and len(b_idx) > 0:
                    tau = 0.1
                    s0 = sim_a[c_idx_csr]
                    sim_bc_mat = feat_region[b_idx] @ feat_region[c_idx_csr].t()
                    qb = q_region[b_idx]
                    delta_mat = torch.zeros_like(sim_bc_mat)
                    for kk in range(len(b_idx)):
                        sB = torch.log(
                            (torch.exp(s0) + qb[kk]
                             * torch.exp(sim_bc_mat[kk] / tau))
                            / (1 + qb[kk])
                        )
                        delta_mat[kk] = sB - s0

                    gt_c_np = gt_region[c_idx_csr.cpu().numpy()] == sampled_class
                    gt_c = torch.BoolTensor(gt_c_np).to(device)
                    eta_opts = {'none': torch.zeros(len(b_idx), device=device)}
                    if 'mean_sim_Bneg_D' in rec and 'mean_sim_Bpos_D' in rec:
                        pass  # support_D not per-bridge here
                    if 'B_pos_delta_bg_risk' in rec:
                        # per-bridge risk is already in rec for B only
                        pass
                    # use base_prob as eta (per-bridge, available)
                    bp_region = agg_mean(
                        u_base_pred.max(dim=1)[0], seg_t, n_regions
                    )
                    eta_opts['base_prob'] = bp_region[b_idx]
                    # original support-negative eta (per-bridge)
                    if '_support_D_sim' in rec:
                        eta_opts['support_D'] = torch.FloatTensor(
                            rec['_support_D_sim']
                        ).to(device)
                    # pool background risk (recompute cheaply)
                    pool_bg2 = (q_region < 0.15) & (~fg_region)
                    if pool_bg2.sum() > 0:
                        s0p = sim_a[pool_bg2]
                        sim_bp = feat_region[b_idx] @ feat_region[pool_bg2].t()
                        risk2 = torch.zeros(len(b_idx), device=device)
                        for kk in range(len(b_idx)):
                            sB = torch.log(
                                (torch.exp(s0p) + qb[kk]
                                 * torch.exp(sim_bp[kk] / tau))
                                / (1 + qb[kk])
                            )
                            d = (sB - s0p).clamp(min=0)
                            risk2[kk] = torch.logsumexp(d, dim=0) - float(
                                np.log(max(len(d), 1))
                            )
                        eta_opts['pool_bg'] = risk2

                    # combined eta* = base_prob + pool_bg + local_bg
                    if 'pool_bg' in eta_opts and 'base_prob' in eta_opts:
                        try:
                            combined = (eta_opts['base_prob']
                                        + eta_opts['pool_bg'])
                            if ('B_pos_eta_intra' in rec
                                    and 'B_neg_eta_intra' in rec):
                                full = torch.zeros(len(b_idx), device=device)
                                pv = rec['B_pos_eta_intra']
                                nv = rec['B_neg_eta_intra']
                                full[pos_local] = torch.FloatTensor(
                                    [0.0 if np.isnan(x) else x for x in pv]
                                ).to(device)
                                full[neg_local] = torch.FloatTensor(
                                    [0.0 if np.isnan(x) else x for x in nv]
                                ).to(device)
                                combined = combined + full
                            eta_opts['combined'] = combined
                        except Exception as e:
                            rec['combined_err'] = repr(e)

                    for name, eta in eta_opts.items():
                        V = delta_mat - eta[:, None]
                        Vp = V.clamp(min=0)
                        alpha = Vp / (0.5 + Vp.sum(dim=0, keepdim=True))
                        g = (alpha * Vp).sum(dim=0)
                        recovered = g > 0
                        tp = (recovered & gt_c).sum().item()
                        fp = (recovered & (~gt_c)).sum().item()
                        fn = ((~recovered) & gt_c).sum().item()
                        tn = ((~recovered) & (~gt_c)).sum().item()
                        rec['csr_%s_recall' % name] = tp / max(tp + fn, 1)
                        rec['csr_%s_false' % name] = fp / max(fp + tn, 1)
                        rec['csr_%s_precision' % name] = tp / max(tp + fp, 1)

                        # Task 4c: end-to-end with B + recovered C
                        if name == 'pool_bg':
                            try:
                                sc = torch.FloatTensor(
                                    rec.get('_e2e_score_combo', rec.get('_e2e_score', []))
                                ).to(device)
                                if sc.numel() == len(b_idx):
                                    top2 = torch.topk(
                                        sc, min(2, sc.numel())
                                    ).indices
                                    sel_b = b_idx[top2]
                                    sel_c = c_idx_csr[recovered]
                                    all_sel = torch.cat([sel_b, sel_c])
                                    m = torch.zeros(u_x.shape[0],
                                                    dtype=torch.bool,
                                                    device=device)
                                    for r in all_sel:
                                        m |= (seg_full == r)
                                    pts = u_x[m]
                                    if pts.shape[0] > 0:
                                        y2 = torch.ones(
                                            pts.shape[0], dtype=torch.long,
                                            device=device
                                        )
                                        off2 = torch.tensor(
                                            [pts.shape[0]],
                                            dtype=torch.int32
                                        ).to(device)
                                        qx2 = torch.cat(
                                            [q_coord, q_feat], 1
                                        ).to(device)
                                        qo2 = torch.tensor(
                                            [q_coord.shape[0]],
                                            dtype=torch.int32
                                        ).to(device)
                                        qm2 = q_map.to(device)
                                        lg_bc, _ = model(
                                            off2, pts, y2, qo2, qx2, qm2,
                                            100, None, None,
                                            np.array([sampled_class]),
                                        )
                                        lg_s0, _ = model(
                                            support_offset,
                                            support_x, s_bin.to(device),
                                            qo2, qx2, qm2,
                                            100, None, None,
                                            np.array([sampled_class]),
                                        )
                                        for lam in (0.2, 0.3):
                                            f = ((1 - lam) * lg_s0[0, 1]
                                                 + lam * lg_bc[0, 1])
                                            pr = (f > 0).long()
                                            tp2 = ((pr == 1)
                                                   & (qm2 == 1)).sum().item()
                                            fp2 = ((pr == 1)
                                                   & (qm2 == 0)).sum().item()
                                            fn2 = ((pr == 0)
                                                   & (qm2 == 1)).sum().item()
                                            rec['e2e_bc_lam%.1f' % lam] = (
                                                tp2 / max(tp2 + fp2 + fn2, 1)
                                            )
                            except Exception as e:
                                rec['e2e_bc_err'] = repr(e)
            except Exception as e:
                rec['csr_err'] = repr(e)
    except Exception as e:
        rec['oracle_err'] = repr(e)

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/data01/jh/data/ScanNet/blocks_bs1_s1/data')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--M', type=int, default=3)
    ap.add_argument('--taus', default='0.5,0.6,0.7,0.8')
    ap.add_argument('--out', default='/data01/jh/data/diag_bc')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--manifest', default='',
                    help='fixed paired episode list (see make_manifest.py); '
                         'when set, episodes come from the file instead of '
                         'being resampled every run')
    ap.add_argument('--proto-mode', default='fps',
                    choices=['fps', 'balanced', 'both'],
                    help='balanced gives every selected superpoint the same '
                         'number of points before FPS, so huge bridges cannot '
                         'monopolise the 100 prototype slots')
    ap.add_argument('--n-per', type=int, default=200,
                    help='points per superpoint in balanced mode')
    ap.add_argument('--u-mode', default='cross_scene',
                    choices=['cross_scene', 'same_scene', 'self'],
                    help='where the unlabeled set U comes from; all of them are '
                         'drawn from blocks containing the target class so the '
                         'bridge set is never trivially empty')
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    class_names = load_class_names()
    name2id = {n: i for i, n in enumerate(class_names)}
    novel_ids = [name2id[n] for n in NOVEL_FOLD0]
    taus = [float(x) for x in args.taus.split(',')]

    print('scanning blocks...', flush=True)
    blocks = scan_blocks(args.data)
    print('total blocks:', len(blocks), flush=True)

    class2blocks = {c: [b for b in blocks if c in b['classes']] for c in novel_ids}
    for c in novel_ids:
        print('  class', c, class_names[c], '->', len(class2blocks[c]), 'blocks', flush=True)

    model, cfg = build_model()
    print('model loaded', flush=True)

    manifest = None
    if args.manifest:
        with open(args.manifest) as mf:
            manifest = json.load(mf)['episodes']
        info_by_name = {b['name']: b for b in blocks}
        print('manifest episodes:', len(manifest), flush=True)

    records = []
    n_loop = (min(len(manifest), args.episodes) if manifest
              else args.episodes)
    for ep in range(n_loop):
        if manifest is not None:
            # Reset per-episode so the RNG state cannot drift between runs.
            # FPS consumes draws, and different prototype modes consume
            # different amounts, which used to desynchronise S/Q sampling and
            # made the same manifest episode give different baselines.
            np.random.seed(10000 + ep)
            torch.manual_seed(10000 + ep)
            spec = manifest[ep]
            c = spec['class']
            s_info = info_by_name[spec['support']]
            q_info = info_by_name[spec['query']]
            mkey = 'u_cross' if args.u_mode == 'cross_scene' else 'u_same'
            u_infos = [info_by_name[n] for n in spec.get(mkey, [])
                       if n in info_by_name]
            if len(u_infos) < 1:
                continue
            M_eff = len(u_infos)
            try:
                rec = run_episode(model, cfg, s_info, q_info, u_infos, c, taus,
                                  proto_mode=args.proto_mode, n_per=args.n_per)
            except Exception as e:
                import traceback
                print('ep', ep, 'failed:', repr(e), flush=True)
                traceback.print_exc()
                raise
            if rec is None:
                continue
            rec['episode'] = ep
            rec['manifest_id'] = ep
            rec['support'] = s_info['name']
            rec['query'] = q_info['name']
            rec['u'] = [b['name'] for b in u_infos]
            rec['u_mode'] = args.u_mode
            rec['u_M'] = len(u_infos)
            rec['proto_mode'] = args.proto_mode
            rec['s_q_same_scene'] = int(s_info['scene'] == q_info['scene'])
            records.append(rec)
            if (ep + 1) % 10 == 0:
                print('ep', ep + 1, '/', n_loop, 'ok', len(records),
                      flush=True)
                os.makedirs(args.out, exist_ok=True)
                with open(os.path.join(args.out, 'records.jsonl'), 'w') as f:
                    for r in records:
                        f.write(json.dumps(r) + chr(10))
            continue
        c = int(np.random.choice(novel_ids))
        cand = class2blocks[c]
        if len(cand) < 2:
            print('  [skip] too few blocks for class', c, flush=True)
            continue
        np.random.shuffle(cand)
        s_info, q_info = cand[0], cand[1]
        used_scenes = {s_info['scene'], q_info['scene']}
        # oracle U sampling: blocks that contain the target class. The scene
        # constraint is the whole point of this knob -- cross_scene is the
        # strict leakage-free setting we ran first, same_scene keeps U inside
        # the query scene, self makes U the query block itself.
        if args.u_mode == 'cross_scene':
            u_pool = [b for b in class2blocks[c] if b['scene'] not in used_scenes]
        elif args.u_mode == 'same_scene':
            u_pool = [b for b in class2blocks[c]
                      if b['scene'] == q_info['scene']
                      and b['name'] not in (q_info['name'], s_info['name'])]
        else:
            u_pool = [q_info]
        if len(u_pool) < args.M:
            if args.u_mode == 'cross_scene':
                # keep the original strict behaviour so the new runs stay
                # comparable with the 504-episode shards already on disk
                print('  [skip] too few U blocks', flush=True)
                continue
            # small scenes cannot supply M same-scene blocks; take what is
            # there instead of dropping the episode, and record the real M
        M_eff = min(args.M, len(u_pool))
        if M_eff < 1:
            print('  [skip] too few U blocks', flush=True)
            continue
        pick = np.random.choice(len(u_pool), M_eff, replace=False)
        u_infos = [u_pool[i] for i in pick]

        try:
            rec = run_episode(model, cfg, s_info, q_info, u_infos, c, taus,
                              proto_mode=args.proto_mode, n_per=args.n_per)
        except Exception as e:
            import traceback
            print('ep', ep, 'failed:', repr(e), flush=True)
            traceback.print_exc()
            if ep >= 0:
                raise
            continue
        if rec is None:
            continue
        rec['episode'] = ep
        rec['support'] = s_info['name']
        rec['query'] = q_info['name']
        rec['u'] = [b['name'] for b in u_infos]
        rec['u_mode'] = args.u_mode
        rec['u_M'] = len(u_infos)
        rec['proto_mode'] = args.proto_mode
        rec['s_q_same_scene'] = int(s_info['scene'] == q_info['scene'])
        records.append(rec)
        if (ep + 1) % 10 == 0:
            print('ep', ep + 1, '/', args.episodes, 'ok', len(records), flush=True)
            # incremental save so a killed shard keeps its finished episodes
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, 'records.jsonl'), 'w') as f:
                for r in records:
                    f.write(json.dumps(r) + '\n')

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'records.jsonl'), 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    # summary
    summary = {'n_episodes': len(records), 'taus': {}}
    for tau in taus:
        vals = [r['taus'][str(tau)] for r in records]
        summary['taus'][str(tau)] = {
            'B_nonempty_rate': float(np.mean([v['n_B'] > 0 for v in vals])),
            'B_precision': float(np.mean([v['precision'] for v in vals])),
            'B_soft_precision': float(np.mean([v['soft_precision'] for v in vals])),
            'B_precision_30': float(np.mean([v['precision_30'] for v in vals])),
            'omission_ratio': float(np.mean([v['omission'] for v in vals])),
        }
    adv = [r['adv_single'] for r in records if r.get('adv_single') is not None]
    summary['advantage_single'] = float(np.mean(adv)) if adv else None
    summary['n_adv_episodes'] = len(adv)
    adv_m = [r['adv_multi'] for r in records if r.get('adv_multi') is not None]
    summary['advantage_multi'] = float(np.mean(adv_m)) if adv_m else None
    adv_neg = [r['adv_neg_on_cpos'] for r in records
               if r.get('adv_neg_on_cpos') is not None]
    summary['adv_neg_on_cpos'] = float(np.mean(adv_neg)) if adv_neg else None
    bp_d = [r['mean_sim_Bpos_D'] for r in records
            if r.get('mean_sim_Bpos_D') is not None]
    bn_d = [r['mean_sim_Bneg_D'] for r in records
            if r.get('mean_sim_Bneg_D') is not None]
    summary['mean_sim_Bpos_D'] = float(np.mean(bp_d)) if bp_d else None
    summary['mean_sim_Bneg_D'] = float(np.mean(bn_d)) if bn_d else None
    summary['n_neg_ctrl_episodes'] = len(bp_d)
    u_ok = [r.get('u_with_class', 0) == 3 for r in records]
    summary['u_all_with_class_rate'] = float(np.mean(u_ok))
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
