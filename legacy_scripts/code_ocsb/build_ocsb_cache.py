"""提取 3D-OCSB 的冻结特征缓存。

因为 Stage I 的主干 F0 和初始预测器 f0 完全冻结，每个 episode 只需要前向
一次并把下列量存盘，之后所有公式（1)-(19）、五项自检、五步消融都在缓存上
秒级重跑：

  s_h [Ns,d]   s_y [Ns]   s_seg [Ns]       支持点投影前的特征/标签/超点
  u_h [Nu,d]   u_seg [Nu] u_y [Nu] u_p0 [Nu]   U 点特征/超点/真值/初始前景概率
  q_h [Nq,d]   q_y [Nq]   q_margin [Nq]     查询点特征/真值/COSeg 原生 margin
  q_y_full / q_margin_full / q_coord / q_coord_full   原始点级评测用

两个集合：
  val_100    manifest_fold0.json 前 100 个 novel episode（跨场景 U）
  train_200  Fold-0 的 10 个 base 类，每类 20 个 episode，跨场景
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_bc as D  # noqa: E402

# cvfold=0 的 base 类（scannet_classnames.txt 的 id）
BASE_FOLD0_NAMES = ['wall', 'table', 'sofa', 'sink', 'toilet', 'window',
                    'shower curtain', 'refridgerator', 'picture',
                    'otherfurniture']
BASE_TO_SLOT = {1: 0, 4: 1, 8: 2, 9: 3, 11: 4, 15: 5, 16: 6, 17: 7, 18: 8,
                20: 9}


def load_block(info, cfg):
    d = np.load(info['path'])
    return D.data_prepare_seg(d[:, 0:3], d[:, 3:6], d[:, 6], d[:, 7],
                              cfg.voxel_size, cfg.voxel_max)


def nn_index(src_xyz, dst_xyz):
    """dst 每个点在 src 中的最近邻索引。"""
    _, idx = cKDTree(src_xyz).query(dst_xyz)
    return idx


def remap_ids(seg):
    _, inv = np.unique(np.asarray(seg), return_inverse=True)
    return inv.astype(np.int32)


@torch.no_grad()
def extract_episode(model, cfg, s_info, q_info, u_infos, cls, dev):
    s_coord, s_feat, s_label, s_seg = load_block(s_info, cfg)
    q_coord, q_feat, q_label, q_seg = load_block(q_info, cfg)
    us = [load_block(i, cfg) for i in u_infos]

    s_bin = (s_label == cls).long()
    q_map = (q_label == cls).long()

    u_coord = torch.cat([x[0] for x in us], 0)
    u_feat = torch.cat([x[1] for x in us], 0)
    u_label = torch.cat([x[2] for x in us], 0)
    # 多 block 的超点 id 加偏移保证全局唯一
    u_seg_raw = torch.cat([x[3] + i * 10 ** 7 for i, x in enumerate(us)], 0)
    _, u_seg = torch.unique(u_seg_raw, return_inverse=True)

    support_x = torch.cat([s_coord, s_feat], 1).to(dev)
    query_x = torch.cat([q_coord, q_feat], 1).to(dev)
    u_x = torch.cat([u_coord, u_feat], 1).to(dev)
    s_off = torch.tensor([s_coord.shape[0]], dtype=torch.int32).to(dev)
    q_off = torch.tensor([q_coord.shape[0]], dtype=torch.int32).to(dev)
    u_off = torch.tensor([u_coord.shape[0]], dtype=torch.int32).to(dev)

    # U 作为 query：初始前景概率（原始点级）
    u_pred, _ = model(s_off, support_x, s_bin.to(dev), u_off, u_x,
                      torch.zeros_like(u_label).to(dev), 100, None, None,
                      np.array([cls]))
    p_u = F.softmax(u_pred[0], dim=0)[1]

    # Q 的原生 logits（修正点 B 的基准项）
    q_pred, _ = model(s_off, support_x, s_bin.to(dev), q_off, query_x,
                      q_map.to(dev), 100, None, None, np.array([cls]))
    q_margin_full = q_pred[0][1] - q_pred[0][0]

    # 冻结主干特征
    u_fd, u_cd, _, _, _, _ = model.getFeatures(
        u_x, u_off, torch.zeros_like(u_label).to(dev), None)
    s_fd, s_cd, _, _, _, _ = model.getFeatures(
        support_x, s_off, s_bin.to(dev), None)
    q_fd, q_cd, _, _, _, _ = model.getFeatures(
        query_x, q_off, q_map.to(dev), None)

    u_nn = nn_index(u_coord.numpy(), u_cd.cpu().numpy())
    s_nn = nn_index(s_coord.numpy(), s_cd.cpu().numpy())
    q_nn = nn_index(q_coord.numpy(), q_cd.cpu().numpy())

    return {
        's_h': s_fd.cpu().numpy().astype(np.float16),
        's_y': (s_label.numpy()[s_nn] == cls).astype(np.int8),
        's_seg': remap_ids(s_seg.numpy()[s_nn]),
        'u_h': u_fd.cpu().numpy().astype(np.float16),
        'u_seg': u_seg.numpy()[u_nn].astype(np.int32),
        'u_y': (u_label.numpy()[u_nn] == cls).astype(np.int8),
        'u_p0': p_u.cpu().numpy()[u_nn].astype(np.float16),
        'q_h': q_fd.cpu().numpy().astype(np.float16),
        'q_y': (q_label.numpy()[q_nn] == cls).astype(np.int8),
        'q_margin': q_margin_full.cpu().numpy()[q_nn].astype(np.float16),
        'q_y_full': (q_label.numpy() == cls).astype(np.int8),
        'q_margin_full': q_margin_full.cpu().numpy().astype(np.float16),
        'q_coord': q_cd.cpu().numpy().astype(np.float32),
        'q_coord_full': q_coord.numpy().astype(np.float32),
    }


def build_train_specs(class2blocks, base_ids, n_per_class, rng):
    """跨场景采样 200 个 base 类训练任务。"""
    specs = []
    for c in base_ids:
        cand = class2blocks.get(c, [])
        scenes = sorted({b['scene'] for b in cand})
        if len(scenes) < 4:
            print('  [skip] class %d only %d scenes' % (c, len(scenes)))
            continue
        made, tries = 0, 0
        while made < n_per_class and tries < n_per_class * 30:
            tries += 1
            perm = rng.permutation(len(cand))
            s = cand[perm[0]]
            q = next((cand[j] for j in perm[1:] if cand[j]['scene'] != s['scene']),
                     None)
            if q is None:
                continue
            used = {s['scene'], q['scene']}
            upool = [b for b in cand if b['scene'] not in used]
            if len(upool) < 3:
                continue
            pick = rng.choice(len(upool), 3, replace=False)
            specs.append({
                'class': int(c),
                'support': s['name'],
                'query': q['name'],
                'u_cross': [upool[k]['name'] for k in pick],
            })
            made += 1
        print('  class %d -> %d episodes' % (c, made))
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/data01/jh/data/ScanNet/blocks_bs1_s1/data')
    ap.add_argument('--manifest', default='/data01/jh/data/manifest_fold0.json')
    ap.add_argument('--out', default='/data01/jh/data/ocsb_cache')
    ap.add_argument('--val-eps', type=int, default=100)
    ap.add_argument('--val-start', type=int, default=0,
                    help='从 manifest 的第几个 episode 开始（分段并行用）')
    ap.add_argument('--val-end', type=int, default=None,
                    help='到第几个结束；默认等于 --val-eps')
    ap.add_argument('--train-per-class', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip-train', action='store_true')
    ap.add_argument('--skip-val', action='store_true')
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device('cuda:0')

    names = D.load_class_names()
    name2id = {n: i for i, n in enumerate(names)}
    novel_ids = [name2id[n] for n in D.NOVEL_FOLD0]
    base_ids = [name2id[n] for n in BASE_FOLD0_NAMES]
    print('novel ids:', novel_ids)
    print('base ids:  ', base_ids)
    assert set(BASE_TO_SLOT) == set(base_ids), (sorted(BASE_TO_SLOT), base_ids)

    print('scanning blocks...', flush=True)
    blocks = D.scan_blocks(args.data)
    info_by_name = {b['name']: b for b in blocks}
    class2blocks = {c: [b for b in blocks if c in b['classes']]
                    for c in novel_ids + base_ids}
    print('total blocks:', len(blocks), flush=True)
    for c in base_ids:
        print('  base %2d %-16s %4d blocks' % (c, names[c], len(class2blocks[c])))

    model, cfg = D.build_model(dev)
    print('model loaded', flush=True)

    os.makedirs(args.out, exist_ok=True)
    meta = []

    if not args.skip_val:
        all_eps = json.load(open(args.manifest))['episodes']
        end = args.val_end if args.val_end is not None else args.val_eps
        spec = all_eps[args.val_start:end]
        print('val episodes: [%d, %d) -> %d' % (args.val_start, end, len(spec)),
              flush=True)
        for i, sp in enumerate(spec):
            ep_idx = args.val_start + i
            np.random.seed(10000 + ep_idx)
            torch.manual_seed(10000 + ep_idx)
            c = sp['class']
            s_info = info_by_name[sp['support']]
            q_info = info_by_name[sp['query']]
            u_infos = [info_by_name[n] for n in sp.get('u_cross', [])
                       if n in info_by_name]
            if len(u_infos) < 1:
                continue
            dst = os.path.join(args.out, 'val_%03d.npz' % ep_idx)
            if os.path.exists(dst):
                meta.append({'split': 'val', 'idx': ep_idx, 'class': int(c),
                             'support': s_info['name'], 'query': q_info['name'],
                             'u': [b['name'] for b in u_infos]})
                continue
            try:
                arr = extract_episode(model, cfg, s_info, q_info, u_infos, c, dev)
            except Exception as e:
                print('val ep', ep_idx, 'failed:', repr(e)[:100], flush=True)
                continue
            np.savez_compressed(dst, **arr)
            meta.append({'split': 'val', 'idx': ep_idx, 'class': int(c),
                         'support': s_info['name'], 'query': q_info['name'],
                         'u': [b['name'] for b in u_infos]})
            if (i + 1) % 10 == 0:
                print('  val', ep_idx + 1, '/', end, flush=True)
        json.dump(meta, open(os.path.join(
            args.out, 'val_meta_%03d.json' % args.val_start), 'w'), indent=1)

    if not args.skip_train:
        rng = np.random.RandomState(args.seed + 1)
        specs = build_train_specs(class2blocks, base_ids,
                                  args.train_per_class, rng)
        print('train episodes:', len(specs), flush=True)
        base_meta = []
        for i, sp in enumerate(specs):
            dst = os.path.join(args.out, 'train_%03d.npz' % i)
            s_info = info_by_name[sp['support']]
            q_info = info_by_name[sp['query']]
            u_infos = [info_by_name[n] for n in sp['u_cross']]
            if os.path.exists(dst):      # 断点续跑
                base_meta.append({'split': 'train', 'idx': i,
                                  'class': int(sp['class']),
                                  'support': s_info['name'],
                                  'query': q_info['name'],
                                  'u': [b['name'] for b in u_infos]})
                continue
            np.random.seed(20000 + i)
            torch.manual_seed(20000 + i)
            c = sp['class']
            try:
                arr = extract_episode(model, cfg, s_info, q_info, u_infos, c, dev)
            except Exception as e:
                print('train ep', i, 'failed:', repr(e)[:100], flush=True)
                continue
            np.savez_compressed(dst, **arr)
            base_meta.append({'split': 'train', 'idx': i, 'class': int(c),
                              'support': s_info['name'], 'query': q_info['name'],
                              'u': [b['name'] for b in u_infos]})
            if (i + 1) % 10 == 0:
                print('  train', i + 1, '/', len(specs), flush=True)
        json.dump(base_meta, open(os.path.join(args.out, 'train_meta.json'), 'w'),
                  indent=1)

    print('done ->', args.out)


if __name__ == '__main__':
    main()
