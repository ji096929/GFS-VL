"""补齐 intra-query 方向需要的字段。

原缓存只有 downsampled 的 q_h/q_margin/q_coord，缺少：
  q_seg_full / q_rgb_full   voxel 后原始点的超点 id 与颜色
  q_seg / q_rgb             downsampled 点对应的超点 id 与颜色
  s_coord_full / s_seg_full / s_rgb_full   支持集的空间信息（边界腐蚀用）

全部从原始 .npy 重新预处理得到，不跑 COSeg。
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import diag_bc as D  # noqa: E402

CACHE = '/data01/jh/data/ocsb_cache'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'


def prep(name, cfg):
    d = np.load(os.path.join(DATA, name + '.npy'))
    coord, feat, label, seg = D.data_prepare_seg(
        d[:, 0:3], d[:, 3:6], d[:, 6], d[:, 7],
        cfg.voxel_size, cfg.voxel_max)
    return coord.numpy(), (feat.numpy() * 255.0), label.numpy(), seg.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--end', type=int, default=560)
    ap.add_argument('--out', default=os.path.join(CACHE, 'intra'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = D.load_cfg_from_cfg_file(D.CFG_PATH)
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))

    done = 0
    for i in range(args.start, min(args.end, len(metas))):
        dst = os.path.join(args.out, 'intra_%03d.npz' % i)
        if os.path.exists(dst):
            continue
        m = metas[i]
        base = np.load(os.path.join(CACHE, 'val_%03d.npz' % m['idx']))
        q_xyz = base['q_coord'].astype(np.float32)          # downsampled
        q_xyz_full = base['q_coord_full'].astype(np.float32)

        q_coord, q_rgb, q_lab, q_seg = prep(m['query'], cfg)
        s_coord, s_rgb, s_lab, s_seg = prep(m['support'], cfg)

        # 校验：缓存的 q_coord_full 与重新预处理一致
        n = min(len(q_xyz_full), len(q_coord))
        if not np.allclose(q_xyz_full[:n], q_coord[:n], atol=1e-4):
            print('  [warn] ep %d coord mismatch %d vs %d'
                  % (i, len(q_xyz_full), len(q_coord)), flush=True)

        idx = cKDTree(q_coord.astype(np.float64)).query(
            q_xyz.astype(np.float64))[1]
        np.savez_compressed(
            dst,
            q_seg_full=q_seg.astype(np.int32),
            q_rgb_full=q_rgb.astype(np.uint8),
            q_seg=q_seg[idx].astype(np.int32),
            q_rgb=q_rgb[idx].astype(np.uint8),
            s_coord_full=s_coord.astype(np.float32),
            s_seg_full=s_seg.astype(np.int32),
            s_rgb_full=s_rgb.astype(np.uint8),
        )
        done += 1
        if done % 20 == 0:
            print('  %d / %d' % (i + 1, args.end), flush=True)
    print('done', done)


if __name__ == '__main__':
    main()
