"""细节 4 第二步：用 Sonata 纯几何特征池化同一批区域（sonata 环境运行）。

输入  /data01/jh/data/region_index/regions_index_{split}_s{N}.pt
输出  /data01/jh/data/sonata_region/sonata_regions_{split}_s{N}.pt
      { scene: {feats: float16[R,1232], off: int32[R+1], pred: int16[R]} }

区域边界完全复用 VLM 侧导出的点索引，保证两种特征在同一容器上比较。
"""
import argparse
import os
import sys

sys.path.insert(0, '/data01/jh/code/sonata')
sys.path.insert(0, '/data01/jh/code')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import sonata  # noqa: E402

DATA = '/data01/jh/data/scannet200_gfs'
IDX = '/data01/jh/data/region_index'
OUT = '/data01/jh/data/sonata_region'
CKPT = '/data01/jh/code/sonata/ckpt/sonata.pth'


def estimate_normals(coord):
    """数据已带 normal.npy 时不该走到这里；保留兜底（法向朝向杯内）。"""
    import numpy as np
    n = np.zeros_like(coord, dtype=np.float32)
    if len(coord) < 3:
        return n
    from scipy.spatial import cKDTree
    tree = cKDTree(coord[:, :3])
    _, nb = tree.query(coord[:, :3], k=9)
    p = coord[:, :3][nb]
    c = p - p.mean(1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', c, c)
    w, v = np.linalg.eigh(cov)
    n = v[:, :, 0]
    return n.astype(np.float32)


def feat_sonata(model, coord, color, normal):
    """PTv3 编码器逐层解码，返回原始点数的 1232 维特征。

    注意 sonata.transform 用 numpy 语义（coord.min(axis=0) 期望返回三元组），
    所以这里必须传 numpy 数组，不能预先转 torch.Tensor。
    """
    pt = dict(coord=np.asarray(coord, dtype=np.float32),
              color=np.asarray(color, dtype=np.float32),
              normal=np.asarray(normal, dtype=np.float32))
    pt = sonata.transform.default()(pt)
    with torch.inference_mode():
        for k in list(pt.keys()):
            if isinstance(pt[k], torch.Tensor):
                pt[k] = pt[k].cuda(non_blocking=True)
        base = model(pt)
        chain, cur = [], base
        while 'pooling_parent' in cur.keys():
            chain.append((cur['pooling_parent'], cur['pooling_inverse']))
            cur = cur['pooling_parent']
        top = cur
        base_inv = top['inverse']
        feat = base.feat
        for par, iv in chain:
            feat = torch.cat([par.feat, feat[iv]], dim=-1)
        return feat[base_inv].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--split', default='train')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    p = os.path.join(IDX, 'regions_index_%s_s%d.pt' % (args.split, args.shard))
    d = torch.load(p, map_location='cpu')
    print('scenes', len(d), flush=True)

    cfg = dict(enc_patch_size=[1024 for _ in range(5)], enable_flash=False)
    model = sonata.load(CKPT, custom_config=cfg).cuda().eval()
    print('[sonata] %.2fM params'
          % (sum(q.numel() for q in model.parameters()) / 1e6), flush=True)

    out = {}
    for i, (sc, r) in enumerate(sorted(d.items())):
        base = os.path.join(DATA, args.split, sc)
        coord = np.load(os.path.join(base, 'coord.npy')).astype(np.float32)
        color = np.load(os.path.join(base, 'color.npy')).astype(np.float32)
        np_path = os.path.join(base, 'normal.npy')
        if os.path.exists(np_path):
            normal = np.load(np_path).astype(np.float32)
        else:
            normal = estimate_normals(coord)
        f = feat_sonata(model, coord, color, normal)
        if len(f) != len(r['pred']):
            print('  !! %s 点数不一致 %d vs %d' % (sc, len(f), len(r['pred'])))
            continue
        off, idx = r['off'], r['idx']
        rows = []
        for k in range(len(off) - 1):
            seg = idx[off[k]:off[k + 1]]
            rows.append(f[seg].mean(0))
        feats = np.stack(rows).astype(np.float16) if rows else \
            np.zeros((0, f.shape[1]), np.float16)
        out[sc] = dict(feats=feats,
                       off=np.array(off, dtype=np.int32),
                       pred=r['pred'][off[:-1]] if len(off) > 1
                       else np.empty(0, np.int16))
        if (i + 1) % 20 == 0:
            print('  %d / %d' % (i + 1, len(d)), flush=True)

    q = os.path.join(OUT, 'sonata_regions_%s_s%d.pt' % (args.split, args.shard))
    torch.save(out, q)
    print('->', q)


if __name__ == '__main__':
    main()
