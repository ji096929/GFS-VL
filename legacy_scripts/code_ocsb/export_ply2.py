"""导出改进版 PLY：
  1. 每个 U block 单独一个文件（原始 RGB + 区域标注）
  2. counter_007 / cabinet_009 额外导出完整房间（gamma 提亮 + 切块高亮）

输出：/data01/jh/data/ocsb_cache/ply2/
"""
import json
import os

import numpy as np
from plyfile import PlyData

CACHE = '/data01/jh/data/ocsb_cache'
RAW = '/data01/jh/data/scannet_raw/scans'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'
OUT = os.path.join(CACHE, 'ply2')
TASKS = (7, 9, 11)
FULL_ROOM_TASKS = (7, 9)


def read_ply_xyzrgb(path):
    ply = PlyData.read(path)
    v = ply['vertex']
    xyz = np.stack([np.asarray(v['x']), np.asarray(v['y']),
                    np.asarray(v['z'])], 1).astype(np.float64)
    rgb = np.stack([np.asarray(v['red']), np.asarray(v['green']),
                    np.asarray(v['blue'])], 1).astype(np.float64)
    return xyz, rgb


def write_ply(path, xyz, rgb):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write('element vertex %d\n' % len(xyz))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\n'
                'property uchar blue\n')
        f.write('end_header\n')
        for i in range(len(xyz)):
            f.write('%.4f %.4f %.4f %d %d %d\n'
                    % (xyz[i, 0], xyz[i, 1], xyz[i, 2],
                       int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])))


def brighten(rgb, g=0.5):
    return np.clip((rgb / 255.0) ** g * 255.0, 0, 255)


def main():
    os.makedirs(OUT, exist_ok=True)
    metas = json.load(open(os.path.join(CACHE, 'val_meta_all.json')))
    index = []

    for idx in TASKS:
        m = next(x for x in metas if x['idx'] == idx)
        d = np.load(os.path.join(CACHE, 'val_%03d.npz' % idx))

        xyz_all, rgb_all, lab_all, seg_all = [], [], [], []
        block_off = []
        off = 0
        for i, nm in enumerate(m['u']):
            r = np.load(os.path.join(DATA, nm + '.npy'))
            n = len(r)
            xyz_all.append(r[:, 0:3].astype(np.float64))
            rgb_all.append(brighten(r[:, 3:6].astype(np.float64)))
            lab_all.append(r[:, 6].astype(np.int64))
            seg_all.append(r[:, 7].astype(np.int64) + i * 10 ** 7)
            block_off.append((off, off + n))
            off += n
        xyz_all = np.concatenate(xyz_all)
        rgb_all = np.concatenate(rgb_all)
        lab_all = np.concatenate(lab_all)
        seg_all = np.concatenate(seg_all)
        uniq, inv = np.unique(seg_all, return_inverse=True)
        n_u = len(uniq)

        u_seg = d['u_seg'].astype(np.int64)
        u_p0 = d['u_p0'].astype(np.float64)
        u_y = d['u_y'].astype(np.float64)
        cnt = np.maximum(
            np.bincount(u_seg, minlength=n_u).astype(np.float64), 1)
        q = np.bincount(u_seg, weights=u_p0, minlength=n_u) / cnt
        fg = np.bincount(u_seg, weights=u_y, minlength=n_u) / cnt
        B = q >= 0.6
        B_pos = B & (fg > 0.5)
        B_neg = B & (fg <= 0.5)
        C_pos = (q < 0.5) & (fg > 0.5)

        for bi, nm in enumerate(m['u']):
            a, b = block_off[bi]
            xyz_b = xyz_all[a:b]
            rgb_b = rgb_all[a:b]
            inv_b = inv[a:b]
            tag = np.zeros(len(xyz_b), dtype=np.int8)
            tag[np.isin(inv_b, np.where(B_pos)[0])] = 1
            tag[np.isin(inv_b, np.where(B_neg)[0])] = 2
            tag[np.isin(inv_b, np.where(C_pos)[0])] = 3
            tag_rgb = rgb_b * 0.25
            tag_rgb[tag == 1] = [0, 220, 0]
            tag_rgb[tag == 2] = [230, 0, 0]
            tag_rgb[tag == 3] = [255, 210, 0]
            base = 'task%02d_b%d_%s' % (idx, bi, nm)
            write_ply(os.path.join(OUT, base + '_rgb.ply'), xyz_b, rgb_b)
            write_ply(os.path.join(OUT, base + '_tag.ply'), xyz_b, tag_rgb)
            index.append({'task': idx, 'block': nm, 'file': base,
                          'n_pts': int(len(xyz_b)),
                          'n_B_pos': int((tag == 1).sum()),
                          'n_B_neg': int((tag == 2).sum()),
                          'n_C_pos': int((tag == 3).sum())})
            print('  %s  pts %d  B+%d B-%d C+%d'
                  % (base, len(xyz_b), (tag == 1).sum(),
                     (tag == 2).sum(), (tag == 3).sum()), flush=True)

    for idx in TASKS:
        m = next(x for x in metas if x['idx'] == idx)
        scenes = sorted(set(nm.split('_block_')[0] for nm in m['u']))
        if idx in FULL_ROOM_TASKS:
            scenes = sorted(set(scenes) | {m['query'].split('_block_')[0]})
        for sc in scenes:
            p = os.path.join(RAW, sc, sc + '_vh_clean_2.ply')
            if not os.path.exists(p):
                print('  missing room', sc, flush=True)
                continue
            xyz, rgb = read_ply_xyzrgb(p)
            rgb = brighten(rgb)
            inside = np.zeros(len(xyz), dtype=bool)
            rel = [x for x in m['u'] if x.startswith(sc)]
            if m['query'].startswith(sc):
                rel = rel + [m['query']]
            for nm in rel:
                b = np.load(os.path.join(DATA, nm + '.npy'))
                mn, mx = b[:, 0:3].min(0), b[:, 0:3].max(0)
                inside |= ((xyz[:, 0] >= mn[0] - 0.02) &
                           (xyz[:, 0] <= mx[0] + 0.02) &
                           (xyz[:, 1] >= mn[1] - 0.02) &
                           (xyz[:, 1] <= mx[1] + 0.02) &
                           (xyz[:, 2] >= mn[2] - 0.02) &
                           (xyz[:, 2] <= mx[2] + 0.02))
            base = 'task%02d_room_%s' % (idx, sc)
            write_ply(os.path.join(OUT, base + '_rgb.ply'), xyz, rgb)
            box = rgb * 0.45
            box[inside] = [255, 230, 0]
            write_ply(os.path.join(OUT, base + '_box.ply'), xyz, box)
            index.append({'task': idx, 'scene': sc, 'file': base,
                          'n_room': int(len(xyz)),
                          'n_inside': int(inside.sum())})
            print('  room %s  pts %d  inside %d'
                  % (sc, len(xyz), int(inside.sum())), flush=True)

    json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), indent=1)
    print('->', OUT)


if __name__ == '__main__':
    main()
