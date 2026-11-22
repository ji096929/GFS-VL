"""Phase 1: build an orthogonal visual feature f_orth with a frozen 2D encoder.

Server has no CLIP/DINOv2 and no HF access, but download.pytorch.org works, so
we use torchvision's ImageNet ResNet-50 -- not CLIP-level semantics, but a real
visual encoder that never saw COSeg's 3D training data.

Pipeline per block:
  XYZ+RGB -> 5 axis-aligned orthographic views (224x224, z-buffer) ->
  frozen ResNet50 layer4 (2048 x 7 x 7) -> for every 3D point look up the cell
  it projects to in each view -> average across views -> per-superpoint mean ->
  L2 normalise.
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import clip_vit


CLIP_PATH = '/data01/ysz/UniMoS/.runtime/cache/clip/ViT-B-16.pt'


def load_clip():
    m = clip_vit.load_from_torchscript(CLIP_PATH, verbose=True)
    return m.cuda()


VIEWS = [  # (u axis, v axis, depth axis, u sign, v sign)
    (1, 2, 0, -1.0, 1.0),   # +X
    (1, 2, 0, 1.0, 1.0),    # -X
    (0, 2, 1, 1.0, 1.0),    # +Y
    (0, 2, 1, -1.0, 1.0),   # -Y
    (0, 1, 2, 1.0, 1.0),    # +Z (top)
]
S = 224
GRID = 14
CELL = S // GRID  # 16 for patch-16 CLIP


def render(xyz, rgb):
    """Return [V,3,S,S] images and per-view (u,v) pixel coords of every point."""
    out = np.zeros((len(VIEWS), 3, S, S), np.float32)
    pixs = []
    c = xyz.mean(0)
    x = xyz - c
    for vi, (ua, va, da, us, vs) in enumerate(VIEWS):
        u = x[:, ua] * us
        v = x[:, va] * vs
        d = x[:, da]
        ur = u.max() - u.min()
        vr = v.max() - v.min()
        un = ((u - u.min()) / (ur + 1e-6) * (S - 1)).astype(np.int32)
        vn = ((v - v.min()) / (vr + 1e-6) * (S - 1)).astype(np.int32)
        pix = vn * S + un
        zbuf = np.full(S * S, -np.inf, np.float32)
        np.maximum.at(zbuf, pix, d)
        keep = d >= zbuf[pix] - 1e-6
        img = np.zeros((S * S, 3), np.float32)
        img[pix[keep]] = rgb[keep]
        out[vi] = img.reshape(S, S, 3).transpose(2, 0, 1)
        pixs.append(pix)
    return out, pixs


@torch.no_grad()
def extract_block(model, path, dev='cuda:0'):
    d = np.load(path, mmap_mode='r')
    xyz = np.asarray(d[:, 0:3], np.float32)
    rgb = np.asarray(d[:, 3:6], np.float32) / 255.0
    seg = np.asarray(d[:, 7]).astype(np.int64) if d.shape[1] > 7 else None
    imgs, pixs = render(xyz, rgb)
    t = torch.from_numpy(imgs).to(dev)
    # imagenet normalisation
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    t = (t - mean) / std
    feat = model(t)                       # [V, 196, 768] patch tokens
    feat = feat.permute(0, 2, 1)          # [V, 768, 196]
    V, C, _ = feat.shape

    # per point: average the cell it projects to across views
    N = xyz.shape[0]
    acc = np.zeros((N, C), np.float32)
    cnt = np.zeros(N, np.float32)
    feat_cpu = feat.permute(0, 2, 1).cpu().numpy()  # [V,49,C]
    for vi in range(V):
        pix = pixs[vi]
        cell = ((pix % S) // CELL) + GRID * (pix // S // CELL)
        cell = np.clip(cell, 0, GRID * GRID - 1)
        acc += feat_cpu[vi][cell]
        cnt += 1
    acc /= np.maximum(cnt, 1)[:, None]

    if seg is None:
        return acc
    uniq, inv = np.unique(seg, return_inverse=True)
    reg = np.zeros((len(uniq), C), np.float32)
    np.add.at(reg, inv, acc)
    n = np.bincount(inv, minlength=len(uniq)).astype(np.float32)
    reg /= np.maximum(n, 1)[:, None]
    return reg, uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/data01/jh/data/ScanNet/blocks_bs1_s1/data')
    ap.add_argument('--out', default='/data01/jh/data/orth_cache')
    ap.add_argument('--manifest', default='/data01/jh/data/manifest_fold0.json')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--arch', default='resnet18')
    ap.add_argument('--eps', type=int, default=0,
                    help='only cover the first N manifest episodes')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import json
    eps = json.load(open(args.manifest))['episodes']
    if args.eps:
        eps = eps[:args.eps]
    names = set()
    for e in eps:
        names.add(e['support'])
        names.add(e['query'])
        names.update(e.get('u_cross', []))
        names.update(e.get('u_same', []))
    names = sorted(names)
    if args.limit:
        names = names[:args.limit]
    print('blocks to process:', len(names), flush=True)

    model = load_clip()
    done = 0
    for nm in names:
        dst = os.path.join(args.out, nm + '.npz')
        if os.path.exists(dst):
            continue
        src = os.path.join(args.data, nm + '.npy')
        if not os.path.exists(src):
            continue
        try:
            reg, uniq = extract_block(model, src)
            np.savez_compressed(dst, feat=reg.astype(np.float16),
                                seg=uniq.astype(np.int64))
            done += 1
        except Exception as e:
            print('fail', nm, repr(e)[:80], flush=True)
        if done % 20 == 0 and done:
            print('  done', done, flush=True)
    print('finished, new blocks:', done)


if __name__ == '__main__':
    main()
