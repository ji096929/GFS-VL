"""Cut 2 阶段三：用训练好的 Linear Probe 在 312 val 上流式累计混淆矩阵。

用法: python cut2_eval.py <S1|S2> [shard] [shards]
输出 /data01/jh/data/cut2_feat/{tag}/cm_s{i}.npz（6 个 57x57 CM）

口径：与测试器一致（fragment 推理 + 概率按 index 累加）；本脚本额外除以
覆盖次数 cover，得到真正的平均概率（已有 val_logits 没除，概率值虚高）。
"""
import os
import sys
import time

import numpy as np
import torch

from cut2_common import (LAYERS, NC, NO, Probe, build_model_and_cfg, make_ds)
from pointcept.datasets import collate_fn  # noqa: E402

FEAT = '/data01/jh/data/cut2_feat'
MULTI = ['dec0', 'dec1', 'dec2', 'dec3']
NAMES = LAYERS + ['multi']
NN = NC - NO


def load_protos(tag):
    """1-shot 支持原型（L2 归一化均值）。"""
    import glob
    files = sorted(glob.glob(os.path.join(FEAT, tag, 'support_s*.npz')))
    assert files, '没有 support 特征，先跑 cut2_extract.py %s support' % tag
    acc = {L: [] for L in LAYERS}
    gt = []
    for f in files:
        z = np.load(f)
        for L in LAYERS:
            acc[L].append(z['f_' + L])
        gt.append(z['gt'])
    gt = np.concatenate(gt, 0).astype(np.int64)
    feats = {L: np.concatenate(acc[L], 0).astype(np.float32) for L in LAYERS}
    P = {}
    for nm in NAMES:
        X = np.concatenate([feats[L] for L in MULTI], 1) if nm == 'multi' \
            else feats[nm]
        M = np.zeros((NN, X.shape[1]), np.float32)
        for i, c in enumerate(range(NO, NC)):
            w = gt == c
            if w.sum() == 0:
                continue
            v = X[w].mean(0)
            M[i] = v / (np.linalg.norm(v) + 1e-9)
        P[nm] = M
    print('%s support %d 点，原型已建' % (tag, len(gt)), flush=True)
    return P


def metrics(cm):
    d = np.diag(cm).astype(np.float64)
    r = cm.sum(1).astype(np.float64)
    c = cm.sum(0).astype(np.float64)
    iou = d / np.maximum(r + c - d, 1)
    b = float(iou[:NO].mean())
    n = float(iou[NO:].mean())
    return b, n, 2 * b * n / max(b + n, 1e-9), iou


def main():
    tag = sys.argv[1]
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out = os.path.join(FEAT, tag, 'cm_s%d.npz' % shard)
    if os.path.exists(out):
        print('已存在，跳过', out, flush=True)
        return

    pz = np.load(os.path.join(FEAT, tag, 'probes.npz'))
    model, cfg = build_model_and_cfg(tag)
    pb = Probe()
    pb.attach(model.backbone)
    ds = make_ds(cfg, 'val')
    ds.data_list = ds.data_list[shard::shards]
    print('%s val: 本片 %d 场景' % (tag, len(ds.data_list)), flush=True)

    cms = {n: np.zeros((NC, NC), np.int64) for n in NAMES}
    cms_os = {n: np.zeros((NN, NN), np.int64) for n in NAMES}
    # 探针参数只搬到 GPU 一次
    PAR = {}
    for nm in NAMES:
        PAR[nm] = (torch.tensor(pz['mu_' + nm], device='cuda'),
                   torch.tensor(pz['sd_' + nm], device='cuda'),
                   torch.tensor(pz['W_' + nm], device='cuda'),
                   torch.tensor(pz['b_' + nm], device='cuda'))
    PROTO = load_protos(tag)
    PROTO = {nm: torch.tensor(v, device='cuda') for nm, v in PROTO.items()}
    t0 = time.time()
    for i in range(len(ds.data_list)):
        data_dict = ds[i]
        frags = data_dict.pop('fragment_list')
        seg = data_dict.pop('segment')
        gt = np.asarray(seg).astype(np.int64)
        n = len(gt)
        probs = {nm: torch.zeros((n, NC), device='cuda') for nm in NAMES}
        cover = torch.zeros(n, device='cuda')
        for frag in frags:
            inp = collate_fn([frag])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp['index']
            g_frag = gt[ip.detach().cpu().numpy()]
            m = torch.from_numpy(g_frag >= NO).cuda()
            with torch.no_grad():
                model(inp)
                fs = pb.layers()
                for nm in NAMES:
                    if nm == 'multi':
                        X = torch.cat([fs[L] for L in MULTI], 1)
                    else:
                        X = fs[nm]
                    mu, sd, W, b = PAR[nm]
                    lg = ((X - mu) / sd) @ W + b
                    probs[nm][ip] += torch.softmax(lg.float(), -1)
                    # 1-shot 原型（同一份特征，不额外前向）
                    if bool(m.any()):
                        Xn = X[m]
                        Xn = Xn / (Xn.norm(dim=1, keepdim=True) + 1e-9)
                        po = (Xn @ PROTO[nm].T).argmax(1)
                        gg = torch.from_numpy(g_frag[g_frag >= NO] - NO).cuda()
                        cms_os[nm] += torch.bincount(
                            gg * NN + po, minlength=NN * NN).cpu().numpy() \
                            .reshape(NN, NN)
            cover[ip] += 1
        cv = cover.clamp(min=1)
        ok = torch.tensor(gt >= 0, device='cuda')
        gt_t = torch.tensor(np.clip(gt, 0, NC - 1), device='cuda')
        for nm in NAMES:
            pred = (probs[nm] / cv[:, None]).argmax(1)
            idx = (gt_t[ok] * NC + pred[ok])
            cms[nm] += torch.bincount(idx, minlength=NC * NC).cpu().numpy() \
                .reshape(NC, NC)
        if (i + 1) % 20 == 0:
            print('  %d/%d  %.0fs' % (i + 1, len(ds.data_list),
                                      time.time() - t0), flush=True)

    np.savez(out, **cms, **{'os_' + k: v for k, v in cms_os.items()})
    b0, n0, h0, _ = metrics(cms['dec0'])
    print('-> %s | dec0 Base %.4f Novel %.4f HM %.4f' % (out, b0, n0, h0),
          flush=True)
    for nm in NAMES:
        cmx = cms_os[nm]
        d = np.diag(cmx).astype(np.float64)
        r = cmx.sum(1).astype(np.float64)
        c = cmx.sum(0).astype(np.float64)
        iou = d / np.maximum(r + c - d, 1)
        print('   1shot %-5s acc %.4f mIoU %.4f'
              % (nm, d.sum() / max(cmx.sum(), 1), iou.mean()), flush=True)


if __name__ == '__main__':
    main()
