"""Stage II: light fine-tuning of the bridge-path HCA only.

Design constraints that motivate every choice here:

* Epoch-0 must equal the zero-shot 2C result. We therefore deep-copy the
  pretrained COSeg model into `model_b` and unfreeze ONLY the HCA stage
  (lin1 / agglayers / class_reduce / kpconv / cls). Everything else, including
  the shared encoder and the support-path HCA, stays frozen. Since model_b
  starts as an exact copy, its output at step 0 is identical to the frozen
  model's bridge output.
* Training episodes are drawn from BASE classes only, which the encoder has
  seen. To stop the HCA from learning "bridges are always clean", we
     1) treat the target class as if unseen (no base-prototype guidance),
     2) keep S/U/Q in different scenes,
     3) inject hard negatives into the selected bridge set.
* Loss is the plain CE on the fused query logits, which directly penalises the
  false positives that Task 6 identified as the main failure mode.

Usage:
  python train_stage2.py --epochs 8 --steps-per-epoch 200 --out /data01/jh/data/s2
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/data01/jh/code')
# reuse the diagnostic harness so the bridge construction stays identical
import diag_bc as H  # noqa: E402

HCA_PREFIX = ('lin1', 'agglayers', 'class_reduce', 'kpconv', 'cls')


_BLOCK_CACHE = {}


def load_block(info, cfg):
    """Cached load. Voxelisation is pure CPU numpy and dominated step time:
    without this cache each training step re-read and re-voxelised 4 blocks."""
    key = (info['name'], cfg.voxel_size, cfg.voxel_max)
    hit = _BLOCK_CACHE.get(key)
    if hit is None:
        d = np.load(info['path'])
        hit = H.data_prepare_seg(d[:, 0:3], d[:, 3:6], d[:, 6], d[:, 7],
                                 cfg.voxel_size, cfg.voxel_max)
        _BLOCK_CACHE[key] = hit
    return hit


def build_frozen(epoch_override=100):
    model, cfg = H.build_model()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, cfg


def make_bridge_model(model):
    """Deep copy, then unfreeze only the HCA stage."""
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad = False
    n_train = 0
    for name, p in m.named_parameters():
        if any(name.startswith(pref) for pref in HCA_PREFIX):
            p.requires_grad = True
            n_train += p.numel()
    m.train()
    # frozen submodules must stay in eval mode (BN / dropout)
    for name, mod in m.named_modules():
        if not any(name.startswith(pref) for pref in HCA_PREFIX):
            mod.eval()
    return m, n_train


def run_bridge(model_b, pts, q_offset, q_x, q_map, cls):
    """Bridge-path forward through model_b (identical to H._run_bridge)."""
    if pts.shape[0] == 0:
        return None
    y = torch.ones(pts.shape[0], dtype=torch.long, device=pts.device)
    off = torch.tensor([pts.shape[0]], dtype=torch.int32).to(pts.device)
    lg, _ = model_b(off, pts, y, q_offset, q_x, q_map, 100, None, None,
                    np.array([cls]))
    return lg[0]  # [2, N] -- keep both classes so the CE loss is well defined


def evaluate(model_s, model_b, specs, by, cfg, K, lam, max_eps):
    """Same-process paired evaluation: base and fused come from the same
    forward pass, so no cross-process drift can leak in."""
    dev = next(model_s.parameters()).device
    base_ious, fused_ious = [], []
    n = 0
    for spec in specs:
        if n >= max_eps:
            break
        try:
            c = spec['class']
            s_info, q_info = by[spec['support']], by[spec['query']]
            u_infos = [by[nm] for nm in spec['u_cross'] if nm in by][:2]
            if not u_infos:
                continue
            s_coord, s_feat, s_label, _ = load_block(s_info, cfg)
            q_coord, q_feat, q_label, _ = load_block(q_info, cfg)
            u_list = [load_block(i, cfg) for i in u_infos]
            u_coord = torch.cat([x[0] for x in u_list], 0)
            u_feat = torch.cat([x[1] for x in u_list], 0)
            u_seg_raw = torch.cat([x[3] for x in u_list], 0)
            u_seg = u_seg_raw.clone()
            off = 0
            for i, x in enumerate(u_list):
                nn = x[0].shape[0]
                u_seg[off:off + nn] = u_seg_raw[off:off + nn] + i * 10 ** 7
                off += nn
            _, u_seg = torch.unique(u_seg, return_inverse=True)

            s_bin = (s_label == c).long().to(dev)
            q_map = (q_label == c).long().to(dev)
            support_x = torch.cat([s_coord, s_feat], 1).to(dev)
            q_x = torch.cat([q_coord, q_feat], 1).to(dev)
            u_x = torch.cat([u_coord, u_feat], 1).to(dev)
            s_off = torch.tensor([s_coord.shape[0]], dtype=torch.int32).to(dev)
            q_off = torch.tensor([q_coord.shape[0]], dtype=torch.int32).to(dev)
            u_off = torch.tensor([u_x.shape[0]], dtype=torch.int32).to(dev)
            seg_full = u_seg.to(dev)

            with torch.no_grad():
                q_pred, _ = model_s(s_off, support_x, s_bin, q_off, q_x,
                                    q_map, 100, None, None, np.array([c]))
                base = q_pred[0]
                u_pred, _ = model_s(s_off, support_x, s_bin, u_off, u_x,
                                    torch.zeros_like(seg_full), 100, None,
                                    None, np.array([c]))
                p = F.softmax(u_pred[0], dim=0)
                q_reg = H.agg_mean(p[1], seg_full, int(seg_full.max()) + 1)
                cand = (q_reg > 0.6).nonzero().flatten()
                if cand.numel() == 0:
                    continue
                order = torch.argsort(q_reg[cand], descending=True)
                sel = cand[order[:min(K, cand.numel())]]
                m = torch.zeros(u_x.shape[0], dtype=torch.bool, device=dev)
                for r in sel:
                    m |= (seg_full == r)
                pts = u_x[m]
                lg = run_bridge(model_b, pts, q_off, q_x, q_map, c)
                if lg is None:
                    continue
                fused = (1 - lam) * base + lam * lg

            def iou_of(logits):
                pred = (logits[1] > logits[0]).long()
                tp = ((pred == 1) & (q_map == 1)).sum().item()
                fp = ((pred == 1) & (q_map == 0)).sum().item()
                fn = ((pred == 0) & (q_map == 1)).sum().item()
                return tp / max(tp + fp + fn, 1)

            base_ious.append(iou_of(base))
            fused_ious.append(iou_of(fused))
            n += 1
        except Exception as e:
            print('  eval skip:', repr(e)[:60], flush=True)
            continue
    if not base_ious:
        return None
    b = np.array(base_ious)
    f = np.array(fused_ious)
    d = f - b
    return {'n': len(b), 'base': float(b.mean()), 'fused': float(f.mean()),
            'delta': float(d.mean()),
            'se': float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/data01/jh/data/ScanNet/blocks_bs1_s1/data')
    ap.add_argument('--out', default='/data01/jh/data/s2')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--steps-per-epoch', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--noise-neg', type=int, default=1,
                    help='hard negatives injected into the bridge set')
    ap.add_argument('--val-manifest', default='/data01/jh/data/manifest_fold0.json')
    ap.add_argument('--val-every', type=int, default=2)
    ap.add_argument('--val-eps', type=int, default=100,
                    help='validate on this many manifest episodes each epoch; '
                         'base and fused come from the same forward pass')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    print('scanning blocks...', flush=True)
    blocks = H.scan_blocks(args.data)
    name2id = {b['name']: b for b in blocks}

    class_names = H.load_class_names()
    novel_ids = [i for i, n in enumerate(class_names) if n in H.NOVEL_FOLD0]
    base_ids = [i for i in range(len(class_names))
                if i not in novel_ids and i != 0]
    print('novel:', novel_ids)
    print('base :', base_ids)

    class2blocks = {c: [b for b in blocks if c in b['classes']]
                    for c in base_ids}

    model_s, cfg = build_frozen()
    model_b, n_par = make_bridge_model(model_s)
    print('trainable HCA params: %.2fM' % (n_par / 1e6), flush=True)

    params = [p for p in model_b.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)

    # fixed validation set: reuse the novel manifest, first N episodes
    val = json.load(open(args.val_manifest))['episodes'][:120]

    def episode_pool():
        c = int(np.random.choice(base_ids))
        cand = class2blocks.get(c, [])
        if len(cand) < 3:
            return None
        order = np.random.permutation(len(cand))
        s_info, q_info = cand[order[0]], cand[order[1]]
        used = {s_info['scene'], q_info['scene']}
        u_pool = [b for b in cand if b['scene'] not in used]
        if len(u_pool) < 2:
            return None
        pick = np.random.choice(len(u_pool), 2, replace=False)
        return c, s_info, q_info, [u_pool[i] for i in pick]

    def one_step():
        ep = None
        for _ in range(20):
            ep = episode_pool()
            if ep is not None:
                break
        if ep is None:
            return None
        c, s_info, q_info, u_infos = ep
        try:
            s_coord, s_feat, s_label, _ = load_block(s_info, cfg)
            q_coord, q_feat, q_label, _ = load_block(q_info, cfg)
            u_list = [load_block(i, cfg) for i in u_infos]
            u_coord = torch.cat([x[0] for x in u_list], 0)
            u_feat = torch.cat([x[1] for x in u_list], 0)
            u_seg_raw = torch.cat([x[3] for x in u_list], 0)
            u_seg = u_seg_raw.clone()
            off = 0
            for i, x in enumerate(u_list):
                n = x[0].shape[0]
                u_seg[off:off + n] = u_seg_raw[off:off + n] + i * 10 ** 7
                off += n
            _, u_seg = torch.unique(u_seg, return_inverse=True)

            dev = next(model_s.parameters()).device
            s_bin = (s_label == c).long().to(dev)
            q_map = (q_label == c).long().to(dev)
            support_x = torch.cat([s_coord, s_feat], 1).to(dev)
            q_x = torch.cat([q_coord, q_feat], 1).to(dev)
            u_x = torch.cat([u_coord, u_feat], 1).to(dev)
            s_off = torch.tensor([s_coord.shape[0]], dtype=torch.int32).to(dev)
            q_off = torch.tensor([q_coord.shape[0]], dtype=torch.int32).to(dev)
            u_off = torch.tensor([u_x.shape[0]], dtype=torch.int32).to(dev)
            seg_full = u_seg.to(dev)

            with torch.no_grad():
                # support -> query : the frozen baseline logits
                q_pred, _ = model_s(s_off, support_x, s_bin, q_off, q_x,
                                    q_map, 100, None, None, np.array([c]))
                base_logits = q_pred[0]  # [2, N]
                # support -> U : per-superpoint foreground probability
                u_pred, _ = model_s(s_off, support_x, s_bin, u_off, u_x,
                                    torch.zeros_like(seg_full), 100, None,
                                    None, np.array([c]))
                p = F.softmax(u_pred[0], dim=0)
                q_reg = H.agg_mean(p[1], seg_full, int(seg_full.max()) + 1)

            cand = (q_reg > 0.6).nonzero().flatten()
            if cand.numel() == 0:
                return None
            order = torch.argsort(q_reg[cand], descending=True)
            kk = min(args.K, cand.numel())
            sel = cand[order[:kk]]
            if args.noise_neg > 0:
                bg = (q_reg < 0.6).nonzero().flatten()
                if bg.numel():
                    bo = torch.argsort(q_reg[bg], descending=True)
                    nn_ = min(args.noise_neg, bg.numel())
                    sel = torch.cat([sel, bg[bo[:nn_]]])

            m = torch.zeros(u_x.shape[0], dtype=torch.bool, device=dev)
            for r in sel:
                m |= (seg_full == r)
            pts = u_x[m]
            if pts.shape[0] == 0:
                return None

            lg_b = run_bridge(model_b, pts, q_off, q_x, q_map, c)
            if lg_b is None:
                return None
            fused = (1 - args.lam) * base_logits + args.lam * lg_b  # [2, N]
            # Background dominates the query (foreground is a few percent), so
            # an unweighted CE is minimised by predicting all-background --
            # which drove the validation delta to -17pp within 20 steps.
            # Mirror COSeg's own weighting: 0.1 for background, 1.0 for fg.
            w = torch.tensor([0.1, 1.0], device=fused.device)
            return F.cross_entropy(fused.t(), q_map, weight=w)
        except Exception as e:
            print('step failed:', repr(e), flush=True)
            return None

    print('training...', flush=True)
    log = []
    for ep in range(args.epochs):
        tot, nb = 0.0, 0
        for st in range(args.steps_per_epoch):
            loss = one_step()
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss)
            nb += 1
            if (st + 1) % 50 == 0:
                print('  ep%d %d/%d loss %.4f'
                      % (ep, st + 1, args.steps_per_epoch, tot / max(nb, 1)),
                      flush=True)
        print('epoch %d done, mean loss %.4f over %d steps'
              % (ep, tot / max(nb, 1), nb), flush=True)
        torch.save({'hca': {k: v for k, v in model_b.state_dict().items()
                            if any(k.split('.')[0].startswith(p)
                                   for p in HCA_PREFIX)},
                    'args': vars(args)}, os.path.join(args.out, 'ep%d.pt' % ep))
        ev = evaluate(model_s, model_b, val, name2id, cfg, args.K,
                      args.lam, args.val_eps)
        if ev is not None:
            ev['epoch'] = ep
            ev['loss'] = tot / max(nb, 1)
            ev['steps'] = nb
            log.append(ev)
            print('  VAL ep%d n=%d base %.4f fused %.4f delta %+.4f +/- %.4f'
                  % (ep, ev['n'], ev['base'], ev['fused'], ev['delta'],
                     ev['se']), flush=True)
        with open(os.path.join(args.out, 'train_log.json'), 'w') as f:
            json.dump(log, f, indent=1)

    print('done')


if __name__ == '__main__':
    main()
