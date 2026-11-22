"""Baseline query mIoU for COSeg 1-way 1-shot on ScanNet val.

Used as the sanity gate from the task spec: make sure the frozen model
reproduces reasonable novel-class performance before trusting B/C stats.
"""
import sys

sys.path.insert(0, '/data01/jh/code')
import fake_tp  # noqa: F401,E402
sys.path.insert(0, '/data01/jh/code/COSeg')

import numpy as np  # noqa: E402
import torch  # noqa: E402

from diag_bc import (  # noqa: E402
    build_model, data_prepare_seg, scan_blocks, load_class_names, NOVEL_FOLD0,
)

DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'


def run_query(model, cfg, s_info, q_info, c):
    def load(info):
        d = np.load(info['path'])
        return data_prepare_seg(
            d[:, 0:3], d[:, 3:6], d[:, 6], d[:, 7],
            cfg.voxel_size, cfg.voxel_max,
        )

    s_coord, s_feat, s_label, _ = load(s_info)
    q_coord, q_feat, q_label, _ = load(q_info)
    device = torch.device('cuda:0')
    s_bin = (s_label == c).long().to(device)
    q_map = (q_label == c).long().to(device)

    with torch.no_grad():
        pred, _ = model(
            torch.tensor([s_coord.shape[0]], dtype=torch.int32).to(device),
            torch.cat([s_coord, s_feat], 1).to(device), s_bin,
            torch.tensor([q_coord.shape[0]], dtype=torch.int32).to(device),
            torch.cat([q_coord, q_feat], 1).to(device), q_map,
            100, None, None, np.array([c]),
        )
    p = pred[0].argmax(0)
    tp = ((p == 1) & (q_map == 1)).sum().item()
    fp = ((p == 1) & (q_map == 0)).sum().item()
    fn = ((p == 0) & (q_map == 1)).sum().item()
    return tp / max(tp + fp + fn, 1)


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    names = load_class_names()
    name2id = {n: i for i, n in enumerate(names)}
    novel_ids = [name2id[n] for n in NOVEL_FOLD0]

    blocks = scan_blocks(DATA)
    class2blocks = {c: [b for b in blocks if c in b['classes']] for c in novel_ids}

    model, cfg = build_model()
    ious = []
    for ep in range(100):
        c = int(np.random.choice(novel_ids))
        cand = class2blocks[c]
        if len(cand) < 2:
            continue
        np.random.shuffle(cand)
        s, q = cand[0], cand[1]
        if s['scene'] == q['scene']:
            continue
        try:
            iou = run_query(model, cfg, s, q, c)
        except Exception as e:
            print('ep', ep, 'failed', repr(e), flush=True)
            continue
        ious.append(iou)
        if (ep + 1) % 10 == 0:
            print('ep', ep + 1, 'mean IoU %.4f (n=%d)' % (np.mean(ious), len(ious)),
                  flush=True)
    print('FINAL mean query IoU: %.4f  n=%d' % (np.mean(ious), len(ious)))


if __name__ == '__main__':
    main()
