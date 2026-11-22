"""Minimal COSeg forward check: load weights, run one support+query episode."""
import sys
sys.path.insert(0, '/data01/jh/code')
import fake_tp  # noqa: E402
sys.path.insert(0, '/data01/jh/code/COSeg')

import numpy as np
import torch
from util.config import load_cfg_from_cfg_file, merge_cfg_from_list
from util.data_util import voxelize
from model.coseg import COSeg
from util.common_util import load_pretrain_checkpoint

CFG_PATH = '/data01/jh/code/COSeg/config/scannetv2_COSeg_fs.yaml'
DATA = '/data01/jh/data/ScanNet/blocks_bs1_s1/data'


def data_prepare_seg(coord, feat, label, seg, split, voxel_size, voxel_max):
    """data_prepare with an extra seg column that follows the same sampling."""
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


def build_model():
    cfg = load_cfg_from_cfg_file(CFG_PATH)
    cfg = merge_cfg_from_list(cfg, [
        'data_root', DATA,
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


def run_episode(model, cfg, support_path, query_path, sampled_class):
    s = np.load(support_path)
    q = np.load(query_path)
    s_coord, s_feat, s_label, s_seg = data_prepare_seg(
        s[:, 0:3], s[:, 3:6], s[:, 6], s[:, 7], 'test', cfg.voxel_size, cfg.voxel_max
    )
    q_coord, q_feat, q_label, q_seg = data_prepare_seg(
        q[:, 0:3], q[:, 3:6], q[:, 6], q[:, 7], 'test', cfg.voxel_size, cfg.voxel_max
    )

    s_bin = (s_label == sampled_class).long()
    q_map = torch.zeros_like(q_label)
    q_map[q_label == sampled_class] = 1

    support_x = torch.cat([s_coord, s_feat], 1).cuda()
    query_x = torch.cat([q_coord, q_feat], 1).cuda()
    support_offset = torch.tensor([s_coord.shape[0]], dtype=torch.int32).cuda()
    query_offset = torch.tensor([q_coord.shape[0]], dtype=torch.int32).cuda()

    with torch.no_grad():
        final_pred, loss = model(
            support_offset, support_x, s_bin.cuda(),
            query_offset, query_x, q_map.cuda(),
            100, None, None, np.array([sampled_class]),
        )

    pred = final_pred[0].argmax(0)
    tp = ((pred == 1) & (q_map.cuda() == 1)).sum().item()
    fp = ((pred == 1) & (q_map.cuda() == 0)).sum().item()
    fn = ((pred == 0) & (q_map.cuda() == 1)).sum().item()
    iou = tp / max(tp + fp + fn, 1)
    print('support points (after voxel):', s_coord.shape[0])
    print('query points (after voxel):', q_coord.shape[0])
    print('final_pred shape:', tuple(final_pred.shape))
    print('tp/fp/fn:', tp, fp, fn, 'IoU:', round(iou, 4))
    print('query novel points:', int((q_map == 1).sum()))


if __name__ == '__main__':
    model, cfg = build_model()
    print('model loaded, params:', sum(p.numel() for p in model.parameters()))
    import glob
    files = sorted(glob.glob(DATA + '/*.npy'))
    # find two blocks containing chair (class 4, fold0 novel)
    picked = []
    for f in files[:200]:
        d = np.load(f)
        if (d[:, 6] == 4).sum() > 1000:
            picked.append(f)
        if len(picked) == 2:
            break
    print('picked:', picked)
    run_episode(model, cfg, picked[0], picked[1], 4)
