"""Step 1：保存 312 val 的逐点 top-5 概率/索引/预测/GT，供冲突区仲裁分析。

按 tester 的分片推理；输出 /data01/jh/data/val_logits/{scene}.npz
  top5p float16[N,5]  top5i uint8[N,5]  pred int8[N]  gt int8[N]
点位顺序与 /data01/jh/data/scannet200_gfs/val/{scene}/coord.npy 一致。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, '/data01/jh/code/GFS-VL')

from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402

NC = 57
OUT = sys.argv[3] if len(sys.argv) > 3 else '/data01/jh/data/val_logits'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config'); ap.add_argument('--weight')
    ap.add_argument('--out'); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    a, _ = ap.parse_known_args()
    cfg = default_config_parser(a.config or sys.argv[1], {})
    cfg.save_path = '/tmp/cm_probe'
    cfg.weight = a.weight or sys.argv[2]
    cfg = default_setup(cfg)

    model = build_model(cfg.model).cuda().eval()
    ck = torch.load(cfg.weight, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    model.load_state_dict(sd, strict=False)
    print('weight', cfg.weight, flush=True)

    ds = build_dataset(cfg.data.test)
    if a.shards > 1:
        ds.data_list = ds.data_list[a.shard::a.shards]
        print('shard %d/%d -> %d scenes' % (a.shard, a.shards, len(ds.data_list)), flush=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1,
                        collate_fn=lambda b: b)
    print('scenes', len(ds), flush=True)
    os.makedirs(OUT, exist_ok=True)
    ignore = cfg.data.ignore_index

    t0 = time.time()
    for idx, data_dict in enumerate(loader):
        if isinstance(data_dict, (list, tuple)):
            data_dict = data_dict[0]
        frags = data_dict.pop("fragment_list")
        segment = data_dict.pop("segment")
        name = data_dict.pop("name")
        n = int(segment.size)
        prob = torch.zeros((n, NC), device='cuda')
        for i in range(len(frags)):
            inp = collate_fn(frags[i:i + 1])
            for k in inp:
                if isinstance(inp[k], torch.Tensor):
                    inp[k] = inp[k].cuda(non_blocking=True)
            ip = inp["index"]
            with torch.no_grad():
                logits = model(inp)["seg_logits"]
                p = F.softmax(logits, -1)
                bs = 0
                for be in inp["offset"]:
                    prob[ip[bs:be], :] += p[bs:be]
                    bs = be
        if "inverse" in data_dict:
            prob = prob[data_dict["inverse"]]
        if "origin_segment" in data_dict:
            segment = data_dict["origin_segment"]
        gt = segment.cpu().numpy().astype(np.int8) if torch.is_tensor(segment) \
            else np.asarray(segment, dtype=np.int8)
        top5p, top5i = prob.topk(5, dim=1)
        np.savez_compressed(
            os.path.join(OUT, name + '.npz'),
            top5p=top5p.cpu().numpy().astype(np.float16),
            top5i=top5i.cpu().numpy().astype(np.uint8),
            pred=top5i[:, 0].cpu().numpy().astype(np.int8),
            gt=gt)
        if (idx + 1) % 25 == 0:
            print('  %d/%d  %.0fs' % (idx + 1, len(ds), time.time() - t0), flush=True)
    print('->', OUT, flush=True)


if __name__ == '__main__':
    main()
