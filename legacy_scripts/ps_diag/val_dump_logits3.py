"""Step 1：保存 312 val 的逐点 top-5 概率/索引/预测/GT（支持分片）。

用法: python val_dump_logits2.py <config.py> <weight.pth> [out_dir] [shard] [shards]
输出 {out}/{scene}.npz: top5p float16[N,5] / top5i uint8[N,5] / pred int8[N] / gt int8[N]
点位顺序与 val/{scene}/coord.npy 一致。
"""
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


def main():
    cfg_file = sys.argv[1]
    wpath = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else '/data01/jh/data/val_logits'
    shard = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    shards = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    cfg = default_config_parser(cfg_file, {})
    cfg.save_path = os.environ.get('CM_SAVE', '/tmp/cm_probe')
    cfg.weight = wpath
    cfg = default_setup(cfg)

    model = build_model(cfg.model).cuda().eval()
    ck = torch.load(wpath, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    model.load_state_dict(sd, strict=False)
    print('weight', wpath, '| shard %d/%d' % (shard, shards), flush=True)

    ds = build_dataset(cfg.data.test)
    if shards > 1:
        ds.data_list = ds.data_list[shard::shards]
    print('scenes', len(ds.data_list), flush=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1,
                        collate_fn=lambda b: b)
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    for idx, data_dict in enumerate(loader):
        if isinstance(data_dict, (list, tuple)):
            data_dict = data_dict[0]
        frags = data_dict.pop("fragment_list")
        segment = data_dict.pop("segment")
        name = data_dict.pop("name")
        dst = os.path.join(out, name + '.npz')
        if os.path.exists(dst):
            continue
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
            dst,
            top5p=top5p.cpu().numpy().astype(np.float16),
            top5i=top5i.cpu().numpy().astype(np.uint8),
            pred=top5i[:, 0].cpu().numpy().astype(np.int8),
            gt=gt)
        if (idx + 1) % 20 == 0:
            print('  %d/%d  %.0fs' % (idx + 1, len(ds.data_list),
                                      time.time() - t0), flush=True)
    print('-> shard %d done' % shard, flush=True)


if __name__ == '__main__':
    main()
