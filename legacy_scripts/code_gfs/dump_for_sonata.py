"""把 357 个官方口径任务的 support/query 点云与 COSeg 的 p0 落盘，
供 sonata 环境（torch 2.5）二次读取（pointops 与 sonata 环境不兼容，必须分两步）。

每个任务存一个 npz：
  s_coord/s_rgb/s_lab  support 的坐标、RGB(0-255)、二值标签
  q_coord/q_rgb/q_lab  query 同上
  q_p0                 query 每点的前景概率（原始分辨率，与 q_lab 对齐）
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/data01/jh/code")
sys.path.insert(0, "/data01/jh/code/COSeg")
import fake_tp  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as Fs  # noqa: E402

from ocsb_core import (CACHE, build_episodes, build_proto, episode_tensors,  # noqa: E402
                       feats_one, load_model, predict_agg, prepare_block,
                       seed_everything)

OUT = "/data01/jh/data/sonata_dump"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--min-gt", type=float, default=0.05)
    ap.add_argument("--need-support-gt", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    eps = build_episodes(56)
    gtr = json.load(open(os.path.join(CACHE, "val_query_gt.json")))
    sgr = json.load(open(os.path.join(CACHE, "val_support_gt.json")))

    def qk(m):
        return "%s|%d" % (m["query"], m["class"])

    def sk(m):
        return "%s|%d" % (m["support"], m["class"])

    keep = [i for i, m in enumerate(eps)
            if gtr[qk(m)] >= args.min_gt and sgr[sk(m)] >= args.need_support_gt]
    todo = keep[args.shard::args.shards]
    print("kept %d / %d, this shard %d" % (len(keep), len(eps), len(todo)), flush=True)

    model, cfg = load_model()
    dev = torch.device("cuda:0")
    done = 0
    for g in todo:
        m = eps[g]
        out = os.path.join(OUT, "ep%03d.npz" % g)
        if os.path.exists(out):
            done += 1
            continue
        seed_everything(10000 + g)
        c = m["class"]
        s = prepare_block(m["support"], cfg)
        q = prepare_block(m["query"], cfg)
        s_x, s_bin = episode_tensors(s, c)
        q_x, q_map = episode_tensors(q, c)
        sf = feats_one(model, s_x, s_bin, dev)
        qf = feats_one(model, q_x, q_map, dev)
        fg_s = sf["yl"].bool()
        sparse_s = torch.cat([build_proto(model, sf, ~fg_s),
                              build_proto(model, sf, fg_s)], 0)
        pred, _ = predict_agg(model, qf, [sparse_s], 0.0, dev)
        p0 = Fs.softmax(pred, dim=0)[1].cpu().numpy().astype(np.float32)
        np.savez_compressed(
            out,
            s_coord=s["coord"].numpy().astype(np.float32),
            s_rgb=(s["feat"] * 255.0).numpy().astype(np.float32),
            s_lab=s_bin.numpy().astype(np.uint8),
            q_coord=q["coord"].numpy().astype(np.float32),
            q_rgb=(q["feat"] * 255.0).numpy().astype(np.float32),
            q_lab=q_map.numpy().astype(np.uint8),
            q_p0=p0,
            cls=np.array([c], dtype=np.int64))
        done += 1
        if done % 20 == 0:
            print("  %d / %d" % (done, len(todo)), flush=True)
    print("done", done)


if __name__ == "__main__":
    main()
