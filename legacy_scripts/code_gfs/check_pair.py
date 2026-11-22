"""对拍：同一 episode 上，官方 COSeg.forward 与我们复刻的 predict_agg 是否给出相同 IoU。"""
import sys

sys.path.insert(0, "/data01/jh/code")
sys.path.insert(0, "/data01/jh/code/COSeg")
import fake_tp  # noqa: F401,E402

import torch  # noqa: E402

import diag_bc as D  # noqa: E402
from ocsb_core import (build_proto, feats_one, iou_counts, predict_agg,  # noqa: E402
                       seed_everything)

CKPT = ("/data01/jh/data/ScanNet/scenes/"
        "Scan_0_N_1_K_1_episodes_20_pts_20480_vs_0.02_rdmsp")

model, cfg = D.build_model()
model.eval()
dev = torch.device("cuda:0")

for idx in [0, 1, 5, 40, 100]:
    o = torch.load("%s/%d.pt" % (CKPT, idx))
    s_x = o["support_feat"][0].float().contiguous().cuda()
    s_y = o["support_label"][0].long().cuda()
    q_x = o["query_feat"][0].float().contiguous().cuda()
    q_y = o["query_label"][0].long().cuda()
    cls = int(o["sampled_classes"][0])
    print("ep %3d cls %2d  s_fg %5d q_fg %6d (%.2f%%)" % (
        idx, cls, int(s_y.sum()), int(q_y.sum()), 100 * float(q_y.float().mean())))

    s_off = torch.tensor([len(s_x)], dtype=torch.int32, device=dev)
    q_off = torch.tensor([len(q_x)], dtype=torch.int32, device=dev)
    with torch.no_grad():
        out, _ = model(s_off, s_x, s_y, q_off, q_x, q_y, 5,
                       sampled_classes=[cls])
    p_off = out.max(1)[1].squeeze(0)
    tp, fp, fn, iou = iou_counts(p_off, q_y)
    print("   official  iou %.4f  tp %6d fp %6d fn %6d  pred_fg %.2f%%" % (
        iou, tp, fp, fn, 100 * float((p_off == 1).float().mean())))

    sf = feats_one(model, s_x, s_y, dev)
    qf = feats_one(model, q_x, q_y, dev)
    fg_s = sf["yl"].bool()
    sparse_s = torch.cat([build_proto(model, sf, ~fg_s),
                          build_proto(model, sf, fg_s)], 0)
    pred_raw, _ = predict_agg(model, qf, [sparse_s], 0.0, dev)
    tp, fp, fn, iou = iou_counts(pred_raw, q_y)
    pf = pred_raw.argmax(0)
    print("   ours      iou %.4f  tp %6d fp %6d fn %6d  pred_fg %.2f%%  "
          "agree %.4f" % (iou, tp, fp, fn, 100 * float((pf == 1).float().mean()),
                          100 * float((pf == p_off).float().mean())))
