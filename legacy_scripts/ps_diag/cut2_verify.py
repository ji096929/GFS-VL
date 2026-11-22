"""Cut 2 前置验证：确认 5 层 hook 与 pooling_inverse 链完全正确。

判据（全部必须通过）：
  V1  enc4/dec3/dec2/dec1/dec0 特征长度 == 对应层级的点数
  V2  用 pooling_inverse 链把每层映射回全分辨率后，长度 == ip 长度
  V3  用 dec0 特征 + seg_head/novel_seg_head 复算 logits，与 model 原始输出
      逐元素一致（max|diff| < 1e-5）—— 证明 dec0 hook 抓的就是 head 输入
  V4  打印每个 fragment 的 index 是否有重复（重复会让概率累加虚高）
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')

from pointcept.datasets import build_dataset, collate_fn  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402

CFG = '/data01/jh/code/GFS-VL/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1.py'
W = '/data01/jh/code/GFS-VL/exp/scannet200/sc200_k1_ONLINE_s42/model/regis1_model_last.pth'


class Probe:
    """抓 enc0..enc4 / dec0..dec3 的 feat，以及 4 个 unpool 的 inverse。"""

    def __init__(self):
        self.enc, self.dec, self.inv = {}, {}, {}
        self.handles = []

    def attach(self, backbone):
        for s in range(5):
            blk = backbone.enc._modules[f'enc{s}']
            self.handles.append(blk.register_forward_hook(
                lambda m, i, o, s=s: self.enc.__setitem__(
                    s, o.feat.detach().clone())))
        for s in range(4):
            sub = backbone.dec._modules[f'dec{s}']
            self.handles.append(sub.register_forward_hook(
                lambda m, i, o, s=s: self.dec.__setitem__(
                    s, o.feat.detach().clone())))
            up = sub._modules['up']
            self.handles.append(up.register_forward_pre_hook(
                lambda m, i, s=s: self.inv.__setitem__(
                    s, i[0]['pooling_inverse'].detach().clone())))

    def to_full(self, feat, start_level):
        """start_level: 特征当前分辨率编码。0=全,1=1/2,2=1/4,3=1/8,4=1/16。
        1/8 对应 dec3/enc3；需要依次用 inv[2],inv[1],inv[0] 上采样。"""
        f = feat
        for s in range(start_level - 1, -1, -1):
            f = f[self.inv[s]]
        return f

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def main():
    cfg = default_config_parser(CFG, {})
    cfg.save_path = '/tmp/cut2_verify'
    cfg.weight = W
    cfg = default_setup(cfg)

    model = build_model(cfg.model).cuda().eval()
    ck = torch.load(W, map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    miss = model.load_state_dict(sd, strict=False)
    print('load missing=%d unexpected=%d' % (len(miss.missing_keys),
                                             len(miss.unexpected_keys)),
          flush=True)
    model.backbone.shuffle_orders = False   # 确定性

    pb = Probe()
    pb.attach(model.backbone)

    ds = build_dataset(cfg.data.test)
    ds.data_list = ds.data_list[:1]
    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda b: b)
    data_dict = next(iter(loader))
    if isinstance(data_dict, (list, tuple)):
        data_dict = data_dict[0]
    frags = data_dict.pop('fragment_list')
    segment = data_dict.pop('segment')
    name = data_dict.pop('name')
    print('scene %s | 原始点数 %d | fragment 数 %d'
          % (name, segment.size, len(frags)), flush=True)

    n = int(segment.size)
    prob = torch.zeros((n, 57), device='cuda')
    cover = torch.zeros(n, device='cuda')
    ok_v1 = ok_v2 = ok_v3 = True

    for fi, frag in enumerate(frags):
        inp = collate_fn([frag])
        for k in inp:
            if isinstance(inp[k], torch.Tensor):
                inp[k] = inp[k].cuda(non_blocking=True)
        ip = inp['index']
        with torch.no_grad():
            out = model(inp)['seg_logits']
        # ---- V1：各层长度
        lens = {('enc%d' % s): pb.enc[s].shape[0] for s in range(5)}
        lens.update({('dec%d' % s): pb.dec[s].shape[0] for s in range(4)})
        # ---- V2：映射回全分辨率
        full = {
            'enc4': pb.to_full(pb.enc[4], 4),
            'dec3': pb.to_full(pb.dec[3], 3),
            'dec2': pb.to_full(pb.dec[2], 2),
            'dec1': pb.to_full(pb.dec[1], 1),
            'dec0': pb.dec[0],
        }
        # ---- V3：dec0 复算 logits
        with torch.no_grad():
            re = torch.cat([model.seg_head(full['dec0']),
                            model.novel_seg_head(full['dec0'])], dim=1)
        dmax = float((re - out).abs().max())
        if fi == 0:
            print('层级点数: %s' % lens, flush=True)
            print('映射后长度: %s' % {k: v.shape[0] for k, v in full.items()},
                  flush=True)
            print('ip 长度: %d | 原始点名数 %d | index 去重后 %d'
                  % (ip.numel(), n, torch.unique(ip).numel()), flush=True)
            print('V3 dec0 复算 logits max|diff| = %.3e' % dmax, flush=True)
        if len(set(v.shape[0] for v in full.values())) != 1:
            ok_v2 = False
        if full['dec0'].shape[0] != ip.numel():
            ok_v2 = False
        if dmax > 1e-5:
            ok_v3 = False
        if len(set(lens.values())) < 2:
            ok_v1 = False
        prob[ip] += torch.softmax(out, -1)
        cover[ip] += 1

    print('V1 各层长度 %s' % ('通过' if ok_v1 else '失败'), flush=True)
    print('V2 映射回全分辨率 %s' % ('通过' if ok_v2 else '失败'), flush=True)
    print('V3 dec0 == head 输入 %s' % ('通过' if ok_v3 else '失败'), flush=True)
    cov = cover.cpu().numpy()
    print('V4 覆盖次数 min=%d max=%d | 被覆盖点 %d/%d | 重复覆盖点 %d'
          % (cov.min(), cov.max(), int((cov > 0).sum()), n,
             int((cov > 1).sum())), flush=True)


if __name__ == '__main__':
    main()
