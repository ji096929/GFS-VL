"""Cut 2 公共库：PTv3 逐层特征提取（精确 unpool 映射）+ 模型/数据构建。

层位与分辨率编码（start_level）：
    enc4  1/16 -> 4
    dec3  1/8  -> 3
    dec2  1/4  -> 2
    dec1  1/2  -> 1
    dec0  1    -> 0
映射回全分辨率只用 SerializedUnpooling 前被 pop 的 pooling_inverse，
逐级索引，O(N)、零插值误差。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/data01/jh/code/GFS-VL')

from pointcept.datasets import build_dataset  # noqa: E402
from pointcept.engines.defaults import (  # noqa: E402
    default_config_parser, default_setup)
from pointcept.models import build_model  # noqa: E402

CFG = '/data01/jh/code/GFS-VL/configs/scannet200/semseg-pt-v3m1-0-gfsregistrain_k1.py'
ROOT = '/data01/jh/code/GFS-VL/exp/scannet200'
CKPTS = {
    'S1': ROOT + '/ptv3_bk/model/model_best.pth',
    'S2': ROOT + '/sc200_k1_ONLINE_s42/model/regis1_model_last.pth',
    'S2s7': ROOT + '/sc200_k1_ONLINE_s7/model/regis1_model_last.pth',
    # 40 epoch（regis1 20 + regis2 20），用于疑点 3 的坍塌是否随轮数缓解
    'S2_40ep': ROOT + '/sc200_k1_offline/model/regis2_model_last.pth',
    'S1_ontr': ROOT + '/ptv3_bk_ontr.pth',
}
LAYERS = ['enc4', 'dec3', 'dec2', 'dec1', 'dec0']
LEVEL = {'enc4': 4, 'dec3': 3, 'dec2': 2, 'dec1': 1, 'dec0': 0}
DIM = {'enc4': 512, 'dec3': 256, 'dec2': 128, 'dec1': 64, 'dec0': 64}
NO, NC = 12, 57


class Probe:
    def __init__(self):
        self.enc, self.dec, self.inv = {}, {}, {}
        self.handles = []

    def attach(self, backbone):
        for s in range(5):
            blk = backbone.enc._modules[f'enc{s}']
            self.handles.append(blk.register_forward_hook(
                lambda m, i, o, s=s: self.enc.__setitem__(
                    s, o.feat.detach())))
        for s in range(4):
            sub = backbone.dec._modules[f'dec{s}']
            self.handles.append(sub.register_forward_hook(
                lambda m, i, o, s=s: self.dec.__setitem__(
                    s, o.feat.detach())))
            up = sub._modules['up']
            self.handles.append(up.register_forward_pre_hook(
                lambda m, i, s=s: self.inv.__setitem__(
                    s, i[0]['pooling_inverse'])))

    def _to_full(self, feat, start_level):
        f = feat
        for s in range(start_level - 1, -1, -1):
            f = f[self.inv[s]]
        return f

    def layers(self):
        """返回 {层名: 全分辨率特征}。"""
        out = {
            'enc4': self._to_full(self.enc[4], 4),
            'dec3': self._to_full(self.dec[3], 3),
            'dec2': self._to_full(self.dec[2], 2),
            'dec1': self._to_full(self.dec[1], 1),
            'dec0': self.dec[0],
        }
        return out

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def build_model_and_cfg(tag):
    cfg = default_config_parser(CFG, {})
    # 每个进程独立 save_path，避免 default_setup 写快照互相覆盖
    cfg.save_path = os.environ.get('CUT2_TMP', '/tmp/cut2_%s' % tag)
    cfg.weight = CKPTS[tag]
    cfg = default_setup(cfg)
    model = build_model(cfg.model).cuda().eval()
    ck = torch.load(CKPTS[tag], map_location='cpu')
    sd = {k[7:] if k.startswith('module.') else k: v
          for k, v in ck['state_dict'].items()}
    if tag in ('S1', 'S1_ontr'):
        bb = {k: v for k, v in sd.items() if k.startswith('backbone.')}
        m, u = model.load_state_dict(bb, strict=False)
        model.seg_head.load_state_dict(
            {'weight': sd['seg_head.weight'], 'bias': sd['seg_head.bias']})
        print('S1: backbone 载入 missing=%d unexpected=%d | seg_head 12x64 载入'
              % (len(m), len(u)), flush=True)
    else:
        m, u = model.load_state_dict(sd, strict=False)
        print('%s: 载入 missing=%d unexpected=%d' % (tag, len(m), len(u)),
              flush=True)
    model.backbone.shuffle_orders = False     # 确定性推理
    print('%s: epoch=%s' % (tag, ck.get('epoch')), flush=True)
    return model, cfg


def make_ds(cfg, split, data_list=None):
    """用 test 配置（fragment 推理）构建数据集，可替换 split / data_list。"""
    d = cfg.data.test.copy()
    d.split = split
    ds = build_dataset(d)
    if data_list is not None:
        ds.data_list = list(data_list)
    return ds


def support_list(cfg):
    """regis1（seed=10）的 45 个 1-shot 支持场景路径。

    ScanNet200Dataset_REGIS 的 data_list 元素是 (class_id_str, abs_scan)。
    """
    ds = build_dataset(cfg.data.regis1)
    out = []
    for x in ds.data_list:
        out.append(x[1] if isinstance(x, (tuple, list)) else x)
    return out
