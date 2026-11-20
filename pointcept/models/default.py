import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from .builder import MODELS, build_model
from collections import OrderedDict


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def getFeatures(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        return point.feat

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class RegisTrainSegmentor(nn.Module):
    """Base/novel 双头分割器，支持三种 novel 读出增强。

    cross_layers/cross_dims/cross_dim: 跨层读出（dec1/dec2 -> cross_dim）。
    mlp_hidden: >0 时 novel head 改成 dec0-only 的两层 MLP（消融用）。
    base_cross: True 时给 seg_head 加零初始化残差，输入 proj1/proj2。
    depth_gate: True 时用逐点门控缩放 proj1/proj2。
    """

    def __init__(
        self,
        num_base_classes,
        num_novel_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        cross_layers=(),
        cross_dims=(),
        cross_dim=64,
        mlp_hidden=0,
        base_cross=False,
        depth_gate=False,
        detach_cross=False,
        lam_cross=0.0,
        decorr_weight=0.0,
    ):
        super().__init__()
        self.cross_layers = tuple(cross_layers)
        self.mlp_hidden = int(mlp_hidden)
        self.base_cross = bool(base_cross)
        self.depth_gate = bool(depth_gate)
        self.detach_cross = bool(detach_cross)
        self.lam_cross = float(lam_cross)
        self.decorr_weight = float(decorr_weight)
        self._decorr_first = None

        self.seg_head = (
            nn.Linear(backbone_out_channels, num_base_classes)
            if num_base_classes > 0
            else nn.Identity()
        )

        if self.mlp_hidden:
            assert not self.cross_layers
            self.novel_mlp = nn.Sequential(
                nn.LayerNorm(backbone_out_channels),
                nn.Linear(backbone_out_channels, self.mlp_hidden),
                nn.GELU(),
                nn.LayerNorm(self.mlp_hidden),
                nn.Linear(self.mlp_hidden, num_novel_classes, bias=False),
            )
            self.novel_seg_head = None
        elif self.cross_layers:
            assert len(cross_dims) == len(self.cross_layers)
            self.dec0_norm = nn.LayerNorm(backbone_out_channels)
            self.cross_norms = nn.ModuleDict()
            self.cross_projs = nn.ModuleDict()
            self.cross_outs = nn.ModuleDict()
            for s, c in zip(self.cross_layers, cross_dims):
                self.cross_norms[str(s)] = nn.LayerNorm(c)
                self.cross_projs[str(s)] = nn.Linear(c, cross_dim)
                self.cross_outs[str(s)] = nn.LayerNorm(cross_dim)
            novel_in = backbone_out_channels + cross_dim * len(
                self.cross_layers
            )
            self.novel_seg_head = nn.Linear(
                novel_in, num_novel_classes, bias=False
            )
        else:
            self.novel_seg_head = nn.Linear(
                backbone_out_channels, num_novel_classes, bias=False
            )

        if self.base_cross:
            assert self.cross_layers
            self.base_cross_head = nn.Linear(
                cross_dim * len(self.cross_layers),
                num_base_classes,
                bias=False,
            )
            nn.init.zeros_(self.base_cross_head.weight)

        if self.depth_gate:
            assert self.cross_layers
            assert novel_in == backbone_out_channels + cross_dim * len(
                self.cross_layers
            )
            self.depth_gate = nn.Sequential(
                nn.Linear(novel_in, 64),
                nn.GELU(),
                nn.Linear(64, 2),
            )
            nn.init.zeros_(self.depth_gate[-1].weight)
            nn.init.zeros_(self.depth_gate[-1].bias)

        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

        self._cr = {}
        if self.cross_layers:
            n_dec = len(self.backbone.dec._modules)
            for s in range(n_dec):
                up = self.backbone.dec._modules["dec%d" % s]._modules["up"]
                up.register_forward_pre_hook(self._make_inv_hook(s))
            for s in self.cross_layers:
                dec = self.backbone.dec._modules["dec%d" % s]
                dec.register_forward_hook(self._make_feat_hook(s))

    def _make_inv_hook(self, s):
        key = "inv%d" % s

        def hook(module, inputs):
            self._cr[key] = inputs[0]["pooling_inverse"]

        return hook

    def _make_feat_hook(self, s):
        key = "f%d" % s

        def hook(module, inputs, output):
            self._cr[key] = output.feat

        return hook

    def _to_full(self, s):
        feat = self._cr["f%d" % s]
        for t in range(s - 1, -1, -1):
            feat = feat[self._cr["inv%d" % t]]
        return feat

    def _decorr(self, ln0, projs):
        # ln0 必须 detach：正则只推 proj 分支，不碰 dec0 的 Base 流形
        z = torch.cat([ln0.detach()] + list(projs), dim=1).float()
        n = z.shape[0]
        if n > 8192:
            idx = torch.randperm(n, device=z.device)[:8192]
            z = z[idx]
        zh = (z - z.mean(0)) / (z.std(0) + 1e-5)
        c = (zh.T @ zh) / zh.shape[0]
        eye = torch.eye(c.shape[0], device=c.device)
        loss = (c - eye).pow(2).sum() / (c.shape[0] * (c.shape[0] - 1))
        if self._decorr_first is None:
            self._decorr_first = float(loss)
            if 0.1 * self._decorr_first > 0.1:
                self.decorr_weight = 0.02
        return self.decorr_weight * loss

    def _cross_projs(self, feat):
        ln0 = self.dec0_norm(feat.float())
        projs = []
        for s in self.cross_layers:
            _h = self._to_full(s).float()
            if self.detach_cross:
                _h = _h.detach()
            h = self.cross_norms[str(s)](_h)
            h = torch.nn.functional.gelu(self.cross_projs[str(s)](h))
            projs.append(self.cross_outs[str(s)](h))
        if self.depth_gate:
            g = torch.sigmoid(
                self.depth_gate(torch.cat([ln0] + projs, dim=1))
            )
            projs = [g[:, i:i + 1] * projs[i] for i in range(len(projs))]
        return ln0, projs

    def forward(self, input_dict):
        try:
            point = Point(input_dict)
            point = self.backbone(point)
            # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
            # TODO: remove this part after make all backbone return Point only.
            if isinstance(point, Point):
                feat = point.feat
            else:
                feat = point

            base_seg_logits = self.seg_head(feat)
            decorr_loss = None
            cross_logits = None
            if self.mlp_hidden:
                novel_seg_logits = self.novel_mlp(
                    feat.float()
                ).to(feat.dtype)
            elif self.cross_layers:
                ln0, projs = self._cross_projs(feat)
                novel_in = torch.cat([ln0] + projs, dim=1).to(feat.dtype)
                novel_seg_logits = self.novel_seg_head(novel_in)
                if self.base_cross:
                    _cf = torch.cat(projs, dim=1).to(feat.dtype)
                    _bc = self.base_cross_head(_cf)
                    base_seg_logits = base_seg_logits + _bc
                    # L_cross: 在纯跨层子空间做辅助监督，消除 base_cross_head
                    # 被 seg_head(dec0) 遮住梯度的问题（推理时不用）。
                    if self.training and self.lam_cross > 0:
                        _cn = torch.nn.functional.linear(
                            _cf, self.novel_seg_head.weight[:, 64:]
                        )
                        cross_logits = torch.cat([_bc, _cn], dim=1)
                if self.training and self.decorr_weight > 0:
                    decorr_loss = self._decorr(ln0, projs)
            else:
                novel_seg_logits = self.novel_seg_head(feat)
            seg_logits = torch.cat([base_seg_logits, novel_seg_logits], dim=1)
        finally:
            # 必须及时清掉：缓存的中间张量带 grad_fn，留着会跨 iter 泄漏显存
            self._cr.clear()

        # train
        if self.training:
            out = dict(seg_logits=seg_logits)
            if decorr_loss is not None:
                out["decorr_loss"] = decorr_loss
            if cross_logits is not None:
                out["cross_logits"] = cross_logits
            return out
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
