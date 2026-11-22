"""COSeg + S+U 可微软原型（Stage II Exp 1）。

forward_su 相对官方 forward 只改三处：
1. 额外对 U 做一次 getFeatures（U 无标签，gt 传全 0，不做 base 监督）
2. 用 S 的 100 个 fg/bg 子原型预测 U，得到逐点软前景概率 p_u（可微）
3. 软原型 p_u_fg = Σ p_u f / Σ p_u，p_u_bg = Σ (1-p_u) f / Σ (1-p_u)，
   与 S 子原型线性融合 (1-λ)s + λu，再用融合原型预测 Q

其余（base prototype EMA、base guidance、loss、插值回原分辨率）与官方一致。
"""
import torch
import torch.nn.functional as F
import torch_points_kernels as tp

from lib.pointops2.functions import pointops
from model.coseg import COSeg


class COSegSU(COSeg):
    def predict_from_protos(self, feat, x_low, off_low_cpu, sparse,
                            epoch, base_avail):
        """用一组 sparse embeddings 逐块预测 logits（Q 或 U 共用）。"""
        out_list = []
        feat_l = torch.tensor_split(feat, off_low_cpu)
        x_l = torch.tensor_split(x_low, off_low_cpu)
        for i, q_feat in enumerate(feat_l):
            if epoch < 1 or base_avail is None:
                base_guidance = None
            else:
                bs = F.cosine_similarity(
                    q_feat[:, None, :], base_avail[None, :, :], dim=2)
                base_guidance = bs.max(dim=1, keepdim=True)[0]
            corr = F.cosine_similarity(
                q_feat[:, None, :], sparse[None, :, :], dim=2)
            corr = self.lin1(
                corr.view(corr.shape[0], self.n_way + 1, -1)
            ).permute(2, 1, 0).unsqueeze(0)
            for layer in self.agglayers:
                corr = layer(corr, base_guidance)
            corr = corr.squeeze(0).permute(2, 1, 0).contiguous()
            if self.n_way == 1:
                corr = self.class_reduce(corr).squeeze(1)
            else:
                corr = self.class_reduce(corr.view(corr.shape[0], -1))
            coord = x_l[i]
            batch = torch.zeros(corr.shape[0], dtype=torch.int64,
                                device=coord.device)
            sigma = 2.0
            radius = 2.5 * self.args.grid_size * sigma
            neighbors = tp.ball_query(
                radius, self.args.max_num_neighbors, coord, coord,
                mode="partial_dense", batch_x=batch, batch_y=batch)[0]
            corr = self.kpconv(corr, coord, batch, neighbors.clone())
            out_list.append(self.cls(corr))
        return torch.cat(out_list)

    def forward_su(self, support_offset, support_x, support_y,
                   u_offset, u_x,
                   query_offset, query_x, query_y,
                   epoch, support_base_y=None, query_base_y=None,
                   sampled_classes=None, lam=0.3):
        # ---------- 1. S features + 子原型（与官方一致） ----------
        (s_feat, s_x_low, s_off_low, s_y_low, _,
         s_base_low) = self.getFeatures(
            support_x, support_offset, support_y, support_base_y)
        s_off_cpu = s_off_low[:-1].long().cpu()
        s_feat_l = torch.tensor_split(s_feat, s_off_cpu)
        s_x_l = torch.tensor_split(s_x_low, s_off_cpu)
        fg_m = torch.tensor_split(s_y_low.bool(), s_off_cpu)
        bg_m = torch.tensor_split(~s_y_low.bool(), s_off_cpu)
        fg_protos = self.getPrototypes(
            s_x_l, s_feat_l, fg_m, k=self.n_subprototypes // self.k_shot)
        bg_proto = self.getPrototypes(
            s_x_l, s_feat_l, bg_m, k=self.n_subprototypes // self.k_shot)
        sparse_s = torch.cat([bg_proto, fg_protos])   # (200, C)

        # ---------- 2. Q features（官方在更新 base 前算 Q features） ----------
        (q_feat, q_x_low, q_off_low, q_y_low, q_base_pred,
         q_base_y_low) = self.getFeatures(
            query_x, query_offset, query_y, query_base_y)
        q_off_cpu = q_off_low[:-1].long().cpu()

        # ---------- 3. base prototypes EMA + base guidance（官方逻辑） ----------
        if self.training:
            q_base_l = (torch.tensor_split(q_base_y_low, q_off_cpu)
                        if q_base_y_low is not None
                        else [None] * len(q_off_cpu))
            s_base_l = (torch.tensor_split(s_base_low, s_off_cpu)
                        if s_base_low is not None
                        else [None] * len(s_off_cpu))
            for base_feat, base_y in zip(
                list(torch.tensor_split(q_feat, q_off_cpu)) + list(s_feat_l),
                list(q_base_l) + list(s_base_l),
            ):
                if base_y is None:
                    continue
                cur = base_y.unique()
                cur = cur[cur != 0]
                for cl in cur:
                    m = base_y == cl
                    cf = (base_feat[m].sum(0) / m.sum()).detach()
                    if torch.all(self.base_prototypes[cl - 1] == 0):
                        self.base_prototypes[cl - 1] = cf
                    else:
                        self.base_prototypes[cl - 1] = (
                            self.base_prototypes[cl - 1] * 0.995 + cf * 0.005)
            mask_list = [self.base_class_to_pred_label[bc] - 1
                         for bc in sampled_classes]
            base_mask = self.base_prototypes.new_ones(
                (self.base_prototypes.shape[0]), dtype=torch.bool)
            base_mask[mask_list] = False
            base_avail = self.base_prototypes[base_mask]
        else:
            base_avail = self.base_prototypes

        # ---------- 4. U features（无标签，不做 base 监督） ----------
        u_gt = torch.zeros(u_x.shape[0], dtype=torch.long)
        u_feat, u_x_low, u_off_low, _, _, _ = self.getFeatures(
            u_x, u_offset, u_gt, None)
        u_off_cpu = u_off_low[:-1].long().cpu()

        # ---------- 5. U 的软原型 ----------
        u_logits = self.predict_from_protos(
            u_feat, u_x_low, u_off_cpu, sparse_s, epoch, base_avail)
        p_u = F.softmax(u_logits, dim=1)[:, 1]              # (N_u,)
        w_fg = p_u
        w_bg = 1.0 - p_u
        p_u_fg = (w_fg[:, None] * u_feat).sum(0) / (w_fg.sum() + 1e-8)
        p_u_bg = (w_bg[:, None] * u_feat).sum(0) / (w_bg.sum() + 1e-8)

        # ---------- 6. 融合并预测 Q ----------
        fg_merged = (1.0 - lam) * fg_protos + lam * p_u_fg[None, :]
        bg_merged = (1.0 - lam) * bg_proto + lam * p_u_bg[None, :]
        sparse_su = torch.cat([bg_merged, fg_merged])
        q_pred = self.predict_from_protos(
            q_feat, q_x_low, q_off_cpu, sparse_su, epoch, base_avail)

        loss = self.criterion(q_pred, q_y_low)
        if q_base_y_low is not None:
            loss += self.criterion_base(q_base_pred, q_base_y_low.cuda())

        final_pred = (
            pointops.interpolation(
                q_x_low,
                query_x[:, :3].cuda().contiguous(),
                q_pred.contiguous(),
                q_off_low,
                query_offset.cuda(),
            ).transpose(0, 1).unsqueeze(0)
        )
        return final_pred, loss
