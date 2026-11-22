"""OCSB-CMC：在 ocsb_faithful 之上的三个结构性修正。

Fix 1  式(5) 的 s_c^0 从"单均值原型"换成"支持集局部多子原型的 Top-M 均值"，
       避免椅背/椅腿这类部件差异被误判成 Δ>0。
Fix 2  式(15) 换成与式(10)-(11) 同构的"点条件化正向选择"：
           Δ_q(R,i) = relu(z_R·z_i - c0(i))
           β(R|i)   = w_R Δ_q / (κ_q + Σ w_R' Δ_q)
           δ_corr(i)= Σ β Δ_q  >= 0
       没有任何记忆超点比支持锚点更匹配时 δ_corr 严格为 0，全局负常数消失。
Fix 3  把 δ_corr 注入 COSeg 相关性层（lin1 之前），让余弦增量与 100 维相关性
       同尺度，并穿过冻结的 lin1 -> HCA -> kpconv -> cls。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ocsb_faithful import logmeanexp, proto_norm, sp_mean_norm


class OCSBCMC(nn.Module):
    """OASB + CSR（沿用原式 3-14）+ 多子原型基准 + 点条件化查询增量。"""

    def __init__(self, dim, tau=0.1, tau_b=0.6, kappa=0.1, beta=1.0,
                 eps=1e-6, m_anchor=3, kappa_q=1e-3):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.tau_b = tau_b
        self.kappa = kappa
        self.beta = beta
        self.eps = eps
        self.m_anchor = m_anchor
        self.kappa_q = kappa_q
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    def project(self, h):
        return F.normalize(self.proj(h), dim=-1)

    def weight_reg(self):
        w = self.proj.weight
        eye = torch.eye(self.dim, device=w.device, dtype=w.dtype)
        return ((w - eye) ** 2).sum()

    # ------------------------------------------------------------ Fix 1

    def support_anchors(self, z_s, s_y, s_seg, want_fg=True):
        """支持集超点特征作为局部锚点（前景或纯背景）。"""
        s_seg = s_seg.long()
        n_s = int(s_seg.max().item()) + 1
        z_sR = sp_mean_norm(z_s, s_seg, n_s)
        frac = torch.zeros(n_s, device=z_s.device)
        frac.index_add_(0, s_seg, (s_y > 0.5).float())
        cnt = torch.bincount(s_seg, minlength=n_s).clamp_min(1).float()
        frac = frac / cnt
        keep = (frac > 0.5) if want_fg else (frac <= 0.0)
        a = z_sR[keep]
        if a.shape[0] == 0:
            base = (proto_norm(z_s, s_y > 0.5) if want_fg
                    else proto_norm(z_s, s_y <= 0.5))
            a = base.unsqueeze(0)
        return a

    def topm_score(self, z_R, anchors, m=None):
        """Top-M 均值：s0(R) = mean of top m (anchor · z_R) / tau。"""
        m = self.m_anchor if m is None else m
        sim = (z_R @ anchors.t()) / self.tau
        k = min(m, sim.shape[1])
        return sim.topk(k, dim=1).values.mean(1)

    # ------------------------------------------------------------ forward

    def forward(self, ep, use_csr=True, top_m=None, bg_baseline=False):
        tau = self.tau
        dev = ep['s_h'].device

        z_s = self.project(ep['s_h'])
        z_u = self.project(ep['u_h'])
        z_q = self.project(ep['q_h'])

        u_seg = ep['u_seg'].long()
        n_reg = int(u_seg.max().item()) + 1
        z_R = sp_mean_norm(z_u, u_seg, n_reg)

        a_fg = self.support_anchors(z_s, ep['s_y'], ep['s_seg'], True)
        a_bg = self.support_anchors(z_s, ep['s_y'], ep['s_seg'], False)

        q_reg = torch.zeros(n_reg, dtype=torch.float32, device=dev)
        q_reg.index_add_(0, u_seg, ep['u_p0'].to(torch.float32))
        cnt = torch.bincount(u_seg, minlength=n_reg).clamp_min(1).to(torch.float32)
        q_reg = (q_reg / cnt).clamp(1e-4, 1.0 - 1e-4)

        B_idx = (q_reg >= self.tau_b).nonzero().flatten()
        C_idx = (q_reg < 0.5).nonzero().flatten()

        # Fix 1：多子原型 Top-M
        s0 = self.topm_score(z_R, a_fg)
        s0_bg = self.topm_score(z_R, a_bg)
        # S4：基准也减去背景负对照（前景-背景裕度）
        if bg_baseline:
            neg_all = a_bg
            if ep.get('base_protos') is not None and ep['base_protos'].numel():
                bp = F.normalize(ep['base_protos'].to(z_R.dtype), dim=-1)
                neg_all = torch.cat([neg_all, bp], 0)
            s_neg = self.topm_score(z_R, neg_all)
            s0 = s0 - s_neg
            s0_bg = s0_bg - s_neg

        out = {'q_reg': q_reg, 'z_R': z_R, 's0': s0, 's0_bg': s0_bg,
               'B_idx': B_idx, 'C_idx': C_idx, 'n_reg': n_reg,
               'a_fg': a_fg, 'a_bg': a_bg, 'z_q': z_q}

        if B_idx.numel() == 0:
            out.update({'delta_corr': torch.zeros(z_q.shape[0], device=dev),
                        'M_feat': torch.zeros(0, self.dim, device=dev),
                        'M_w': torch.zeros(0, device=dev),
                        'w_C': torch.zeros(C_idx.numel(), device=dev),
                        'w_B': torch.zeros(0, device=dev),
                        'V': torch.zeros(0, C_idx.numel(), device=dev),
                        'eta': torch.zeros(0, device=dev)})
            return out

        z_B = z_R[B_idx]
        q_B = q_reg[B_idx]
        sim_BC = (z_B @ z_R[C_idx].t()) / tau
        log_qB = torch.log(q_B)
        sB = torch.logaddexp(s0[C_idx].unsqueeze(0),
                             log_qB[:, None] + sim_BC) - torch.log1p(q_B)[:, None]
        delta_pos = sB - s0[C_idx].unsqueeze(0)

        # 式(8) 负对照（同原版，含 C2 扩集）
        eta = torch.zeros(B_idx.numel(), device=dev)
        if use_csr:
            neg_feat, neg_src = self._negative_set(ep, z_s, z_R, q_reg)
            if neg_feat.shape[0] > 0:
                s0_neg = self.topm_score(neg_feat, a_fg)
                sB_neg = torch.logaddexp(
                    s0_neg.unsqueeze(0),
                    log_qB[:, None] + (z_B @ neg_feat.t()) / tau
                ) - torch.log1p(q_B)[:, None]
                delta_neg = sB_neg - s0_neg.unsqueeze(0)
                eta = tau * logmeanexp(F.relu(delta_neg) / tau, dim=1)
                out['neg_src'] = neg_src

        V = delta_pos - eta.unsqueeze(1)
        pos = F.relu(V)
        denom = self.kappa + pos.sum(0)
        alpha = pos / denom.unsqueeze(0)

        s_bar_fg = s0[C_idx] + (alpha * delta_pos).sum(0)
        s_bar = torch.stack([s0_bg[C_idx], s_bar_fg], 0)
        P_bar = F.softmax(s_bar, 0)

        g = (alpha * F.relu(V)).sum(0)
        c0_is_c = (q_reg[C_idx] >= 0.5).float()
        argmax_is_c = (P_bar.argmax(0) == 1).float()
        w_C = (1.0 - c0_is_c) * argmax_is_c * P_bar[1] * (g / (1.0 + g))

        w_B = q_B * (alpha * w_C.unsqueeze(0)).sum(1) / (self.eps + w_C.sum())

        M_feat = torch.cat([z_B, z_R[C_idx]], 0)
        M_w = torch.cat([w_B, w_C], 0)

        # Fix 2：点条件化正向增量
        delta_corr = self.point_conditional_delta(z_q, a_fg, M_feat, M_w,
                                                  top_m=top_m, c0=None)

        out.update({'s_bar': s_bar, 'P_bar': P_bar, 'delta_corr': delta_corr,
                    'w_C': w_C, 'w_B': w_B, 'V': V, 'delta': delta_pos,
                    'eta': eta, 'alpha': alpha, 'M_feat': M_feat, 'M_w': M_w,
                    'q_B': q_B, 'z_B': z_B, 's_bar_c': s_bar_fg})
        return out

    # ------------------------------------------------------------ Fix 2

    def point_conditional_delta(self, z_q, anchors, M_feat, M_w,
                                top_m=None, c0=None):
        """δ_corr(i) = Σ_R β(R|i) Δ_q(R,i)，Δ_q = relu(z_R·z_q - c0)。"""
        if M_feat.shape[0] == 0:
            return torch.zeros(z_q.shape[0], device=z_q.device)
        if c0 is None:
            c0 = self.topm_cosine(z_q, anchors)
        sim = z_q @ M_feat.t()                       # [Nq, K]
        dq = F.relu(sim - c0.unsqueeze(1))           # [Nq, K]
        if top_m is not None and dq.shape[1] > top_m:
            th = dq.topk(top_m, dim=1).values[:, -1:]
            dq = dq * (dq >= th)
        num = M_w.unsqueeze(0) * dq                  # [Nq, K]
        den = self.kappa_q + num.sum(1, keepdim=True)
        beta = num / den
        return (beta * dq).sum(1)

    def topm_cosine(self, z_q, anchors, m=None):
        """c0(i)：支持锚点对查询点的 Top-M 余弦（与 Fix 1 同口径）。"""
        m = self.m_anchor if m is None else m
        sim = z_q @ anchors.t()
        k = min(m, sim.shape[1])
        return sim.topk(k, dim=1).values.mean(1)

    def _negative_set(self, ep, z_s, z_R, q_reg):
        feats, src = [], []
        if ep.get('s_seg') is not None:
            s_seg = ep['s_seg'].long()
            n_s = int(s_seg.max().item()) + 1
            fg = torch.zeros(n_s, device=z_s.device)
            fg.index_add_(0, s_seg, (ep['s_y'] > 0.5).float())
            cnt = torch.bincount(s_seg, minlength=n_s).clamp_min(1).float()
            fg = fg / cnt
            z_sR = sp_mean_norm(z_s, s_seg, n_s)
            keep = fg <= 0.0
            if keep.any():
                feats.append(z_sR[keep])
                src.append(torch.zeros(int(keep.sum()), dtype=torch.int64,
                                       device=z_s.device))
        if ep.get('base_protos') is not None and ep['base_protos'].numel():
            bp = F.normalize(ep['base_protos'].to(z_s.dtype), dim=-1)
            feats.append(bp)
            src.append(torch.ones(bp.shape[0], dtype=torch.int64,
                                  device=z_s.device))
        keep_u = q_reg < 0.05
        if keep_u.any():
            feats.append(z_R[keep_u])
            src.append(torch.full((int(keep_u.sum()),), 2, dtype=torch.int64,
                                  device=z_s.device))
        if not feats:
            return torch.zeros(0, self.dim, device=z_s.device), None
        return torch.cat(feats, 0), torch.cat(src, 0)


class Lin1Injector:
    """Fix 3：在 COSeg 的 lin1[0] 之前把 δ_corr 加进前景通道。

    lin1 的输入是 [N_q, 2, 100]（bg 在前、fg 在后），inject 只改 channel 1。
    """

    def __init__(self):
        self.delta = None
        self.gamma = 1.0
        self.handle = None
        self.applied = 0

    def attach(self, model):
        def fn(module, args):
            if self.delta is None or self.gamma == 0:
                return None
            x = args[0]
            if x.dim() != 3 or x.shape[1] != 2:
                return None
            d = self.delta
            if d.shape[0] != x.shape[0]:
                return None
            x = x.clone()
            x[:, 1, :] = x[:, 1, :] + self.gamma * d.unsqueeze(1)
            self.applied += 1
            return (x,) + tuple(args[1:])
        self.handle = model.lin1[0].register_forward_pre_hook(fn)

    def detach(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
