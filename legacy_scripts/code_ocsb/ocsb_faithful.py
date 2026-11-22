"""3D-OCSB 的忠实实现：导师文档 Method V1 式 (1)-(19)。

只保留三处最小修正，其余严格按文档：

  C1  式(1)  W_phi 单位阵初始化 + ||W-I||_F^2 正则，保证第 0 步与冻结的
            COSeg 特征空间完全一致，同时第 0 步反向传播梯度非零。
  C2  式(8)  外层乘回温度 tau（原文漏乘会让 eta 量纲放大 1/tau 倍）；
            N_c 扩为 D_S ∪ P_base ∪ D_U 三部分。
  C3  式(15) 基准项改用 COSeg 原生预测头的前景-背景 margin，
            使 M_c 为空时严格退化为原始 COSeg（误差 < 1e-6）。

所有量都在 1-way（2 类：背景 / 目标类）下定义，N-way 只是把标量换成向量。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ helpers

def sp_mean_norm(z, seg, n_reg):
    """式(1)：超点特征 = Norm(mean_{i in R} z_i)，z 已逐点归一化。"""
    out = torch.zeros(n_reg, z.shape[1], dtype=z.dtype, device=z.device)
    out.index_add_(0, seg, z)
    cnt = torch.bincount(seg, minlength=n_reg).clamp_min(1).to(z.dtype)
    return F.normalize(out / cnt[:, None], dim=-1)


def proto_norm(z, mask):
    """式(2)：支持原型 = Norm(支持前景点特征均值)。"""
    if mask.sum() == 0:
        return torch.zeros(z.shape[1], dtype=z.dtype, device=z.device)
    return F.normalize(z[mask].mean(0), dim=0)


def logmeanexp(x, dim):
    """log(mean(exp(x)))，数值稳定。"""
    n = x.shape[dim]
    return torch.logsumexp(x, dim=dim) - math.log(max(n, 1))


def sp_soft_labels(y, seg, n_reg, n_cls=2):
    """超点级软标签 y_R(c) = mean_{i in R} 1[Y_i = c]。"""
    out = torch.zeros(n_reg, n_cls, dtype=torch.float32, device=y.device)
    onehot = F.one_hot(y.long(), n_cls).to(torch.float32)
    out.index_add_(0, seg, onehot)
    cnt = torch.bincount(seg, minlength=n_reg).clamp_min(1).to(torch.float32)
    return out / cnt[:, None]


def fuse_query_score(s_s, M_feat, M_w, z_q, tau, gamma):
    """式(15)：log((exp(s_s) + gamma*sum_R w_R exp(sim)) / (1+gamma*sum w_R))。

    M 为空时严格返回 s_s（这是 4.4 检查项 3 的数学基础）。
    """
    active = M_w > 0
    if not bool(active.any()):
        return s_s
    mf = M_feat[active]
    mw = M_w[active]
    sim = (mf @ z_q.t()) / tau                     # [K, Nq]
    agg = torch.logsumexp(torch.log(mw).unsqueeze(1) + sim, dim=0)
    num = torch.logaddexp(s_s, math.log(gamma) + agg)
    den = torch.log1p(gamma * mw.sum())
    return num - den


def b_only_region_prob(s0, s0_bg, z_R, B_idx, q_reg, tau):
    """式(6) 的 B-only 多桥版本，用于 Step 2 的超点级诊断。

    s_B(C) = log((exp(s0(C)) + sum_B q_B exp(sim_BC/tau)) / (1 + sum_B q_B))
    """
    if B_idx.numel() == 0:
        sB = s0
    else:
        qB = q_reg[B_idx]
        sim = (z_R[B_idx] @ z_R.t()) / tau          # [M, R]
        agg = torch.logsumexp(torch.log(qB).unsqueeze(1) + sim, dim=0)
        sB = torch.logaddexp(s0, agg) - torch.log1p(qB.sum())
    return F.softmax(torch.stack([s0_bg, sB], 0), dim=0)


# --------------------------------------------------------------- main module

class OCSB(nn.Module):
    """OASB + CSR + 类别知识扩展 + 查询点级解码，式 (1)-(15)。

    参数
      dim      主干特征维度（COSeg 为 192）
      tau      式(5) 的温度，默认 0.1（与 diag_bc 的 margin 空间一致）
      tau_b    式(3) 的桥梁可信阈值，默认 0.6
      kappa    式(10) 的支持锚点保留项
      gamma0   式(15) 的标度系数初值
      beta     式(16) 的遗漏前景权重增量
    """

    def __init__(self, dim, tau=0.1, tau_b=0.6, kappa=0.1, gamma0=0.2,
                 beta=1.0, eps=1e-6, fusion_mode='A', gamma_fuse=5.0):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.tau_b = tau_b
        self.kappa = kappa
        self.beta = beta
        self.eps = eps
        # 式(15) 的修正：余弦空间算增量，再叠加到 margin
        #   A = 双边增量（严格对应式 11）
        #   B = 单向正增量（只向上修正漏检点）
        self.fusion_mode = fusion_mode
        self.gamma_fuse = float(gamma_fuse)
        # C1：单位阵初始化
        self.proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.proj.weight)

    @property
    def gamma(self):
        return self.gamma_fuse

    def project(self, h):
        """z = Norm(W_phi h)。"""
        return F.normalize(self.proj(h), dim=-1)

    def weight_reg(self):
        """C1 的正则项 ||W_phi - I||_F^2。"""
        w = self.proj.weight
        eye = torch.eye(self.dim, device=w.device, dtype=w.dtype)
        return ((w - eye) ** 2).sum()

    # ------------------------------------------------------------ forward

    def forward(self, ep, enable_oasb=True, enable_csr=True,
                return_negative=True):
        """跑一个 episode 的前向。

        ep 需要以下字段（全部在 GPU、fp32）：
          s_h [Ns,d]  s_y [Ns]  s_seg [Ns]
          u_h [Nu,d]  u_seg [Nu]  u_p0 [Nu]  u_y [Nu]
          q_h [Nq,d]  q_margin [Nq]  q_y [Nq]
          base_protos [P,d]（可为 None）
          n_neg_u    D_U 的 q_c 上界（默认 0.05）

        返回 dict，含式(15) 的 ell 及全部中间量。
        """
        tau = self.tau
        dev = ep['s_h'].device

        # ---- 共享投影 ----
        z_s = self.project(ep['s_h'])
        z_u = self.project(ep['u_h'])
        z_q = self.project(ep['q_h'])

        # ---- 式(1) 超点特征 ----
        u_seg = ep['u_seg'].long()
        n_reg = int(u_seg.max().item()) + 1
        z_R = sp_mean_norm(z_u, u_seg, n_reg)

        # ---- 式(2) 支持原型（前景 + 背景）----
        p_fg = proto_norm(z_s, ep['s_y'] > 0.5)
        p_bg = proto_norm(z_s, ep['s_y'] <= 0.5)

        # ---- 超点级初始概率 q_c(R) ----
        q_reg = torch.zeros(n_reg, dtype=torch.float32, device=dev)
        q_reg.index_add_(0, u_seg, ep['u_p0'].to(torch.float32))
        cnt = torch.bincount(u_seg, minlength=n_reg).clamp_min(1).to(torch.float32)
        q_reg = (q_reg / cnt).clamp(1e-4, 1.0 - 1e-4)

        # ---- 式(3) B_c / 式(4) C_c ----
        # 1-way 下 c_hat_0 == c 等价于 q_c >= 0.5
        B_idx = ((q_reg >= self.tau_b)).nonzero().flatten()
        C_idx = ((q_reg < 0.5)).nonzero().flatten()

        # ---- 式(5) 直接关联 s_c^0 ----
        s0 = (z_R @ p_fg) / tau                      # [R]
        s0_bg = (z_R @ p_bg) / tau                   # [R]

        out = {
            'q_reg': q_reg, 'z_R': z_R, 's0': s0, 's0_bg': s0_bg,
            'B_idx': B_idx, 'C_idx': C_idx, 'n_reg': n_reg,
            'p_fg': p_fg, 'p_bg': p_bg,
        }

        # Step 3 之前（只做 OASB 但关掉 CSR）仍要走 B/C 构造
        if not enable_oasb or B_idx.numel() == 0:
            # 式(3) 为空：预测必须退化为纯支持原型
            s_bar = torch.stack([s0_bg, s0], 0)       # [2, R]
            out.update({
                's_bar': s_bar,
                'P_bar': F.softmax(s_bar, dim=0),
                'ell': ep['q_margin'],
                'w_C': torch.zeros(C_idx.numel(), device=dev),
                'w_B': torch.zeros(0, device=dev),
                'V': torch.zeros(0, C_idx.numel(), device=dev),
                'delta': torch.zeros(0, n_reg, device=dev),
                'eta': torch.zeros(0, device=dev),
                'M_feat': torch.zeros(0, self.dim, device=dev),
                'M_w': torch.zeros(0, device=dev),
            })
            return out

        # ---- 式(6) 桥接关联 s_c^B ----
        # 只对 C 候选算，B 自身不参与（文档：从该区域的桥梁集合中排除自身）
        z_B = z_R[B_idx]                              # [M, d]
        q_B = q_reg[B_idx]                            # [M]
        sim_BC = (z_B @ z_R[C_idx].t()) / tau         # [M, Rc]
        log_qB = torch.log(q_B)
        sB = torch.logaddexp(s0[C_idx].unsqueeze(0),
                             log_qB[:, None] + sim_BC) - torch.log1p(q_B)[:, None]

        # ---- 式(7) 预测变化 ----
        delta_pos = sB - s0[C_idx].unsqueeze(0)       # [M, Rc]

        # ---- C2：式(8) 负对照风险 ----
        eta = torch.zeros(B_idx.numel(), device=dev)
        delta_neg = None
        if enable_csr and return_negative:
            neg_feat, neg_src = self._negative_set(ep, z_s, z_R, q_reg, s0_bg)
            if neg_feat.shape[0] > 0:
                s0_neg = (neg_feat @ p_fg) / tau                      # [P]
                sB_neg = torch.logaddexp(
                    s0_neg.unsqueeze(0),
                    log_qB[:, None] + (z_B @ neg_feat.t()) / tau
                ) - torch.log1p(q_B)[:, None]                          # [M, P]
                delta_neg = sB_neg - s0_neg.unsqueeze(0)
                eta = tau * logmeanexp(F.relu(delta_neg) / tau, dim=1)
                out['neg_src'] = neg_src
        out['delta_neg'] = delta_neg

        # ---- 式(9) 净贡献 ----
        V = delta_pos - eta.unsqueeze(1)               # [M, Rc]

        # ---- 式(10) 可微桥梁权重 ----
        pos = F.relu(V)
        denom = self.kappa + pos.sum(0)                # [Rc]
        alpha = pos / denom.unsqueeze(0)               # [M, Rc]
        alpha0 = self.kappa / denom                    # [Rc]

        # ---- 式(11) 聚合区域得分 ----
        s_bar_fg = s0[C_idx] + (alpha * delta_pos).sum(0)
        s_bar = torch.stack([s0_bg[C_idx], s_bar_fg], 0)   # [2, Rc]
        P_bar = F.softmax(s_bar, dim=0)
        out['s_bar_c'] = s_bar_fg

        # ---- 式(12) 遗漏超点恢复权重 ----
        g = (alpha * F.relu(V)).sum(0)                     # [Rc]
        c0_is_c = (q_reg[C_idx] >= 0.5).float()
        argmax_is_c = (P_bar.argmax(0) == 1).float()
        w_C = (1.0 - c0_is_c) * argmax_is_c * P_bar[1] * (g / (1.0 + g))

        # ---- 式(13) 桥梁贡献反哺 ----
        w_C_sum = w_C.sum()
        w_B = q_B * (alpha * w_C.unsqueeze(0)).sum(1) / (self.eps + w_C_sum)

        # ---- 式(14) 补充参考集合 ----
        M_feat = torch.cat([z_B, z_R[C_idx]], 0)           # [M+Rc, d]
        M_w = torch.cat([w_B, w_C], 0)

        # ---- C3：式(15) 修正版 ----
        # 先在统一余弦空间算记忆库相对支持原型的增量，再乘 τ 回到 margin 量纲
        s0_q = (z_q @ p_fg) / tau                  # [Nq] 支持原型对查询点
        delta_mem = self.memory_delta(s0_q, M_feat, M_w, z_q)
        if self.fusion_mode == 'B':
            delta_mem = F.relu(delta_mem)
        ell = ep['q_margin'] + self.gamma_fuse * delta_mem

        out.update({
            's_bar': s_bar, 'P_bar': P_bar, 'ell': ell,
            'w_C': w_C, 'w_B': w_B, 'V': V, 'delta': delta_pos,
            'eta': eta, 'alpha': alpha, 'alpha0': alpha0,
            'M_feat': M_feat, 'M_w': M_w, 'q_B': q_B, 'z_B': z_B,
            's0_q': s0_q, 'delta_mem': delta_mem,
        })
        return out

    # ------------------------------------------------------------ pieces

    def _negative_set(self, ep, z_s, z_R, q_reg, s0_bg):
        """C2：N_c = D_S ∪ P_base ∪ D_U。返回特征矩阵与来源标签。"""
        feats, src = [], []

        # (a) 支持集纯背景超点
        if 's_seg' in ep and ep['s_seg'] is not None:
            s_seg = ep['s_seg'].long()
            n_s = int(s_seg.max().item()) + 1
            fg_frac = torch.zeros(n_s, dtype=torch.float32, device=z_s.device)
            fg_frac.index_add_(0, s_seg, (ep['s_y'] > 0.5).to(torch.float32))
            cnt = torch.bincount(s_seg, minlength=n_s).clamp_min(1).to(torch.float32)
            fg_frac = fg_frac / cnt
            z_sR = sp_mean_norm(z_s, s_seg, n_s)
            pure_bg = fg_frac <= 0.0
            if pure_bg.any():
                feats.append(z_sR[pure_bg])
                src.append(torch.zeros(int(pure_bg.sum()), dtype=torch.int64,
                                       device=z_s.device))

        # (b) 基类原型（调用方已屏蔽当前目标类）
        if ep.get('base_protos') is not None and ep['base_protos'].numel():
            bp = F.normalize(ep['base_protos'].to(z_s.dtype), dim=-1)
            feats.append(bp)
            src.append(torch.ones(bp.shape[0], dtype=torch.int64, device=z_s.device))

        # (c) U 中高置信背景超点
        bg_u = q_reg < 0.05
        if bg_u.any():
            feats.append(z_R[bg_u])
            src.append(torch.full((int(bg_u.sum()),), 2, dtype=torch.int64,
                                  device=z_s.device))

        if not feats:
            return torch.zeros(0, self.dim, device=z_s.device), None
        return torch.cat(feats, 0), torch.cat(src, 0)

    def memory_delta(self, s0_q, M_feat, M_w, z_q):
        """式(15) 修正：记忆库相对支持原型的余弦增量，乘 τ 后与 margin 同量纲。

        数学性质：
          M 为空或 w=0            -> 0
          z_R·z_q == p_fg·z_q     -> 0（聚合退化为 s0_q）
          z_R 比 p_fg 更匹配 z_q   -> >0
        """
        active = M_w > 0
        if not bool(active.any()):
            return torch.zeros_like(s0_q)
        mf = M_feat[active]
        mw = M_w[active]
        sim = (mf @ z_q.t()) / self.tau                     # [K, Nq]
        agg = torch.logsumexp(torch.log(mw).unsqueeze(1) + sim, dim=0)
        s_mem = torch.logaddexp(s0_q, agg) - torch.log1p(mw.sum())
        return self.tau * (s_mem - s0_q)

    def _query_score(self, s_s, M_feat, M_w, z_q):
        """兼容旧接口：返回 margin + γ·delta_mem。"""
        s0_q = s_s  # 旧调用方把 margin 当基准，这里仅作退化占位
        d = self.memory_delta(s0_q, M_feat, M_w, z_q)
        return s_s + self.gamma_fuse * d


# ------------------------------------------------------------------ losses

def ocsb_losses(out, ep, lam_o=1.0, lam_s=1.0, w_reg=0.0,
                model=None, use_select=True, w_pos=1.0, tau_s=1.0):
    """式(16)-(19)。返回各分量，便于自检时逐项 backward。

    L_seg    查询点级 BCE（等价于 2 类 CE，背景 logit 固定为 0）
    L_omit   超点级软标签 CE，遗漏前景加权 1+beta
    L_select softplus 排名损失，w_pos 放大正类（真实遗漏）一侧，
             tau_s 控制 softplus 的温度，避免负类把 v 整体拉低
    """
    dev = ep['q_h'].device
    parts = {}

    # ---- 式(18) L_seg ----
    ell = out['ell']
    y_q = ep['q_y'].to(ell.dtype)
    parts['seg'] = F.binary_cross_entropy_with_logits(ell, y_q)

    # ---- 式(16) L_omit ----
    loss_omit = torch.zeros((), device=dev)
    if 'u_y' in ep and ep['u_y'] is not None and out['C_idx'].numel() > 0:
        y_R = sp_soft_labels(ep['u_y'], ep['u_seg'].long(), out['n_reg'], 2)
        c_star = y_R.argmax(1)
        c0 = (out['q_reg'] >= 0.5).long()
        omega = 1.0 + model.beta * ((c_star != 0) & (c0 != c_star)).to(torch.float32)
        logP = F.log_softmax(out['s_bar'], dim=0)          # [2, Rc] 或 [2, R]
        if logP.shape[1] == out['n_reg']:                 # B 为空时的全超点版本
            logP = logP[:, out['C_idx']]
        ce = -(y_R[out['C_idx']] * logP.t()).sum(1)
        loss_omit = (omega[out['C_idx']] * ce).mean()
    parts['omit'] = loss_omit

    # ---- 式(17) L_select ----
    loss_sel = torch.zeros((), device=dev)
    if use_select and out['V'].numel() and 'u_y' in ep and ep['u_y'] is not None:
        c_idx = out['C_idx']
        y_R = sp_soft_labels(ep['u_y'], ep['u_seg'].long(), out['n_reg'], 2)
        is_fg_c = y_R[c_idx, 1] >= 0.5
        is_missed = (~(out['q_reg'][c_idx] >= 0.5)) & is_fg_c   # P_c：初始漏检的真前景
        local = torch.arange(c_idx.numel(), device=dev)
        pos = local[is_missed]
        neg = local[~is_fg_c]                                   # D_c：真负类
        v = out['V'].max(0).values                              # [Rc] 未截断
        terms = []
        if pos.numel():
            terms.append(w_pos * F.softplus(-v[pos] / tau_s).mean())
        if neg.numel():
            terms.append(F.softplus(v[neg] / tau_s).mean())
        if terms:
            loss_sel = sum(terms)
    parts['select'] = loss_sel

    total = parts['seg'] + lam_o * parts['omit'] + lam_s * parts['select']
    if w_reg > 0 and model is not None:
        total = total + w_reg * model.weight_reg()
    parts['total'] = total
    return parts


# ------------------------------------------------------------ Step 2 helper

def direct_expansion(out, ep, model, topk=None):
    """Step 2：只用高置信 B 直接扩展原型，无 OASB / 无 CSR / 不恢复 C。

    与主路径共用修正后的式(15)：余弦空间算增量，再叠加到 margin。
    """
    B_idx = out['B_idx']
    if B_idx.numel() == 0:
        return ep['q_margin']
    mf = out['z_R'][B_idx]
    mw = out['q_reg'][B_idx].clone()
    if topk is not None and mw.numel() > topk:
        keep = mw.topk(topk).indices
        mf, mw = mf[keep], mw[keep]
    zq = model.project(ep['q_h'])
    s0_q = (zq @ out['p_fg']) / model.tau
    d = model.memory_delta(s0_q, mf, mw, zq)
    if model.fusion_mode == 'B':
        d = F.relu(d)
    return ep['q_margin'] + model.gamma_fuse * d
