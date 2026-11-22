p = '/data01/jh/code/GFS-VL/pointcept/engines/train.py'
s = open(p, encoding='utf-8').read()

# 1) __init__ 里读配置（找到 GFS_VL_Trainer 的初始化）
o1 = '''            _pw = (
                input_dict[\"pl_weight\"].float()
                if \"pl_weight\" in input_dict else None
            )'''
n1 = '''            _pw = (
                input_dict[\"pl_weight\"].float()
                if \"pl_weight\" in input_dict else None
            )

            # ---- 训练期 Logit Adjustment (Menon et al., ICLR 2021) ----
            _use_la = bool(getattr(self.cfg, \"use_la\", False))
            if _use_la and not hasattr(self, \"_la_bias\"):
                _pi_path = getattr(self.cfg, \"la_prior_path\",
                                   \"/data01/jh/data/pl_class_prior_full.npy\")
                _pi = torch.tensor(np.load(_pi_path), dtype=torch.float32,
                                   device=seg_logits.device)
                _pi = _pi / _pi.sum()
                _tau = float(getattr(self.cfg, \"la_tau\", 0.5))
                self._la_bias = _tau * torch.log(_pi).unsqueeze(0)
                self.logger.info(\"[LA] tau=%.2f pi range %.6f~%.6f\"
                                 % (_tau, float(_pi.min()), float(_pi.max())))
            if _use_la:
                seg_logits = seg_logits + self._la_bias.to(seg_logits.dtype)'''
assert s.count(o1) == 1, 'a1=%d' % s.count(o1)
s = s.replace(o1, n1)

# 2) Lovasz 加在 _weighted_ce 之后
o2 = '''            if decorr_loss is not None:
                loss = loss + decorr_loss'''
n2 = '''            if getattr(self.cfg, \"use_lovasz\", False):
                from pointcept.models.losses import LovaszLoss
                _lv = LovaszLoss(mode=\"multiclass\", ignore_index=-1,
                                 loss_weight=float(getattr(self.cfg, \"lovasz_w\", 1.0)))
                _n = seg_logits.shape[0]
                _lo = _lv(seg_logits.view(_n, -1, 1, 1),
                          new_target.view(_n, 1, 1))
                if not getattr(self, \"_lv_logged\", False):
                    self._lv_logged = True
                    self.logger.info(\"[Lovasz] ce=%.4f lovasz=%.4f\" % (float(loss), float(_lo)))
                loss = loss + _lo
            if decorr_loss is not None:
                loss = loss + decorr_loss'''
assert s.count(o2) == 1, 'a2=%d' % s.count(o2)
s = s.replace(o2, n2)

open(p, 'w', encoding='utf-8').write(s)
print('lovasz + LA added')
