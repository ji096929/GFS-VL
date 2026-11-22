# -*- coding: utf-8 -*-
# ① L_cross 与主 loss 共用加权口径
p = '/data01/jh/code/GFS-VL/pointcept/engines/train.py'
s = open(p, encoding='utf-8').read()

old = '''            if \"pl_weight\" in input_dict:
                ce = torch.nn.functional.cross_entropy(
                    seg_logits, new_target, reduction=\"none\", ignore_index=-1)
                w = input_dict[\"pl_weight\"].float()
                valid = new_target != -1
                loss = (ce * w)[valid].sum() / (w[valid].sum() + 1e-6)
                if os.environ.get(\"GFS_DEBUG_W\") and not getattr(self, \"_w_dbg\", False):
                    self._w_dbg = True
                    pw = w[valid]
                    base = (new_target >= 0) & (new_target < 12)
                    self.logger.info(
                        \"[pl_weight] n=%d mean=%.3f min=%.3f max=%.3f | \"
                        \"base-GT w min=%.3f max=%.3f\"
                        % (int(valid.sum()), float(pw.mean()), float(pw.min()),
                           float(pw.max()),
                           float(w[base].min()) if base.any() else -1.0,
                           float(w[base].max()) if base.any() else -1.0))
            else:
                loss = criteria(seg_logits, new_target)
            if decorr_loss is not None:
                loss = loss + decorr_loss
            if cross_logits is not None:
                _lam = float(getattr(self.cfg, \"lam_cross\", 0.3))
                loss = loss + _lam * criteria(cross_logits, new_target)'''

new = '''            _pw = (
                input_dict[\"pl_weight\"].float()
                if \"pl_weight\" in input_dict else None
            )

            def _weighted_ce(logits):
                # 主 loss 与辅助 loss 共用同一口径，避免软权重模式下静默不一致
                if _pw is None:
                    return criteria(logits, new_target)
                ce = torch.nn.functional.cross_entropy(
                    logits, new_target, reduction=\"none\", ignore_index=-1)
                valid = new_target != -1
                return (ce * _pw)[valid].sum() / (_pw[valid].sum() + 1e-6)

            loss = _weighted_ce(seg_logits)
            if _pw is not None and os.environ.get(\"GFS_DEBUG_W\") and not getattr(self, \"_w_dbg\", False):
                self._w_dbg = True
                valid = new_target != -1
                pw = _pw[valid]
                base = (new_target >= 0) & (new_target < 12)
                self.logger.info(
                    \"[pl_weight] n=%d mean=%.3f min=%.3f max=%.3f | \"
                    \"base-GT w min=%.3f max=%.3f\"
                    % (int(valid.sum()), float(pw.mean()), float(pw.min()),
                       float(pw.max()),
                       float(_pw[base].min()) if base.any() else -1.0,
                       float(_pw[base].max()) if base.any() else -1.0))
            if decorr_loss is not None:
                loss = loss + decorr_loss
            if cross_logits is not None:
                _lam = float(getattr(self.cfg, \"lam_cross\", 0.3))
                loss = loss + _lam * _weighted_ce(cross_logits)'''

assert s.count(old) == 1, 'a1=%d' % s.count(old)
s = s.replace(old, new)

# ③ resume 打 missing/unexpected
old3 = '''                        self.model.load_state_dict(_ck[\"state_dict\"], strict=False)'''
new3 = '''                        _miss, _unexp = self.model.load_state_dict(
                            _ck[\"state_dict\"], strict=False)
                        self.logger.info(
                            \"=> Resume: missing=%d unexpected=%d\"
                            % (len(_miss), len(_unexp)))'''
assert s.count(old3) == 1, 'a3=%d' % s.count(old3)
s = s.replace(old3, new3)
open(p, 'w', encoding='utf-8').write(s)
print('train.py fixed (L_cross weighted + resume log)')

# ② 硬编码切片 -> backbone_out_channels
p2 = '/data01/jh/code/GFS-VL/pointcept/models/default.py'
m = open(p2, encoding='utf-8').read()

old2 = '''        self.cross_layers = tuple(cross_layers)'''
new2 = '''        self.cross_layers = tuple(cross_layers)
        self.backbone_out_channels = int(backbone_out_channels)'''
assert m.count(old2) == 1, 'a2=%d' % m.count(old2)
m = m.replace(old2, new2)

old4 = '''                        _cn = torch.nn.functional.linear(
                            _cf, self.novel_seg_head.weight[:, 64:]
                        )'''
new4 = '''                        _boc = self.backbone_out_channels
                        assert _cf.shape[1] == (
                            self.novel_seg_head.weight.shape[1] - _boc
                        ), \"cross feat dim mismatch: %d vs %d\" % (
                            _cf.shape[1],
                            self.novel_seg_head.weight.shape[1] - _boc)
                        _cn = torch.nn.functional.linear(
                            _cf, self.novel_seg_head.weight[:, _boc:]
                        )'''
assert m.count(old4) == 1, 'a4=%d' % m.count(old4)
m = m.replace(old4, new4)
open(p2, 'w', encoding='utf-8').write(m)
print('default.py fixed (dynamic slice + assert)')
