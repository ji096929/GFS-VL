# -*- coding: utf-8 -*-
"""诊断 A（修正版）：验证集实测尺度 + 三组互不混淆的扫描。

修正了两个数学错误：
  1. base logits 含 base_cross_head（128d），不能只比 seg_head
  2. ||W||/sqrt(d) 方向错：novel 输入是 LN 拼接，模长 ~sqrt(192) 比 dec0 大

三组扫描：
  S1 加性: logits[:, :12] -= delta,       delta in [-3, 3] step 0.25   (25 点)
  S2 乘性: logits[:, 12:] *= alpha,       alpha in [0.6, 2.0] step 0.1 (15 点)
  S3 Novel 内部 tau-norm: logits[:,12:] / ||W_novel_c||^gamma, gamma in {0.5,1.0}
     并配合 alpha 一起扫（把「Base 压 Novel」和「Novel 内部长尾」拆开）
"""
import io
import shutil

P = "/data01/jh/code/GFS-VL/pointcept/engines/test.py"
BAK = P + ".bak_audit"

src = io.open(P, encoding="utf-8").read()
# 先移除旧版（如果存在）
if "GFS_HEAD_AUDIT" in src:
    print("检测到旧版 audit，先回滚")
    shutil.copy2(BAK if BAK else P + ".bak_audit", P)
    src = io.open(P, encoding="utf-8").read()
    if "GFS_HEAD_AUDIT" in src:
        raise SystemExit("回滚失败，请手动检查")

old1 = (
    "        save_path = os.path.join(self.cfg.save_path, \"result\")\n"
    "        make_dirs(save_path)\n"
    "        comm.synchronize()\n"
    "        record = {}\n"
)
new1 = (
    "        save_path = os.path.join(self.cfg.save_path, \"result\")\n"
    "        make_dirs(save_path)\n"
    "        comm.synchronize()\n"
    "        record = {}\n"
    "        # ---- 诊断 A：尺度实测 + 三组扫描（GFS_HEAD_AUDIT=1）----\n"
    "        _audit = bool(os.environ.get(\"GFS_HEAD_AUDIT\"))\n"
    "        _NB = self.cfg.data.num_bases\n"
    "        _NN = self.cfg.data.num_base_novels - _NB\n"
    "        _NC = _NB + _NN\n"
    "        _D = [round(-3.0 + 0.25 * i, 2) for i in range(25)]\n"
    "        _A = [round(0.6 + 0.1 * i, 2) for i in range(15)]\n"
    "        _G = [0.5, 1.0]\n"
    "        _cm_d = {d: np.zeros((_NC, _NC), dtype=np.int64) for d in _D} if _audit else None\n"
    "        _cm_a = {a: np.zeros((_NC, _NC), dtype=np.int64) for a in _A} if _audit else None\n"
    "        _cm_g = {(g, a): np.zeros((_NC, _NC), dtype=np.int64)\n"
    "                 for g in _G for a in _A} if _audit else None\n"
    "        _wn_novel = None\n"
    "        _scale_acc = None\n"
    "        if _audit:\n"
    "            _m = self.model.module if comm.get_world_size() > 1 else self.model\n"
    "            _wn_novel = _m.novel_seg_head.weight.detach().norm(dim=1).cpu().numpy()\n"
    "            _wbc = getattr(_m, \"base_cross_head\", None)\n"
    "            logger.info(\"[audit-w] ||seg_head|| per-class: %s\"\n"
    "                        % np.round(_m.seg_head.weight.detach().norm(dim=1).cpu().numpy(), 3).tolist())\n"
    "            logger.info(\"[audit-w] ||novel_head|| per-class: %s\"\n"
    "                        % np.round(_wn_novel, 3).tolist())\n"
    "            logger.info(\"[audit-w] ||base_cross_head|| mean %.4f\"\n"
    "                        % (float(_wbc.weight.detach().norm(dim=1).mean()) if _wbc is not None else -1.0))\n"
    "            _scale_acc = dict(nb=0, nb_sum=0.0, nn_sum=0.0, lb_sum=0.0, lb_sq=0.0,\n"
    "                              ln_sum=0.0, ln_sq=0.0, lmax_b=0.0, lmax_n=0.0)\n"
)
assert src.count(old1) == 1, "init 锚点 %d" % src.count(old1)
src = src.replace(old1, new1)

old2 = ("            pred = pred.max(1)[1].data.cpu().numpy()\n"
        "            if \"origin_segment\" in data_dict.keys():\n")
new2 = (
    "            if _audit:\n"
    "                _lg = pred\n"
    "                _vld = (segment >= 0) & (segment < _NC)\n"
    "                if _vld.any():\n"
    "                    _gv = segment[_vld].astype(np.int64)\n"
    "                    _lv = _lg[_vld]\n"
    "                    _lb = _lv[:, :_NB].detach()\n"
    "                    _ln = _lv[:, _NB:].detach()\n"
    "                    _s = _scale_acc\n"
    "                    _s[\"nb\"] += int(_lv.shape[0])\n"
    "                    _s[\"lb_sum\"] += float(_lb.sum())\n"
    "                    _s[\"lb_sq\"] += float((_lb * _lb).sum())\n"
    "                    _s[\"ln_sum\"] += float(_ln.sum())\n"
    "                    _s[\"ln_sq\"] += float((_ln * _ln).sum())\n"
    "                    _s[\"lmax_b\"] = max(_s[\"lmax_b\"], float(_lb.max()))\n"
    "                    _s[\"lmax_n\"] = max(_s[\"lmax_n\"], float(_ln.max()))\n"
    "                    # S1 加性\n"
    "                    for _d in _D:\n"
    "                        _l = _lv.clone(); _l[:, :_NB] -= _d\n"
    "                        _pv = _l.argmax(1).cpu().numpy().astype(np.int64)\n"
    "                        _cm_d[_d] += np.bincount(_gv * _NC + _pv, minlength=_NC * _NC).reshape(_NC, _NC)\n"
    "                    # S2 乘性\n"
    "                    for _a in _A:\n"
    "                        _l = _lv.clone(); _l[:, _NB:] *= _a\n"
    "                        _pv = _l.argmax(1).cpu().numpy().astype(np.int64)\n"
    "                        _cm_a[_a] += np.bincount(_gv * _NC + _pv, minlength=_NC * _NC).reshape(_NC, _NC)\n"
    "                    # S3 Novel 内部 tau-norm + alpha\n"
    "                    _wt = torch.from_numpy(_wn_novel).to(_lv.device).clamp(min=1e-9)\n"
    "                    for _g2 in _G:\n"
    "                        _base = _lv.clone()\n"
    "                        _base[:, _NB:] = _base[:, _NB:] / (_wt ** _g2).unsqueeze(0)\n"
    "                        for _a in _A:\n"
    "                            _l = _base.clone(); _l[:, _NB:] *= _a\n"
    "                            _pv = _l.argmax(1).cpu().numpy().astype(np.int64)\n"
    "                            _cm_g[(_g2, _a)] += np.bincount(\n"
    "                                _gv * _NC + _pv, minlength=_NC * _NC).reshape(_NC, _NC)\n"
    "                del pred\n"
    + old2
)
assert src.count(old2) == 1, "argmax 锚点 %d" % src.count(old2)
src = src.replace(old2, new2)

old3 = (
    "            _split = getattr(self.test_loader.dataset, \"split\", \"val\")\n"
    "            _save = os.path.join(save_path, f\"{_split}.pth\")\n"
)
new3 = (
    "            _split = getattr(self.test_loader.dataset, \"split\", \"val\")\n"
    "            if _audit and _cm_d is not None:\n"
    "                _out = {(\"d%.2f\" % k): v for k, v in _cm_d.items()}\n"
    "                _out.update({(\"a%.1f\" % k): v for k, v in _cm_a.items()})\n"
    "                _out.update({(\"g%.1f_a%.1f\" % k): v for k, v in _cm_g.items()})\n"
    "                _out[\"wn_novel\"] = _wn_novel\n"
    "                np.savez_compressed(os.path.join(save_path, \"head_audit.npz\"), **_out)\n"
    "                _s = _scale_acc; _n = max(_s[\"nb\"], 1)\n"
    "                _mb, _sqb = _s[\"lb_sum\"] / _n, _s[\"lb_sq\"] / _n\n"
    "                _mn, _sqn = _s[\"ln_sum\"] / _n, _s[\"ln_sq\"] / _n\n"
    "                logger.info(\"[audit-scale] logits_base: mean %.4f std %.4f max %.3f\"\n"
    "                            % (_mb, (_sqb - _mb * _mb) ** 0.5, _s[\"lmax_b\"]))\n"
    "                logger.info(\"[audit-scale] logits_novel: mean %.4f std %.4f max %.3f\"\n"
    "                            % (_mn, (_sqn - _mn * _mn) ** 0.5, _s[\"lmax_n\"]))\n"
    "                def _hm(_c):\n"
    "                    _c = _c.astype(np.float64)\n"
    "                    _iou = np.diag(_c) / np.maximum(_c.sum(0) + _c.sum(1) - np.diag(_c), 1)\n"
    "                    _b, _nn2 = _iou[:_NB].mean(), _iou[_NB:].mean()\n"
    "                    return _b, _nn2, 2 * _b * _nn2 / max(_b + _nn2, 1e-9)\n"
    "                logger.info(\"[audit-S1] delta  Base   Novel   HM\")\n"
    "                for _d in _D:\n"
    "                    _b, _nn2, _h = _hm(_cm_d[_d])\n"
    "                    logger.info(\"[audit-S1] %+5.2f  %.4f  %.4f  %.4f\" % (_d, _b, _nn2, _h))\n"
    "                logger.info(\"[audit-S2] alpha  Base   Novel   HM\")\n"
    "                for _a in _A:\n"
    "                    _b, _nn2, _h = _hm(_cm_a[_a])\n"
    "                    logger.info(\"[audit-S2] %.1f  %.4f  %.4f  %.4f\" % (_a, _b, _nn2, _h))\n"
    "                logger.info(\"[audit-S3] gamma alpha  Base   Novel   HM\")\n"
    "                for _g2 in _G:\n"
    "                    for _a in _A:\n"
    "                        _b, _nn2, _h = _hm(_cm_g[(_g2, _a)])\n"
    "                        logger.info(\"[audit-S3] %.1f  %.1f  %.4f  %.4f  %.4f\"\n"
    "                                    % (_g2, _a, _b, _nn2, _h))\n"
    + old3
)
assert src.count(old3) == 1, "save 锚点 %d" % src.count(old3)
src = src.replace(old3, new3)

shutil.copy2(P, BAK)
io.open(P, "w", encoding="utf-8").write(src)
print("已写入诊断 A 修正版（备份 %s）" % BAK)
