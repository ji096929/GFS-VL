# -*- coding: utf-8 -*-
"""给 one_regis_test 加分类头体检（GFS_HEAD_AUDIT=1）：
  1. 静态范数：seg_head / novel_seg_head 的每类 ||W_c|| 与 bias
  2. 加性偏置扫描：logits[:, :12] -= delta，17 个点
  3. 乘性范数均衡：logits / ||W_c||^gamma，gamma in {0.5, 1.0}

全部在同一个前向里完成，零额外前向开销。
"""
import io
import shutil

P = "/data01/jh/code/GFS-VL/pointcept/engines/test.py"
BAK = P + ".bak_audit"

src = io.open(P, encoding="utf-8").read()
if "GFS_HEAD_AUDIT" in src:
    print("已存在 GFS_HEAD_AUDIT，跳过")
    raise SystemExit(0)

# 1) 初始化
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
    "        # 分类头体检（GFS_HEAD_AUDIT=1）\n"
    "        _audit = bool(os.environ.get(\"GFS_HEAD_AUDIT\"))\n"
    "        _DELTAS = [round(-2.0 + 0.25 * i, 2) for i in range(17)]\n"
    "        _GAMMAS = [0.5, 1.0]\n"
    "        _NC = self.cfg.data.num_base_novels\n"
    "        _cm_d = {d: np.zeros((_NC, _NC), dtype=np.int64) for d in _DELTAS} if _audit else None\n"
    "        _cm_g = {g: np.zeros((_NC, _NC), dtype=np.int64) for g in _GAMMAS} if _audit else None\n"
    "        _wnorm = None\n"
    "        if _audit:\n"
    "            _m = self.model.module if comm.get_world_size() > 1 else self.model\n"
    "            _Wb = _m.seg_head.weight.detach()\n"
    "            _Wn = _m.novel_seg_head.weight.detach()\n"
    "            _wnorm = torch.cat([_Wb.norm(dim=1), _Wn.norm(dim=1)]).cpu().numpy()\n"
    "            _b = _m.seg_head.bias.detach().cpu().numpy() if _m.seg_head.bias is not None else None\n"
    "            logger.info(\"[audit] ||W_base|| mean %.4f  ||W_novel|| mean %.4f  ratio %.3f\"\n"
    "                        % (float(_Wb.norm(dim=1).mean()), float(_Wn.norm(dim=1).mean()),\n"
    "                           float(_Wn.norm(dim=1).mean() / max(float(_Wb.norm(dim=1).mean()), 1e-9))))\n"
    "            logger.info(\"[audit] ||W_base|| per-class: %s\" % np.round(_Wb.norm(dim=1).cpu().numpy(), 3).tolist())\n"
    "            logger.info(\"[audit] ||W_novel|| per-class: %s\" % np.round(_Wn.norm(dim=1).cpu().numpy(), 3).tolist())\n"
    "            if _b is not None:\n"
    "                logger.info(\"[audit] b_base: %s\" % np.round(_b, 4).tolist())\n"
)
assert src.count(old1) == 1, "初始化锚点 %d" % src.count(old1)
src = src.replace(old1, new1)

# 2) 在 argmax 前插入扫描（只对 logits 做，不改原 pred）
old2 = ("            pred = pred.max(1)[1].data.cpu().numpy()\n"
        "            if \"origin_segment\" in data_dict.keys():\n")
new2 = (
    "            if _audit:\n"
    "                _lg = pred  # (N, 57) logits，尚未 argmax\n"
    "                _vld = (segment >= 0) & (segment < _NC)\n"
    "                if _vld.any():\n"
    "                    _gv = segment[_vld].astype(np.int64)\n"
    "                    for _d in _DELTAS:\n"
    "                        _l = _lg.clone()\n"
    "                        _l[:, :self.cfg.data.num_bases] -= _d\n"
    "                        _pv = _l[_vld].argmax(1).data.cpu().numpy().astype(np.int64)\n"
    "                        _cm_d[_d] += np.bincount(\n"
    "                            _gv * _NC + _pv, minlength=_NC * _NC\n"
    "                        ).reshape(_NC, _NC)\n"
    "                    _wt = torch.from_numpy(_wnorm).to(_lg.device).clamp(min=1e-9)\n"
    "                    for _gm in _GAMMAS:\n"
    "                        _l = _lg / (_wt ** _gm).unsqueeze(0)\n"
    "                        _pv = _l[_vld].argmax(1).data.cpu().numpy().astype(np.int64)\n"
    "                        _cm_g[_gm] += np.bincount(\n"
    "                            _gv * _NC + _pv, minlength=_NC * _NC\n"
    "                        ).reshape(_NC, _NC)\n"
    + old2
)
assert src.count(old2) == 1, "argmax 锚点 %d" % src.count(old2)
src = src.replace(old2, new2)

# 3) 保存 + 打印
old3 = (
    "            _split = getattr(self.test_loader.dataset, \"split\", \"val\")\n"
    "            _save = os.path.join(save_path, f\"{_split}.pth\")\n"
)
new3 = (
    "            _split = getattr(self.test_loader.dataset, \"split\", \"val\")\n"
    "            if _audit and _cm_d is not None:\n"
    "                _out = {(\"d%.2f\" % d): v for d, v in _cm_d.items()}\n"
    "                _out.update({(\"g%.1f\" % g): v for g, v in _cm_g.items()})\n"
    "                _out[\"wnorm\"] = _wnorm\n"
    "                _ap = os.path.join(save_path, \"head_audit.npz\")\n"
    "                np.savez_compressed(_ap, **_out)\n"
    "                logger.info(\"=> head audit saved: %s\" % _ap)\n"
    "                logger.info(\"[audit-delta] delta  Base   Novel   HM\")\n"
    "                for _d in _DELTAS:\n"
    "                    _c = _cm_d[_d].astype(np.float64)\n"
    "                    _iou = np.diag(_c) / np.maximum(_c.sum(0) + _c.sum(1) - np.diag(_c), 1)\n"
    "                    _bb, _nn = _iou[:self.cfg.data.num_bases].mean(), _iou[self.cfg.data.num_bases:].mean()\n"
    "                    logger.info(\"[audit-delta] %+5.2f  %.4f  %.4f  %.4f\"\n"
    "                                % (_d, _bb, _nn, 2 * _bb * _nn / max(_bb + _nn, 1e-9)))\n"
    "                logger.info(\"[audit-gamma] gamma  Base   Novel   HM\")\n"
    "                for _gm in _GAMMAS:\n"
    "                    _c = _cm_g[_gm].astype(np.float64)\n"
    "                    _iou = np.diag(_c) / np.maximum(_c.sum(0) + _c.sum(1) - np.diag(_c), 1)\n"
    "                    _bb, _nn = _iou[:self.cfg.data.num_bases].mean(), _iou[self.cfg.data.num_bases:].mean()\n"
    "                    logger.info(\"[audit-gamma] %.1f  %.4f  %.4f  %.4f\"\n"
    "                                % (_gm, _bb, _nn, 2 * _bb * _nn / max(_bb + _nn, 1e-9)))\n"
    + old3
)
assert src.count(old3) == 1, "保存锚点 %d" % src.count(old3)
src = src.replace(old3, new3)

shutil.copy2(P, BAK)
io.open(P, "w", encoding="utf-8").write(src)
print("已加 GFS_HEAD_AUDIT（备份 %s）" % BAK)
