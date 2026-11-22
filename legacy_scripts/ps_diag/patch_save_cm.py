# -*- coding: utf-8 -*-
"""给 GFSSemSegTester.one_regis_test 加可选的真实混淆矩阵保存（GFS_SAVE_CM=1）。

用途：把 Novel mIoU 的丢分拆成 4 个互斥桶
  A: Novel -> Base（含 Base-Sibling 对）
  B: Novel -> Novel 互混
  C: VLM 盲区（该类 recall≈0）
  D: VLM 幻觉（背景/Base 被误判成 Novel，只看 FP 侧）

设计要点：
  - 纯增量，不改变任何已有指标的计算和记录方式
  - 仅在环境变量 GFS_SAVE_CM=1 时生效，默认完全无行为变化
  - 已加载旧代码的进程不受影响（Python 模块不热更新）
"""
import io
import shutil

P = "/data01/jh/code/GFS-VL/pointcept/engines/test.py"
BAK = P + ".bak_cm"

src = io.open(P, encoding="utf-8").read()
if "GFS_SAVE_CM" in src:
    print("已存在 GFS_SAVE_CM，跳过")
    raise SystemExit(0)

# 1) 在 one_regis_test 的 record = {} 之后初始化累加器（用 save_path 锚点保证唯一）
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
    "        # 可选：完整混淆矩阵（GFS_SAVE_CM=1 时启用，默认关闭）\n"
    "        _cm_on = bool(os.environ.get(\"GFS_SAVE_CM\"))\n"
    "        _cm = np.zeros(\n"
    "            (self.cfg.data.num_base_novels, self.cfg.data.num_base_novels),\n"
    "            dtype=np.int64) if _cm_on else None\n"
)
assert src.count(old1) == 1, "初始化锚点 %d" % src.count(old1)
src = src.replace(old1, new1)

# 2) 在 intersection_and_union 之前累积（pred 与 segment 此时都已对齐到原点云）
old2 = (
    "            intersection, union, target = intersection_and_union(\n"
    "                pred,\n"
    "                segment,\n"
    "                self.cfg.data.num_base_novels,\n"
    "                self.cfg.data.ignore_index,\n"
    "            )\n"
)
new2 = (
    "            if _cm_on:\n"
    "                _NC = self.cfg.data.num_base_novels\n"
    "                _v = (segment >= 0) & (segment < _NC)\n"
    "                if _v.any():\n"
    "                    _cm += np.bincount(\n"
    "                        segment[_v].astype(np.int64) * _NC\n"
    "                        + pred[_v].astype(np.int64),\n"
    "                        minlength=_NC * _NC,\n"
    "                    ).reshape(_NC, _NC)\n"
    + old2
)
assert src.count(old2) == 1, "累积锚点 %d" % src.count(old2)
src = src.replace(old2, new2)

# 3) 在保存 val.pth 的地方额外存混淆矩阵
old3 = (
    "            _split = getattr(self.test_loader.dataset, \"split\", \"val\")\n"
    "            _save = os.path.join(save_path, f\"{_split}.pth\")\n"
)
new3 = (
    "            _split = getattr(self.test_loader.dataset, \"split\", \"val\")\n"
    "            if _cm_on and _cm is not None:\n"
    "                _cmp = os.path.join(save_path, f\"{_split}_cm.npy\")\n"
    "                np.save(_cmp, _cm)\n"
    "                logger.info(\"=> Confusion matrix saved: %s\" % _cmp)\n"
    + old3
)
assert src.count(old3) == 1, "保存锚点 %d" % src.count(old3)
src = src.replace(old3, new3)

shutil.copy2(P, BAK)
io.open(P, "w", encoding="utf-8").write(src)
print("已改 %s（备份 %s）" % (P, BAK))
