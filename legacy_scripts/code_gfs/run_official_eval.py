"""用 fake_tp 垫片跑官方 main_fs.py（官方脚本没有这个 shim，会撞 torch_points3d 的依赖链）。"""
import runpy
import sys

sys.path.insert(0, "/data01/jh/code")
import fake_tp  # noqa: F401,E402

sys.path.insert(0, "/data01/jh/code/COSeg")
sys.argv = ["main_fs.py"] + sys.argv[1:]
runpy.run_path("/data01/jh/code/COSeg/main_fs.py", run_name="__main__")
