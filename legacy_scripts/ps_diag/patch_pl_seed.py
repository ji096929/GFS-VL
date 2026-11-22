# -*- coding: utf-8 -*-
"""给 5-split 伪标签生成链加目录参数（默认值保持不变，向后兼容）。

链路：dump_online_pl.py -> build_pl_weights.py -> build_w_uniform.py
每个脚本原本把输入/输出目录写死，多 support 种子无法复用。
"""
import shutil


def patch(path, pairs):
    shutil.copy(path, path + '.bak_split')
    s = open(path).read()
    for old, new in pairs:
        if old not in s:
            print('MISS %-24s | %s' % (path.split('/')[-1], old.splitlines()[0][:70]))
            continue
        s = s.replace(old, new, 1)
        print('OK   %-24s | %s' % (path.split('/')[-1], old.splitlines()[0][:70]))
    open(path, 'w').write(s)


# ---------- 1. dump_online_pl.py ----------
patch('/data01/jh/code/ps_diag/dump_online_pl.py', [
    ("""def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)""",
     """def main():
    global PROTO, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--proto', default=PROTO,
                    help='支持集原型 npy（多 support 种子各一份）')
    ap.add_argument('--out', default=OUT,
                    help='逐场景 npz 输出目录')
    ap.add_argument('--tag', default='',
                    help='分片统计与拒绝明细文件名后缀，如 _s20')
    args = ap.parse_args()
    PROTO, OUT = args.proto, args.out
    os.makedirs(OUT, exist_ok=True)"""),
    ("""    np.savez('/data01/jh/data/online_pl_shard%d.npz' % args.shard, **agg)
    with open('/data01/jh/data/online_pl_rej_s%d.json' % args.shard, 'w') as f:""",
     """    np.savez('/data01/jh/data/online_pl_shard%s%d.npz' % (args.tag, args.shard), **agg)
    with open('/data01/jh/data/online_pl_rej%s_s%d.json' % (args.tag, args.shard), 'w') as f:"""),
])

# ---------- 2. build_pl_weights.py ----------
patch('/data01/jh/code/ps_diag/build_pl_weights.py', [
    ("""def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto', default=PROTO)""",
     """def main():
    global ONLINE, OUT, REJ
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto', default=PROTO)
    ap.add_argument('--online', default=ONLINE,
                    help='在线伪标签 npz 目录')
    ap.add_argument('--out', default=OUT,
                    help='逐场景 _pl/_w 输出目录')
    ap.add_argument('--rej', default=REJ,
                    help='PS 拒绝明细 json')"""),
    ("""    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)""",
     """    args = ap.parse_args()
    ONLINE, OUT, REJ = args.online, args.out, args.rej
    os.makedirs(OUT, exist_ok=True)"""),
])

# ---------- 3. build_w_uniform.py ----------
patch('/data01/jh/code/ps_diag/build_w_uniform.py', [
    ("""import glob
import os""",
     """import argparse
import glob
import os"""),
    ("""def main():
    os.makedirs(DST, exist_ok=True)""",
     """def main():
    global SRC, DST
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--dst', default=DST)
    args = ap.parse_args()
    SRC, DST = args.src, args.dst
    os.makedirs(DST, exist_ok=True)"""),
])
print('patched')
