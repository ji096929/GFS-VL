"""GPU 占用记录器：每 60s 追加一行，用于摸清 lwh 等共享用户的周期性评估峰值。

输出格式（制表符分隔）
  时间 | g0_used/g0_util ... g7_used/g7_util | pid:user:memM ...
"""
import subprocess
import time
from datetime import datetime

LOG = '/data01/jh/afford_diag/gfs_build/gpu_watch.log'


def run(args):
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def main():
    user_of = {}
    while True:
        ts = datetime.now().strftime('%m-%d %H:%M:%S')
        g = run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                 '--format=csv,noheader,nounits'])
        gpus = []
        for line in g.splitlines():
            idx, used, util = [x.strip() for x in line.split(',')]
            gpus.append('%s:%s/%s' % (idx, used, util))
        procs = []
        p = run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                 '--format=csv,noheader,nounits'])
        for line in p.splitlines():
            pid, mem = [x.strip() for x in line.split(',')]
            u = user_of.get(pid)
            if u is None:
                u = run(['ps', '-o', 'user=', '-p', pid]) or '?'
                user_of[pid] = u
            procs.append('%s:%s:%sM' % (pid, u, mem))
        with open(LOG, 'a') as f:
            f.write('%s\t%s\t%s\n' % (ts, ' '.join(gpus), ' '.join(procs)))
        time.sleep(60)


if __name__ == '__main__':
    main()
