"""重试 transfer 超时（errno=4）的场景。

只处理 bd_transfer.log 里 ERR 行对应的场景：重新取 scans 的 fs_id，
transfer 到 /scannetpp/<scene>（目录已存在，ondup=newcopy 会补进去）。
"""
import json
import re
import sys
import time

sys.path.insert(0, '/data01/jh/code/ps_diag')
import bd_transfer as B  # noqa: E402


def main():
    log = '/data01/jh/afford_diag/gfs_build/bd_transfer.log'
    scenes = []
    for line in open(log):
        m = re.match(r'ERR (\S+) ', line)
        if m:
            scenes.append(m.group(1))
    scenes = sorted(set(scenes))
    print('待重试 %d 个场景' % len(scenes), flush=True)

    s = B.Share()
    base = s.prefix + '/' + s.root_name() + '/data'
    ok = err = 0
    for i, sc in enumerate(scenes):
        sub = s.list_all(base + '/' + sc)
        scans = [y for y in sub
                 if y.get('server_filename') == 'scans' and y.get('isdir') == 1]
        if not scans:
            print('MISS %s' % sc, flush=True)
            err += 1
            continue
        dest = B.ROOT + '/' + sc
        s.mkdir(dest)
        r = s.transfer(dest, [scans[0]['fs_id']])
        if r.get('errno') == 0:
            ok += 1
            print('OK %s' % sc, flush=True)
        else:
            err += 1
            print('ERR %s %s' % (sc, json.dumps(r, ensure_ascii=False)[:160]), flush=True)
        time.sleep(0.5)
    print('DONE ok=%d err=%d' % (ok, err), flush=True)


if __name__ == '__main__':
    main()
