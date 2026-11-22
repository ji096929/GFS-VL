"""百度网盘分享：按场景只转存 scans/ 子目录。

接口要点（2026-11 实测）
  * 分享页改版，baidupcs-py 的 shared_paths() 解析不到 yunData.setData
  * share/list 的 dir 要用虚拟前缀 /sharelink0-<linkid>/<分享根目录名>/...
  * sekey = cookie BDCLND 解一次 URL 编码；不解码会被二次编码成 %252B 报 -9
  * logid = base64(BAIDUID + ':FG=1')，不校验但必须带

用法
  python bd_transfer.py meta            转存 splits/ metadata/
  python bd_transfer.py count           数一下 data/ 下有多少场景
  python bd_transfer.py run             转存所有 data/*/scans
"""
import argparse
import base64
import json
import os
import re
import time
import urllib.parse as up

from baidupcs_py.baidupcs import BaiduPCSApi
from baidupcs_py.baidupcs.pcs import Method, PAN_HEADERS

URL = 'https://pan.baidu.com/s/1XGJUBk_lIX_JStOa7xwvdg'
PWD = 'ss63'
ROOT = '/scannetpp'
BDCK = os.environ.get('BDCK', '/data01/jh/afford_diag/.bdck')
LOG = '/data01/jh/afford_diag/gfs_build/bd_transfer.log'


class Share:
    def __init__(self):
        ck = {}
        with open(BDCK) as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    k, v = line.split('=', 1)
                    ck[k] = v
        self.api = BaiduPCSApi(cookies=ck, user_id=1)
        self.api.access_shared(URL, PWD, show_vcode=False)
        self.pcs = self.api._baidupcs
        d = self.pcs._session.cookies.get_dict()
        self.sekey = up.unquote(d['BDCLND'])
        self.logid = base64.b64encode((d['BAIDUID'] + ':FG=1').encode()).decode()
        h = self.pcs._request(Method.Get, URL).text
        self.uk = int(re.search(r'"share_uk"\s*:\s*"(\d+)"', h).group(1))
        self.sid = int(re.search(r'"shareid"\s*:\s*(\d+)', h).group(1))
        m = re.search(r'sharelink(\d+)-(\d+)', h)
        self.linkid = m.group(2)
        self.prefix = '/sharelink0-' + self.linkid
        self.bt = self.api.bdstoken
        # 分享根是单个目录；HTML 的 file_list 第一条就是它
        self._root = re.search(r'"server_filename"\s*:\s*"([^"]+)"', h).group(1)

    def _get(self, url, params):
        j = {}
        for i in range(4):
            j = self.pcs._request(Method.Get, url, params=params).json()
            if j.get('errno') in (0, None):
                return j
            time.sleep(2 + 3 * i)
        return j

    def list(self, d, page=1, size=100):
        p = {'uk': str(self.uk), 'shareid': str(self.sid), 'sekey': self.sekey,
             'is_from_web': 'true', 'dir': d, 'page': str(page), 'num': str(size),
             'order': 'name', 'desc': '0', 'showempty': '0', 'view_mode': '1',
             'web': '1', 'channel': 'chunlei', 'app_id': '250528', 'bdstoken': '',
             'clienttype': '0', 'logid': self.logid}
        return self._get('https://pan.baidu.com/share/list', p)

    def list_all(self, d):
        out, page = [], 1
        while True:
            j = self.list(d, page)
            lst = j.get('list') or []
            if not lst:
                break
            out.extend(lst)
            if len(lst) < 100:
                break
            page += 1
        return out

    def root_name(self):
        # 分享根是单个目录，其 server_filename 即根目录名
        if not self._root:
            self._root = self.list('/')['list'][0]['server_filename']
        return self._root

    def mkdir(self, path):
        return self.pcs._request(
            Method.Post, 'https://pan.baidu.com/api/create',
            params={'a': 'commit', 'bdstoken': self.bt},
            data={'path': path, 'isdir': '1', 'block_list': '[]', 'size': '0'},
        ).json()

    def transfer(self, remotedir, fs_ids):
        hdrs = dict(PAN_HEADERS)
        hdrs['X-Requested-With'] = 'XMLHttpRequest'
        hdrs['Content-Type'] = 'application/x-www-form-urlencoded; charset=UTF-8'
        hdrs['Origin'] = 'https://pan.baidu.com'
        hdrs['Referer'] = URL
        r = self.pcs._request(
            Method.Post, 'https://pan.baidu.com/share/transfer', headers=hdrs,
            params={'shareid': str(self.sid), 'from': str(self.uk), 'sekey': self.sekey,
                    'ondup': 'newcopy', 'async': '1', 'channel': 'chunlei',
                    'clienttype': '0', 'web': '1', 'app_id': '250528',
                    'bdstoken': self.bt},
            data={'fsidlist': json.dumps(fs_ids), 'path': remotedir})
        return r.json()

    def own_dirs(self):
        out, page = set(), 1
        while True:
            j = self.pcs._request(
                Method.Get, 'https://pan.baidu.com/api/list',
                params={'dir': ROOT, 'page': str(page), 'num': '1000', 'order': 'name',
                        'desc': '0', 'bdstoken': self.bt, 'web': '1',
                        'channel': 'chunlei', 'clienttype': '0', 'app_id': '250528'}
            ).json()
            if j.get('errno') not in (0, None):
                print('own_dirs errno', j.get('errno'), flush=True)
                break
            lst = j.get('list') or []
            for x in lst:
                out.add(x['server_filename'])
            if len(lst) < 1000:
                break
            page += 1
        return out


def log(line):
    with open(LOG, 'a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    args = ap.parse_args()
    s = Share()
    print('uk=%s shareid=%s linkid=%s root=%s'
          % (s.uk, s.sid, s.linkid, s.root_name()), flush=True)

    if args.cmd == 'count':
        for sub in ('data', 'splits', 'metadata'):
            lst = s.list_all(s.prefix + '/' + s.root_name() + '/' + sub)
            dirs = [x for x in lst if x.get('isdir') == 1]
            size = sum(int(x.get('size') or 0) for x in lst if x.get('isdir') == 0)
            print('%-8s 条目 %4d  目录 %4d  文件合计 %.1f MB'
                  % (sub, len(lst), len(dirs), size / 1e6))
        return

    if args.cmd == 'meta':
        s.mkdir(ROOT + '/_meta')
        for sub in ('splits', 'metadata'):
            lst = s.list_all(s.prefix + '/' + s.root_name() + '/' + sub)
            ids = [x['fs_id'] for x in lst]
            r = s.transfer(ROOT + '/_meta', ids)
            print(sub, 'n=%d' % len(ids), 'errno', r.get('errno'), flush=True)
        return

    if args.cmd == 'run':
        base = s.prefix + '/' + s.root_name() + '/data'
        scenes = [x for x in s.list_all(base) if x.get('isdir') == 1]
        scenes.sort(key=lambda x: x['server_filename'])
        print('data 下场景目录 %d 个' % len(scenes), flush=True)
        have = s.own_dirs()
        print('自己网盘已有 %d 个目录' % len(have), flush=True)
        todo = [x for x in scenes if x['server_filename'] not in have]
        print('待转存 %d 个' % len(todo), flush=True)
        ok = err = 0
        t0 = time.time()
        for i, x in enumerate(todo):
            sc = x['server_filename']
            sub = s.list_all(base + '/' + sc)
            scans = [y for y in sub
                     if y.get('server_filename') == 'scans' and y.get('isdir') == 1]
            if not scans:
                log('MISS %s' % sc)
                err += 1
                continue
            dest = ROOT + '/' + sc
            s.mkdir(dest)
            r = s.transfer(dest, [scans[0]['fs_id']])
            if r.get('errno') == 0:
                ok += 1
                log('OK %s %.1fMB' % (sc, int(scans[0].get('size') or 0) / 1e6))
            else:
                err += 1
                log('ERR %s %s' % (sc, json.dumps(r, ensure_ascii=False)[:200]))
            if (i + 1) % 25 == 0:
                print('--- %d/%d  %.0fs  ok=%d err=%d'
                      % (i + 1, len(todo), time.time() - t0, ok, err), flush=True)
            time.sleep(0.2)
        print('DONE ok=%d err=%d %.0fs' % (ok, err, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
