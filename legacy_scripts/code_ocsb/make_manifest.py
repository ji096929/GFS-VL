"""Build a fixed paired episode manifest.

Every experiment (baseline, zero-shot 2C, Stage II validation) runs on the same
list so that per-episode differences can be paired. The earlier free-running
shards resampled episodes every time, which is why the SE on a 500-episode
mean was 0.6-0.7pp -- far too coarse to call a +0.5pp improvement.

Each entry fixes support/query and BOTH U variants:
  u_cross -- class-bearing blocks from other scenes (strict scene-disjoint)
  u_same  -- class-bearing blocks from the query scene, excluding s/q themselves
With one manifest we can run cross and same U on identical (S, Q) pairs.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# Must match diag_bc.NOVEL_FOLD0 exactly: COSeg ScanNet fold-0 has 10 novel
# classes. Listing base classes here would evaluate on classes the backbone was
# trained on and inflate every number.
NOVEL_FOLD0 = ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair',
               'counter', 'curtain', 'desk', 'door', 'floor']


def scan_blocks(data_dir):
    blocks = []
    for f in sorted(glob.glob(os.path.join(data_dir, '*.npy'))):
        d = np.load(f, mmap_mode='r')
        labels = np.asarray(d[:, 6]).astype(int)
        name = os.path.basename(f)[:-4]
        blocks.append({
            'name': name,
            'scene': name.split('_block_')[0],
            'classes': set(np.unique(labels).tolist()),
            'n': int(d.shape[0]),
        })
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',
                    default='/data01/jh/data/ScanNet/blocks_bs1_s1/data')
    ap.add_argument('--out', default='/data01/jh/data/manifest_fold0.json')
    ap.add_argument('--episodes', type=int, default=500)
    ap.add_argument('--M', type=int, default=3)
    ap.add_argument('--min-same', type=int, default=2,
                    help='require this many same-scene U blocks so the two '
                         'protocols stay paired')
    ap.add_argument('--seed', type=int, default=2024)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    print('scanning blocks...', flush=True)
    blocks = scan_blocks(args.data)
    print('blocks:', len(blocks), flush=True)

    # diag_bc reads scannet_classnames.txt; use the same file so class ids
    # line up one-to-one with the block label column.
    class_names = None
    for cand in ('/data01/jh/code/COSeg/datasets/ScanNet/meta/'
                 'scannet_classnames.txt',):
        # exactly the file diag_bc.META points at (20 lines: unannotated, wall,
        # floor, chair, ...), so ids match the block label column
        if os.path.exists(cand):
            class_names = [l.strip() for l in open(cand)]
            break
    if class_names is None:
        raise SystemExit('class name file not found')
    novel_ids = [i for i, n in enumerate(class_names) if n in NOVEL_FOLD0]
    print('novel ids:', novel_ids)

    class2blocks = {c: [b for b in blocks if c in b['classes']]
                    for c in novel_ids}
    for c in novel_ids:
        print('  class %2d -> %d blocks' % (c, len(class2blocks[c])),
              flush=True)

    episodes = []
    tries = 0
    max_tries = args.episodes * 200
    while len(episodes) < args.episodes and tries < max_tries:
        tries += 1
        c = int(rng.choice(novel_ids))
        cand = class2blocks[c]
        if len(cand) < 2:
            continue
        order = rng.permutation(len(cand))
        s_info, q_info = cand[order[0]], cand[order[1]]
        used = {s_info['scene'], q_info['scene']}
        u_cross = [b['name'] for b in cand if b['scene'] not in used]
        u_same = [b['name'] for b in cand
                  if b['scene'] == q_info['scene']
                  and b['name'] not in (q_info['name'], s_info['name'])]
        if len(u_cross) < args.M or len(u_same) < args.min_same:
            continue
        pc = rng.choice(len(u_cross), args.M, replace=False)
        ps = rng.choice(len(u_same), min(args.M, len(u_same)),
                        replace=False)
        episodes.append({
            'class': c,
            'support': s_info['name'],
            'query': q_info['name'],
            'u_cross': [u_cross[i] for i in pc],
            'u_same': [u_same[i] for i in ps],
        })

    print('built %d episodes in %d tries' % (len(episodes), tries))
    if not episodes:
        raise SystemExit('no episodes; loosen --min-same')

    # Round-robin by class so any prefix (first 50 / 100 / 200) is balanced.
    # Random order left class 12 at 20% and class 10 at 5% in the first 100,
    # which made short validation runs unrepresentative.
    buckets = defaultdict(list)
    for e in episodes:
        buckets[e['class']].append(e)
    for c in list(buckets):
        buckets[c] = buckets[c][:max(1, args.episodes // len(buckets))]
    ordered, idx = [], 0
    keys = sorted(buckets)
    while len(ordered) < args.episodes:
        added = False
        for c in keys:
            if idx < len(buckets[c]):
                ordered.append(buckets[c][idx])
                added = True
                if len(ordered) >= args.episodes:
                    break
        if not added:
            break
        idx += 1
    episodes = ordered
    print('round-robin reordered -> %d episodes' % len(episodes))

    cnt = defaultdict(int)
    max_possible = {}
    for e in episodes:
        cnt[e['class']] += 1
    print('class mix: ' + ' '.join('%d:%d' % (c, cnt[c]) for c in sorted(cnt)))

    # how many episodes can actually be paired across protocols
    both = sum(1 for e in episodes if e['u_cross'] and e['u_same'])
    print('episodes with both U variants: %d' % both)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'meta': {'seed': args.seed, 'M': args.M, 'data': args.data,
                     'n': len(episodes), 'min_same': args.min_same},
            'episodes': episodes,
        }, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
