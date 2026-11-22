# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/pointcept/engines/test.py'
lines = open(p, encoding='utf-8').read().split('\n')

idxs = [i for i, l in enumerate(lines) if 'S3DISDataset' in l]
assert len(idxs) == 2, 'S3DISDataset occurrences = %d' % len(idxs)
start = idxs[1]   # GFS 的 one_regis_test 里那一处

# 校验块边界（torch.save 到 ) 共 8 行）
assert 'torch.save(' in lines[start + 1], lines[start + 1]
assert lines[start + 8].strip() == ')', repr(lines[start + 8])

new_block = [
    '            # 保存逐场景 inter/union/target，供离线计算 Precision/Recall',
    '            _split = getattr(self.test_loader.dataset, "split", "val")',
    '            _save = os.path.join(save_path, f"{_split}.pth")',
    '            try:',
    '                torch.save(',
    '                    dict(',
    '                        intersection=intersection,',
    '                        union=union,',
    '                        target=target,',
    '                        record=record,',
    '                    ),',
    '                    _save,',
    '                )',
    '                self.logger.info("Saved per-scene stats to: " + _save)',
    '            except Exception as _e:  # noqa: BLE001',
    '                self.logger.info("Failed to save per-scene stats: %s" % _e)',
]

lines[start:start + 9] = new_block
open(p, 'w', encoding='utf-8').write('\n'.join(lines))
print('patched at line', start + 1)
