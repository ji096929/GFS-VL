import sys
sys.path.insert(0, '/data01/jh/code/GFS-VL')
sys.path.insert(0, '/data01/jh/code/GFS-VL/pointcept/models/PLA')
import torch
from pointcept.models.point_transformer_v3 import PointTransformerV3

m = PointTransformerV3(
    in_channels=6,
    order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
    stride=(2, 2, 2, 2),
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(32, 64, 128, 256, 512),
    enc_num_head=(2, 4, 8, 16, 32),
    enc_patch_size=(1024,) * 5,
    dec_depths=(2, 2, 2, 2),
    dec_channels=(64, 64, 128, 256),
    dec_num_head=(4, 4, 8, 16),
    dec_patch_size=(1024,) * 4,
    mlp_ratio=4,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=False,
)
n = sum(p.numel() for p in m.parameters())
MB = 1024 ** 2
print('backbone params : %.2f M (%d)' % (n / 1e6, n))
print('  fp32 weights  : %8.1f MB' % (n * 4 / MB))
print('  fp32 grads    : %8.1f MB' % (n * 4 / MB))
print('  AdamW exp_avg : %8.1f MB' % (n * 4 / MB))
print('  AdamW exp_sq  : %8.1f MB' % (n * 4 / MB))
print('  subtotal      : %8.1f MB' % (n * 16 / MB))
