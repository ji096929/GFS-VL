"""Compatibility shim: provide torch_points_kernels via pointops2.

COSeg only uses tp.ball_query in getFeatures()/forward(). The real
torch_points_kernels build is extremely slow here, so we patch the module
before model.coseg imports it.
"""
import sys
import types

import torch

sys.path.insert(0, '/data01/jh/code/COSeg')
from lib.pointops2.functions import pointops as _ops  # noqa: E402


def _offsets_from_batch(batch):
    change = (batch[1:] != batch[:-1]).nonzero(as_tuple=False).flatten() + 1
    last = torch.tensor([batch.shape[0]], dtype=torch.int64, device=batch.device)
    return torch.cat([change.to(torch.int64), last])


def ball_query(radius, nsample, xyz, new_xyz=None, mode=None,
               batch_x=None, batch_y=None):
    if new_xyz is None:
        new_xyz = xyz
    if batch_x is None:
        batch_x = torch.zeros(xyz.shape[0], dtype=torch.int64, device=xyz.device)
    if batch_y is None:
        batch_y = batch_x
    off_x = _offsets_from_batch(batch_x)
    off_y = _offsets_from_batch(batch_y)
    idx, _ = _ops.ball_query(radius, nsample, xyz, new_xyz, off_x, off_y)
    return idx, None


try:
    import torch_points_kernels  # noqa: F401
except ImportError:
    pkg = types.ModuleType('torch_points_kernels')
    pkg.__path__ = []
    pkg.ball_query = ball_query
    pkg.__version__ = '0.6.10-shim'

    cpu = types.ModuleType('torch_points_kernels.points_cpu')
    cpu.ball_query = lambda *a, **k: None
    pkg.points_cpu = cpu

    cuda = types.ModuleType('torch_points_kernels.points_cuda')
    cuda.ball_query = lambda *a, **k: None
    pkg.points_cuda = cuda

    sys.modules['torch_points_kernels'] = pkg
    sys.modules['torch_points_kernels.points_cpu'] = cpu
    sys.modules['torch_points_kernels.points_cuda'] = cuda

# torch_points3d 1.3.0's models/base_model.py drags in the whole datasets
# package (h5py, open3d, python-louvain, ...). COSeg only needs
# BaseInternalLossModule from it, so register a stub before KPConv imports it.
import torch.nn as _nn  # noqa: E402

_TP_ROOT = '/data01/jh/conda_envs/coseg/lib/python3.7/site-packages/torch_points3d'
_tp3d = types.ModuleType('torch_points3d')
_tp3d.__path__ = [_TP_ROOT]
sys.modules.setdefault('torch_points3d', _tp3d)

_models = types.ModuleType('torch_points3d.models')
_models.__path__ = [_TP_ROOT + '/models']
sys.modules['torch_points3d.models'] = _models


class BaseInternalLossModule(_nn.Module):
    def get_internal_loss(self, *args, **kwargs):
        return None


_bm = types.ModuleType('torch_points3d.models.base_model')
_bm.BaseInternalLossModule = BaseInternalLossModule
sys.modules['torch_points3d.models.base_model'] = _bm
_models.base_model = _bm
