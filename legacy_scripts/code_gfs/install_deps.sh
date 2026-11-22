#!/bin/bash
# Sonata 环境依赖安装（torch 用本地 wheel，其余走清华镜像）
# 注意：不装 flash-attn，官方支持 enable_flash=False
set -e
PIP=/data01/jh/conda_envs/sonata/bin/pip
IDX="-i https://pypi.tuna.tsinghua.edu.cn/simple"

echo "===== [1/5] torch (local wheel) ====="
$PIP install $IDX /tmp/torch-2.5.0-cp310-cp310-manylinux1_x86_64.whl

echo "===== [2/5] torchvision / torchaudio ====="
$PIP install $IDX torchvision==0.20.0 torchaudio==2.5.0

echo "===== [3/5] basics ====="
$PIP install $IDX "numpy<=1.26.4" ninja scipy addict packaging timm psutil huggingface_hub

echo "===== [4/5] spconv ====="
$PIP install $IDX spconv-cu124

echo "===== [5/5] torch-scatter ====="
$PIP install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

echo "===== verify ====="
/data01/jh/conda_envs/sonata/bin/python - <<'PY'
import torch, spconv, torch_scatter, timm
print('torch', torch.__version__, 'cuda', torch.version.cuda,
      'available', torch.cuda.is_available())
print('spconv', spconv.__version__)
print('torch_scatter', torch_scatter.__version__)
print('timm', timm.__version__)
PY
