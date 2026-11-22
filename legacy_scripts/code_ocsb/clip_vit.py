"""Minimal CLIP ViT-B/16 vision tower, rebuilt from a TorchScript archive.

The server's CLIP checkpoints are TorchScript files that torch 1.11 cannot
execute (the interpreter dies inside conv1). But `named_parameters()` still
works, so we pull the weights out and run a plain PyTorch implementation of the
same architecture. No text tower, no projection head -- we only need dense
patch tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d=768, h=12, mlp=3072):
        super().__init__()
        self.n_heads = h
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d))
        self.out_proj = nn.Linear(d, d)
        self.ln_1 = nn.LayerNorm(d)
        self.ln_2 = nn.LayerNorm(d)
        self.c_fc = nn.Linear(d, mlp)
        self.c_proj = nn.Linear(mlp, d)

    def attn(self, x):
        B, N, C = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        h = self.n_heads
        q = q.view(B, N, h, C // h).transpose(1, 2)
        k = k.view(B, N, h, C // h).transpose(1, 2)
        v = v.view(B, N, h, C // h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v) if hasattr(
            F, 'scaled_dot_product_attention') else self._attn_manual(q, k, v)
        return o.transpose(1, 2).reshape(B, N, C)

    @staticmethod
    def _attn_manual(q, k, v):
        d = q.shape[-1] ** 0.5
        a = (q @ k.transpose(-2, -1)) / d
        return a.softmax(-1) @ v

    def forward(self, x):
        x = x + self.out_proj(self.attn(self.ln_1(x)))
        x = x + self.c_proj(F.gelu(self.c_fc(self.ln_2(x))))
        return x


class CLIPViT(nn.Module):
    def __init__(self, n_layers=12, d=768, n_heads=12, patch=16, img=224):
        super().__init__()
        self.patch = patch
        self.grid = img // patch
        self.class_embedding = nn.Parameter(torch.empty(d))
        self.positional_embedding = nn.Parameter(
            torch.empty(self.grid * self.grid + 1, d))
        self.conv1 = nn.Conv2d(3, d, patch, stride=patch, bias=False)
        self.ln_pre = nn.LayerNorm(d)
        self.ln_post = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(x)                       # [B, d, g, g]
        x = x.flatten(2).transpose(1, 2)        # [B, g*g, d]
        cls = self.class_embedding.expand(B, 1, -1)
        x = torch.cat([cls, x], 1) + self.positional_embedding
        x = self.ln_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_post(x)
        return x[:, 1:]                          # patch tokens only


def load_from_torchscript(path, n_layers=12, verbose=False):
    ts = torch.jit.load(path, map_location='cpu')
    ps = dict(ts.named_parameters())
    d = ps['visual.class_embedding'].shape[0]
    n_heads = 12
    g2 = ps['visual.positional_embedding'].shape[0] - 1
    grid = int(round(g2 ** 0.5))
    patch = 224 // grid
    m = CLIPViT(n_layers, d, n_heads, patch)
    with torch.no_grad():
        m.class_embedding.copy_(ps['visual.class_embedding'])
        m.positional_embedding.copy_(ps['visual.positional_embedding'])
        m.conv1.weight.copy_(ps['visual.conv1.weight'])
        m.ln_pre.weight.copy_(ps['visual.ln_pre.weight'])
        m.ln_pre.bias.copy_(ps['visual.ln_pre.bias'])
        m.ln_post.weight.copy_(ps['visual.ln_post.weight'])
        m.ln_post.bias.copy_(ps['visual.ln_post.bias'])
        for i in range(n_layers):
            src = 'visual.transformer.resblocks.%d.' % i
            dst = m.blocks[i]
            dst.in_proj_weight.copy_(ps[src + 'attn.in_proj_weight'])
            dst.in_proj_bias.copy_(ps[src + 'attn.in_proj_bias'])
            dst.out_proj.weight.copy_(ps[src + 'attn.out_proj.weight'])
            dst.out_proj.bias.copy_(ps[src + 'attn.out_proj.bias'])
            dst.ln_1.weight.copy_(ps[src + 'ln_1.weight'])
            dst.ln_1.bias.copy_(ps[src + 'ln_1.bias'])
            dst.ln_2.weight.copy_(ps[src + 'ln_2.weight'])
            dst.ln_2.bias.copy_(ps[src + 'ln_2.bias'])
            dst.c_fc.weight.copy_(ps[src + 'mlp.c_fc.weight'])
            dst.c_fc.bias.copy_(ps[src + 'mlp.c_fc.bias'])
            dst.c_proj.weight.copy_(ps[src + 'mlp.c_proj.weight'])
            dst.c_proj.bias.copy_(ps[src + 'mlp.c_proj.bias'])
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    if verbose:
        print('CLIPViT d=%d layers=%d grid=%d patch=%d' % (d, n_layers, grid, patch))
    return m


if __name__ == '__main__':
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else \
        '/data01/ysz/UniMoS/.runtime/cache/clip/ViT-B-16.pt'
    m = load_from_torchscript(p, verbose=True).cuda()
    x = torch.randn(2, 3, 224, 224).cuda()
    with torch.no_grad():
        out = m(x)
    print('out', tuple(out.shape), 'mean %.3f std %.3f'
          % (out.mean().item(), out.std().item()))
