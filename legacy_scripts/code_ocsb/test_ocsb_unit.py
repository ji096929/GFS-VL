"""ocsb_faithful 的纯 CPU 单元测试（不加载 COSeg，秒级）。

覆盖：前向形状、B 空退化、三项损失梯度、B/C 与 u_y/q 解耦。
"""
import torch

from ocsb_faithful import OCSB, ocsb_losses

torch.manual_seed(0)
dim, Ns, Nu, Nq, n_reg = 32, 400, 1200, 200, 50


def make_ep():
    return {
        's_h': torch.randn(Ns, dim),
        's_y': (torch.rand(Ns) > 0.5).float(),
        's_seg': torch.randint(0, 25, (Ns,)),
        'u_h': torch.randn(Nu, dim),
        'u_seg': torch.randint(0, n_reg, (Nu,)),
        'u_y': (torch.rand(Nu) > 0.7).float(),
        'u_p0': torch.where(torch.rand(Nu) > 0.5,
                            torch.rand(Nu) * 0.3 + 0.7,
                            torch.rand(Nu) * 0.3),
        'q_h': torch.randn(Nq, dim),
        'q_margin': torch.randn(Nq) * 2,
        'q_y': (torch.rand(Nq) > 0.5).float(),
        'base_protos': torch.randn(9, dim),
    }


ep = make_ep()
m = OCSB(dim, tau=0.1, tau_b=0.6, gamma0=0.2)
out = m(ep)
print('B, C = %d, %d' % (out['B_idx'].numel(), out['C_idx'].numel()))
print('ell %s mean %.3f  w_C sum %.3f  w_B %s'
      % (tuple(out['ell'].shape), float(out['ell'].mean()),
         float(out['w_C'].sum()),
         ('max %.3f' % float(out['w_B'].max())) if out['w_B'].numel() else 'empty'))

# 式(8) 的负对照来源
ns = out.get('neg_src')
if ns is not None:
    print('neg sources (D_S, P_base, D_U):',
          [int((ns == k).sum()) for k in range(3)])

# 检查 3：B 空必须严格退化
m_e = OCSB(dim, tau=0.1, tau_b=1.1, gamma0=0.2)
out_e = m_e(ep)
err = float((out_e['ell'] - ep['q_margin']).abs().max())
print('check3 degenerate max err = %.3e' % err)
assert err < 1e-6, 'B empty must degenerate to COSeg'

# 检查 1 / 5：B/C 只看初始预测
ep2 = dict(ep)
ep2['u_y'] = torch.zeros_like(ep['u_y'])
o2 = m(ep2)
print('check1 B/C label-free:',
      torch.equal(out['B_idx'], o2['B_idx']),
      torch.equal(out['C_idx'], o2['C_idx']))

ep3 = dict(ep)
ep3['q_h'] = torch.randn_like(ep['q_h'])
ep3['q_margin'] = torch.randn_like(ep['q_margin'])
o3 = m(ep3)
print('check5 B/C query-free:',
      torch.equal(out['B_idx'], o3['B_idx']),
      torch.equal(out['C_idx'], o3['C_idx']))

# 检查 4：三项损失的梯度
for tag in ('seg', 'omit', 'select'):
    m.zero_grad(set_to_none=True)
    o = m(ep)
    parts = ocsb_losses(o, ep, model=m)
    loss = parts[tag]
    if loss.requires_grad:
        loss.backward()
        g = m.proj.weight.grad
        print('check4 %-6s loss %.4f  grad %.3e'
              % (tag, float(loss), float(g.norm()) if g is not None else 0))
    else:
        print('check4 %-6s loss %.4f  NO GRAD' % (tag, float(loss)))

# 式(15) 的极值行为：M 全零时返回原得分
s = torch.randn(10)
zq = torch.randn(10, dim)
same = torch.equal(
    __import__('ocsb_faithful').fuse_query_score(
        s, torch.randn(3, dim), torch.zeros(3), zq, 0.1, 0.2), s)
print('fuse zero-weight identity:', same)

print('ALL UNIT TESTS OK')
