p = '/data01/jh/code/GFS-VL/OVERNIGHT_ICCV_REPORT.md'
s = open(p, encoding='utf-8').read()

old = '''**5-shot 结果**：`sym_lcross01_k5_102400_s42` 进行中（约 07:55 出分）。'''

new = '''**5-shot 结果（已完成）**

| 配置（102400 + 真 5-shot PL） | Base | Novel | HM | ΔHM |
|---|---|---|---|---|
| sym_k5_102400（无 L_cross） | 0.6843 | 0.3253 | 0.4410 | — |
| **sym_lcross01_k5_102400（λ=0.1）** | 0.6853 | 0.3246 | **0.4406** | **−0.04** |

**L_cross 完整归档**：

| 口径 | ΔHM vs 无 L_cross |
|---|---|
| 1-shot 65536（λ=0.3） | −0.27 |
| 1-shot 65536（λ=0.1） | −0.23 |
| 5-shot 65536（λ=0.3） | +0.41 |
| **5-shot 102400（λ=0.1）** | **−0.04** |

四个口径里三个负、一个噪声。**在满血口径下，之前 65536 看到的 +0.41 也消失了**——它本来就是低口径下的噪声。L_cross 这条线彻底关闭。'''

assert s.count(old) == 1, 'anchor=%d' % s.count(old)
s = s.replace(old, new)

old2 = '''| L_cross λ=0.1（损失端，跨层子空间） | −0.23 | 待出 | ❌ |'''
new2 = '''| L_cross λ=0.1（损失端，跨层子空间） | −0.23 | **−0.04** | ❌ |'''
assert s.count(old2) == 1
s = s.replace(old2, new2)

open(p, 'w', encoding='utf-8').write(s)
print('final report patched')
