# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/pointcept/engines/train.py'
s = open(p, encoding='utf-8').read()

# 1) 初始化多原型容器
o1 = '''        num_novel = self.cfg.data.num_base_novels - self.cfg.data.num_bases
        average_features = torch.zeros(num_novel, 512, device=\"cuda\")
        class_counts = torch.zeros(num_novel, device=\"cuda\")'''
n1 = '''        num_novel = self.cfg.data.num_base_novels - self.cfg.data.num_bases
        average_features = torch.zeros(num_novel, 512, device=\"cuda\")
        class_counts = torch.zeros(num_novel, device=\"cuda\")
        # 多原型：每个 shot 一个独立原型（修 5-shot 单均值坍塌）
        _K = max(int(getattr(self.cfg, \"k_shot\", 1) or 1), 1)
        multi_protos = torch.zeros(num_novel, _K, 512, device=\"cuda\")
        multi_valid = torch.zeros(num_novel, _K, dtype=torch.bool, device=\"cuda\")
        _shot_cnt = torch.zeros(num_novel, dtype=torch.long, device=\"cuda\")'''
assert s.count(o1) == 1, 'a1=%d' % s.count(o1)
s = s.replace(o1, n1)

# 2) 每个 shot 存一个原型
o2 = '''                        average_features[idx] += class_features.sum(dim=0)
                        class_counts[idx] += class_features.size(0)'''
n2 = '''                        average_features[idx] += class_features.sum(dim=0)
                        class_counts[idx] += class_features.size(0)
                        _k = int(_shot_cnt[idx])
                        if _k < _K:
                            multi_protos[idx, _k] = class_features.mean(dim=0)
                            multi_valid[idx, _k] = True
                            _shot_cnt[idx] += 1'''
assert s.count(o2) == 1, 'a2=%d' % s.count(o2)
s = s.replace(o2, n2)

# 3) 存到 self，供 PS 使用
o3 = '''        if comm.get_world_size() > 1:
            dist.broadcast(average_features, src=0)
        return average_features'''
n3 = '''        if comm.get_world_size() > 1:
            dist.broadcast(average_features, src=0)
        self.vlm_novel_protos = (multi_protos, multi_valid)
        return average_features'''
assert s.count(o3) == 1, 'a3=%d' % s.count(o3)
s = s.replace(o3, n3)

# 4) PS 改为 max-sim
o4 = '''                    novel_proto = self.vlm_novel_proto[
                        class_id - self.cfg.data.num_bases
                    ]
                    similarity = torch.cosine_similarity(
                        proto, novel_proto, dim=0
                    )'''
n4 = '''                    _ni = class_id - self.cfg.data.num_bases
                    similarity = None
                    if hasattr(self, \"vlm_novel_protos\"):
                        _protos, _valid = self.vlm_novel_protos
                        _p = _protos[_ni][_valid[_ni]]
                        if _p.numel() > 0:
                            # 取与该类 K 个 shot 原型中的最大相似度
                            similarity = torch.cosine_similarity(
                                proto.unsqueeze(0), _p, dim=1).max()
                    if similarity is None:
                        similarity = torch.cosine_similarity(
                            proto, self.vlm_novel_proto[_ni], dim=0
                        )'''
assert s.count(o4) == 1, 'a4=%d' % s.count(o4)
s = s.replace(o4, n4)

open(p, 'w', encoding='utf-8').write(s)
print('multi-proto added')
