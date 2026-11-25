"""
ScanNet++ dataset Efficient Generalized Few-Shot Dataset

Author: Zhaochong An (anzhaochong@outlook.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob

from pointcept.utils.cache import shared_dict

from .builder import DATASETS
from .defaults import DefaultDataset

from .preprocessing.scannetpp.metadata.semantic_benchmark.top100 import (
    CLASS_LABELS_BASE,
    CLASS_LABELS_BASE_NOVEL,
    CLASS_LABELS_100,
)
import pointcept.utils.comm as comm
import pickle
import random
from collections import defaultdict


@DATASETS.register_module()
class ScanNetPPDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "instance",
    ]

    def __init__(
        self,
        multilabel=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.multilabel = multilabel

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if not self.multilabel:
            if "segment" in data_dict.keys():
                data_dict["segment"] = data_dict["segment"][:, 0].astype(
                    np.int32
                )
            else:
                data_dict["segment"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )

            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][:, 0].astype(
                    np.int32
                )
            else:
                data_dict["instance"] = (
                    np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
                )
        else:
            raise NotImplementedError
        return data_dict


@DATASETS.register_module()
class ScanNetPPDataset_GFS(ScanNetPPDataset):
    """
    ScanNet++ dataset prototype for generalized few-shot learning.
    Loads the 'segment' asset and converts labels using a lookup array.
    """

    VALID_ASSETS = ["coord", "color", "normal", "segment"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set background label
        self.bg_label = -1
        # Initialize lookup array based on CLASS_LABELS_100.
        self.lookup_array = np.full(
            len(CLASS_LABELS_100), self.bg_label, dtype=np.int32
        )

    def convert_index(self, seg_array):
        """
        Convert segmentation labels using the lookup array.
        """
        converted = np.full(seg_array.shape, self.bg_label)
        valid_mask = seg_array != -1
        converted[valid_mask] = self.lookup_array[seg_array[valid_mask]]
        return converted

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        sample_name = self.get_data_name(idx)
        if self.cache:
            return shared_dict(f"pointcept-{sample_name}")

        sample = {}
        for file in os.listdir(data_path):
            if not file.endswith(".npy"):
                continue
            if file[:-4] not in self.VALID_ASSETS:
                continue
            sample[file[:-4]] = np.load(os.path.join(data_path, file))
        sample["name"] = sample_name
        sample["coord"] = sample["coord"].astype(np.float32)
        sample["color"] = sample["color"].astype(np.float32)
        sample["normal"] = sample["normal"].astype(np.float32)

        # Process segmentation: select the first channel then convert.
        seg = sample["segment"][:, 0]
        sample["segment"] = (
            self.convert_index(seg).reshape(-1).astype(np.int32)
        )
        return sample


@DATASETS.register_module()
class ScanNetPPDataset_BASETrain(ScanNetPPDataset_GFS):
    """
    ScanNet++ dataset class for training on base classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_index_mapping = {
            name: idx for idx, name in enumerate(CLASS_LABELS_BASE)
        }
        self.bg_label = -1
        # Create lookup array based on CLASS_LABELS_100.
        self.lookup_array = np.full(
            len(CLASS_LABELS_100), self.bg_label, dtype=np.int32
        )
        for idx, name in enumerate(CLASS_LABELS_100):
            if name in self.base_index_mapping:
                self.lookup_array[idx] = self.base_index_mapping[name]


@DATASETS.register_module()
class ScanNetPPDataset_REGISTrain(ScanNetPPDataset_BASETrain):
    """
    ScanNet++ dataset class for training on novel and base classes after registration.
    Novel instance patches are extracted from registration data and inserted along training scenes.
    """

    def __init__(self, nb_mix_blks=3, k_shot=5, seed=10, full_gt=False, **kwargs):
        super().__init__(**kwargs)
        self.nb_mix_blks = nb_mix_blks
        self.full_gt = bool(full_gt)

        # Load registration set from file.
        regis_file = os.path.join(
            self.data_root, f"scpp_regis_{k_shot}_{seed}.txt"
        )
        self.regis_cls2scans = defaultdict(list)
        with open(regis_file, "r") as f:
            for line in f.read().splitlines():
                class_id, rel_scan = line.split("\t")
                abs_scan = os.path.join(self.data_root, rel_scan)
                self.regis_cls2scans[class_id].append(abs_scan)

        # Define base and novel class names.
        self.base_class_names = list(CLASS_LABELS_BASE)
        self.novel_class_names = [
            item
            for item in CLASS_LABELS_BASE_NOVEL
            if item not in CLASS_LABELS_BASE
        ]

        # Create unified lookup mapping for base and novel classes.
        class_to_idx = {
            name: idx
            for idx, name in enumerate(
                self.base_class_names + self.novel_class_names
            )
        }
        self.regis_lookup_array = np.full(
            len(CLASS_LABELS_100), self.bg_label, dtype=np.int32
        )
        for idx, name in enumerate(CLASS_LABELS_100):
            if name in class_to_idx:
                self.regis_lookup_array[idx] = class_to_idx[name]

        # Novel classes are the keys in the registration set.
        self.novel_classes = list(self.regis_cls2scans.keys())

        # Oracle 模式：用 0-29 的完整映射（原场景的 novel GT 不再被置 -1）。
        # 只为「全监督上界 / 完美过滤上界」实验使用，普通训练保持 BASETrain 的
        # 单 base 映射不变。
        if self.full_gt:
            self.lookup_array = self.regis_lookup_array

    def _get_instances_from_class(self, class_id):
        """
        Retrieve novel instance samples for a given class.
        Returns a list of samples with binary masks.
        """
        instances = []
        for sample_path in self.regis_cls2scans[class_id]:
            class_int = int(class_id)
            sample_name = os.path.basename(sample_path)
            if self.cache:
                instances.append(shared_dict(f"pointcept-{sample_name}"))
                continue
            # 只读 mmap 缓存（/dev/shm/scpp_inst_cache）：不进 Dataset 对象，
            # spawn 序列化体积为 0，所有 worker 共享同一份页缓存。
            _shm = os.path.join("/dev/shm/scpp_inst_cache", sample_name)
            _seg_f = os.path.join(_shm, "seg_%d.npy" % class_int)
            if os.path.exists(_seg_f):
                instances.append(
                    dict(
                        name=sample_name,
                        coord=np.load(os.path.join(_shm, "coord.npy"), mmap_mode="r"),
                        color=np.load(os.path.join(_shm, "color.npy"), mmap_mode="r"),
                        normal=np.load(os.path.join(_shm, "normal.npy"), mmap_mode="r"),
                        segment=np.load(_seg_f, mmap_mode="r"),
                        class_id=np.array(
                            [self.regis_lookup_array[class_int]], dtype=np.int32
                        ),
                    )
                )
                continue
            sample = {}
            for file in os.listdir(sample_path):
                if not file.endswith(".npy"):
                    continue
                if file[:-4] not in self.VALID_ASSETS:
                    continue
                sample[file[:-4]] = np.load(os.path.join(sample_path, file))
            sample["name"] = sample_name
            sample["coord"] = sample["coord"].astype(np.float32)
            sample["color"] = sample["color"].astype(np.float32)
            sample["normal"] = sample["normal"].astype(np.float32)
            seg = sample.pop("segment")
            sample["segment"] = (
                (seg[:, 0] == class_int).reshape(-1).astype(np.int32)
            )
            sample["class_id"] = np.array(
                [self.regis_lookup_array[class_int]], dtype=np.int32
            )
            instances.append(sample)
        return instances

    def _augment_with_novel_instances(
        self,
        train_coords,
        train_feats,
        train_normals,
        train_labels,
        novel_instances,
    ):
        """
        Augment the training point cloud by inserting novel instance patches.
        Returns augmented coordinates, features, normals, labels, and a mask indicating original (1) vs. novel (0) points.
        """
        # Determine scene floor level (minimum z)
        scene_min = np.min(train_coords, axis=0)
        scene_z_min = scene_min[2]

        aug_coords = train_coords
        aug_feats = train_feats
        aug_normals = train_normals
        aug_labels = train_labels
        orig_mask = np.ones(train_coords.shape[0], dtype=np.int32)

        # Define available edges for placement.
        available_edges = ["bottom", "right", "top", "left"]

        for novel in novel_instances:
            inst_coords = novel["coord"]
            inst_feats = novel["color"]
            inst_normals = novel["normal"]

            # Determine instance patch dimensions.
            x_size = np.max(inst_coords[:, 0]) - np.min(inst_coords[:, 0])
            y_size = np.max(inst_coords[:, 1]) - np.min(inst_coords[:, 1])
            half_x, half_y = x_size / 2, y_size / 2

            # Compute center from target points (where segment==1).
            target = inst_coords[novel["segment"] == 1]
            x_center = (np.max(target[:, 0]) + np.min(target[:, 0])) / 2
            y_center = (np.max(target[:, 1]) + np.min(target[:, 1])) / 2

            # Create square mask around the center.
            mask_x = np.abs(inst_coords[:, 0] - x_center) <= half_x / 2
            mask_y = np.abs(inst_coords[:, 1] - y_center) <= half_y / 2
            square_mask = mask_x & mask_y

            # If too few target points, choose a random target point as center.
            if np.sum(square_mask & (novel["segment"] == 1)) < 0.1 * np.sum(
                novel["segment"] == 1
            ):
                rand_idx = np.random.choice(np.where(novel["segment"] == 1)[0])
                x_center, y_center = (
                    inst_coords[rand_idx, 0],
                    inst_coords[rand_idx, 1],
                )
                mask_x = np.abs(inst_coords[:, 0] - x_center) <= half_x / 2
                mask_y = np.abs(inst_coords[:, 1] - y_center) <= half_y / 2
                square_mask = mask_x & mask_y

            patch_coords = inst_coords[square_mask]
            patch_feats = inst_feats[square_mask]
            patch_normals = inst_normals[square_mask]
            patch_labels = novel["segment"][square_mask]
            # Assign target novel label (class_id) where segment==1; else background.
            patch_labels = np.where(patch_labels == 1, novel["class_id"], -1)

            # Select an edge randomly and compute translation.
            sel_edge = random.choice(available_edges)
            available_edges.remove(sel_edge)
            if sel_edge == "bottom":
                base_pt = train_coords[np.argmin(train_coords[:, 1])]
                inst_pt = patch_coords[np.argmax(patch_coords[:, 1])]
            elif sel_edge == "right":
                base_pt = train_coords[np.argmax(train_coords[:, 0])]
                inst_pt = patch_coords[np.argmin(patch_coords[:, 0])]
            elif sel_edge == "top":
                base_pt = train_coords[np.argmax(train_coords[:, 1])]
                inst_pt = patch_coords[np.argmin(patch_coords[:, 1])]
            else:  # left
                base_pt = train_coords[np.argmin(train_coords[:, 0])]
                inst_pt = patch_coords[np.argmax(patch_coords[:, 0])]

            translation = [base_pt[0] - inst_pt[0], base_pt[1] - inst_pt[1], 0]
            translated_patch = patch_coords + translation
            translated_patch[:, 2] += scene_z_min - np.min(
                translated_patch[:, 2]
            )

            aug_coords = np.vstack((aug_coords, translated_patch))
            aug_feats = np.vstack((aug_feats, patch_feats))
            aug_normals = np.vstack((aug_normals, patch_normals))
            aug_labels = np.concatenate((aug_labels, patch_labels))
            orig_mask = np.concatenate(
                (orig_mask, np.zeros(len(translated_patch), dtype=np.int32))
            )

        return aug_coords, aug_feats, aug_normals, aug_labels, orig_mask

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        sample_name = self.get_data_name(idx)
        if self.cache:
            return shared_dict(f"pointcept-{sample_name}")

        sample = {}
        for file in os.listdir(data_path):
            if not file.endswith(".npy"):
                continue
            if file[:-4] not in self.VALID_ASSETS:
                continue
            sample[file[:-4]] = np.load(os.path.join(data_path, file))
        sample["name"] = sample_name
        sample["coord"] = sample["coord"].astype(np.float32)
        sample["color"] = sample["color"].astype(np.float32)
        sample["normal"] = sample["normal"].astype(np.float32)

        seg = sample.pop("segment")
        # Convert segmentation labels using the GFS conversion.
        sample["segment"] = (
            self.convert_index(seg[:, 0]).reshape(-1).astype(np.int32)
        )

        # Randomly sample novel classes and select one instance per class.
        selected_novels = random.sample(self.novel_classes, self.nb_mix_blks)
        novel_instances = []
        for novel_cls in selected_novels:
            inst_samples = self._get_instances_from_class(novel_cls)
            assert len(inst_samples) > 0
            novel_instances.extend(random.sample(inst_samples, 1))

        # Augment training point cloud with the novel instance patches.
        (
            sample["coord"],
            sample["color"],
            sample["normal"],
            sample["segment"],
            sample["mask"],
        ) = self._augment_with_novel_instances(
            sample["coord"],
            sample["color"],
            sample["normal"],
            sample["segment"],
            novel_instances,
        )
        return sample


@DATASETS.register_module()
class ScanNetPPDataset_TEST(ScanNetPPDataset_GFS):
    """
    ScanNet++ dataset class for testing on base and novel classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_class_names = list(CLASS_LABELS_BASE)
        self.novel_class_names = [
            item
            for item in CLASS_LABELS_BASE_NOVEL
            if item not in CLASS_LABELS_BASE
        ]
        class_to_idx = {
            name: idx
            for idx, name in enumerate(
                self.base_class_names + self.novel_class_names
            )
        }
        self.lookup_array = np.full(
            len(CLASS_LABELS_100), self.bg_label, dtype=np.int32
        )
        for idx, name in enumerate(CLASS_LABELS_100):
            if name in class_to_idx:
                self.lookup_array[idx] = class_to_idx[name]


@DATASETS.register_module()
class ScanNetPPDataset_REGIS(ScanNetPPDataset_GFS):
    """
    ScanNet++ dataset class of registration set.
    Constructs a registration data list based on a registration file.
    If the registration file does not exist, it is generated automatically using a class-to-scans mapping.
    """

    def __init__(self, seed=None, k_shot=None, **kwargs):
        super().__init__(**kwargs)
        self.base_class_names = list(CLASS_LABELS_BASE)
        self.novel_class_names = [
            item
            for item in CLASS_LABELS_BASE_NOVEL
            if item not in CLASS_LABELS_BASE
        ]

        self.orgid2name = {
            i: name.strip() for i, name in enumerate(CLASS_LABELS_100)
        }
        self.name2orgid = {
            name.strip(): i for i, name in enumerate(CLASS_LABELS_100)
        }
        self.train_classes = [
            self.name2orgid[name] for name in self.base_class_names
        ]
        self.test_classes = [
            self.name2orgid[name] for name in self.novel_class_names
        ]
        self.all_classes_orgid = self.train_classes + self.test_classes

        cls_to_idx = {
            name: idx
            for idx, name in enumerate(
                self.base_class_names + self.novel_class_names
            )
        }
        self.lookup_array = np.full(
            len(CLASS_LABELS_100), self.bg_label, dtype=np.int32
        )
        for idx, name in enumerate(CLASS_LABELS_100):
            if name in cls_to_idx:
                self.lookup_array[idx] = cls_to_idx[name]

        self.k_shot = k_shot
        self._create_data_list(seed)

    def _create_data_list(self, seed):
        """
        Create the registration data list from a registration file.
        If the file does not exist, generate it using a class-to-scans mapping.
        The registration file stores scan paths as relative paths (relative to self.data_root).
        """
        regis_file = os.path.join(
            self.data_root, f"scpp_regis_{self.k_shot}_{seed}.txt"
        )
        if not os.path.exists(regis_file) and comm.is_main_process():
            class2scans_file = os.path.join(
                self.data_root, "scpp_class2trainscans.pkl"
            )
            if not os.path.exists(class2scans_file):
                self._create_class2scans(class2scans_file)
            with open(class2scans_file, "rb") as f:
                self.class2scans = pickle.load(f)
            np.random.seed(seed)
            random.seed(seed)
            open(regis_file, "w").close()

            used_scans = []
            for novel_cls in self.test_classes:
                available_scans = [
                    scan
                    for scan in self.class2scans[novel_cls]
                    if scan not in used_scans
                ]
                selected_scans = np.random.choice(
                    available_scans, self.k_shot, replace=False
                )
                used_scans.extend(selected_scans)
                with open(regis_file, "a") as f:
                    # Write the novel class and the relative scan path.
                    for rel_scan in selected_scans:
                        f.write(f"{novel_cls}\t{rel_scan}\n")
        comm.synchronize()
        self.data_list = []
        with open(regis_file, "r") as f:
            for line in f.read().splitlines():
                class_id, rel_scan = line.split("\t")
                abs_scan = os.path.join(self.data_root, rel_scan)
                self.data_list.append((class_id, abs_scan))

    def _create_class2scans(self, class2scans_file):
        """
        Build the mapping from class IDs to scan folders.
        Filters out scans with too few labeled points.
        The mapping is stored with folder paths relative to self.data_root.
        """
        min_pts = 100
        class2scans = {cls: [] for cls in self.all_classes_orgid}
        for folder in glob.glob(os.path.join(self.data_root, self.split, "*")):
            scan_name = os.path.basename(folder)
            labels = np.load(os.path.join(folder, "segment.npy"))[:, 0]
            classes = np.unique(labels)
            classes = classes[classes != -1]
            print(
                f"{scan_name} | shape: {labels.shape} | classes: {list(classes)}"
            )
            for cls in classes:
                if cls not in self.all_classes_orgid:
                    continue
                count = np.count_nonzero(labels == cls)
                if count > min_pts:
                    rel_folder = os.path.relpath(folder, self.data_root)
                    class2scans[cls].append(rel_folder)
        print("==== Class-to-scans mapping completed ====")
        for cls in self.all_classes_orgid:
            assert (
                len(class2scans[cls]) > 0
            ), f"Class {cls} ({self.orgid2name[cls]}) has no data."
            print(
                f"\t Class {cls} | min_pts: {min_pts} | Name: {self.orgid2name[cls]} | Scans: {len(class2scans[cls])}"
            )
        with open(class2scans_file, "wb") as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)

    def get_data(self, idx):
        """
        Load and process a registration training sample.
        Converts segmentation labels into a binary mask for the specified class.
        """
        class_id, data_path = self.data_list[idx % len(self.data_list)]
        class_id = int(class_id)
        sample_name = os.path.basename(data_path)
        if self.cache:
            return shared_dict(f"pointcept-{sample_name}")

        sample = {}
        for file in os.listdir(data_path):
            if file.endswith(".npy") and file[:-4] in self.VALID_ASSETS:
                sample[file[:-4]] = np.load(os.path.join(data_path, file))
        sample["name"] = sample_name
        sample["coord"] = sample["coord"].astype(np.float32)
        sample["color"] = sample["color"].astype(np.float32)
        sample["normal"] = sample["normal"].astype(np.float32)

        seg = sample.pop("segment")
        seg = (seg[:, 0] == class_id).reshape(-1).astype(np.int32)
        seg[seg == 0] = self.bg_label
        seg[seg == 1] = self.lookup_array[class_id]
        sample["segment"] = seg
        return sample
