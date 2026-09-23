"""
ScanNet200 / ScanNet Efficient Generalized Few-Shot Dataset

Author: Zhaochong An (anzhaochong@outlook.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
import pointcept.utils.comm as comm

from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
import pickle
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
    CLASS_LABELS_200,  # scannet200
    CLASS_LABELS_BASE,
    CLASS_LABELS_BASE_NOVEL,
    CLASS_LABELS_20,  # scannetv2
    CLASS20_LABELS_BASE,
    CLASS20_LABELS_BASE_NOVEL,
)
import random
from collections import defaultdict
from copy import deepcopy


@DATASETS.register_module()
class ScanNetDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment20",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        lr_file=None,
        la_file=None,
        **kwargs,
    ):
        self.lr = (
            np.loadtxt(lr_file, dtype=str) if lr_file is not None else None
        )
        self.la = torch.load(la_file, weights_only=False) if la_file is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list

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
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment20" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment20").reshape([-1]).astype(np.int32)
            )
        elif "segment200" in data_dict.keys():
            data_dict["segment"] = (
                data_dict.pop("segment200").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = (
                data_dict.pop("instance").reshape([-1]).astype(np.int32)
            )
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(data_dict["segment"], dtype=bool)
            mask[sampled_index] = False
            data_dict["segment"][mask] = self.ignore_index
            data_dict["sampled_index"] = sampled_index
        return data_dict


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment200",
        "instance",
    ]
    class2id = np.array(VALID_CLASS_IDS_200)


@DATASETS.register_module()
class ScanNet200Dataset_GFS(ScanNetDataset):
    """
    ScanNet200 dataset prototype for generalized few-shot learning.
    Loads the 'segment200' asset and converts labels using a lookup array.
    """

    VALID_ASSETS = ["coord", "color", "normal", "segment200"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bg_label = -1
        # Initialize lookup array (to be updated by child classes)
        self.lookup_array = np.full(
            len(CLASS_LABELS_200), self.bg_label, dtype=np.int32
        )

    def convert_index(self, seg_array):
        """Convert segmentation array from 200-class labels to unified labels."""
        converted = np.full(seg_array.shape, self.bg_label)
        valid_mask = seg_array != -1
        converted[valid_mask] = self.lookup_array[seg_array[valid_mask]]
        return converted

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            return shared_dict(f"pointcept-{name}")

        data_dict = {}
        for asset in os.listdir(data_path):
            if asset.endswith(".npy") and asset[:-4] in self.VALID_ASSETS:
                data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)
        data_dict["segment"] = data_dict.pop("segment200")
        data_dict["segment"] = (
            self.convert_index(data_dict["segment"])
            .reshape([-1])
            .astype(np.int32)
        )
        return data_dict


@DATASETS.register_module()
class ScanNet200Dataset_BASETrain(ScanNet200Dataset_GFS):
    """
    ScanNet200 dataset class for training on base classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Mapping from base class names to indices
        self.base_index_mapping = {
            name: idx for idx, name in enumerate(CLASS_LABELS_BASE)
        }
        # Create lookup array converting from CLASS_LABELS_200 to base indices
        self.lookup_array = np.full(
            len(CLASS_LABELS_200), self.bg_label, dtype=np.int32
        )
        for idx, name in enumerate(CLASS_LABELS_200):
            if name in self.base_index_mapping:
                self.lookup_array[idx] = self.base_index_mapping[name]


@DATASETS.register_module()
class ScanNet200Dataset_REGISTrain(ScanNet200Dataset_BASETrain):
    """
    ScanNet200 dataset class for training on novel and base classes after registration.
    Novel instance patches are extracted from registration data and inserted along training scenes.
    """

    def __init__(self, nb_mix_blks=3, k_shot=5, seed=10, **kwargs):
        super().__init__(**kwargs)
        self.nb_mix_blks = nb_mix_blks

        # Build registration set from file
        reg_file = os.path.join(
            self.data_root, f"sc200_regis_{k_shot}_{seed}.txt"
        )
        self.regis_cls2scans = defaultdict(list)
        with open(reg_file, "r") as f:
            for line in f.read().splitlines():
                class_id, rel_scan = line.split("\t")
                abs_scan = os.path.join(self.data_root, rel_scan)
                self.regis_cls2scans[class_id].append(abs_scan)

        # Define base and novel class names
        self.base_class_names = list(CLASS_LABELS_BASE)
        self.novel_class_names = [
            item
            for item in CLASS_LABELS_BASE_NOVEL
            if item not in CLASS_LABELS_BASE
        ]

        # Build lookup mapping for base+novel classes
        cls_mapping = {
            name: idx
            for idx, name in enumerate(
                self.base_class_names + self.novel_class_names
            )
        }
        self.regis_lookup_array = np.full(
            len(CLASS_LABELS_200), self.bg_label, dtype=np.int32
        )
        for idx, name in enumerate(CLASS_LABELS_200):
            if name in cls_mapping:
                self.regis_lookup_array[idx] = cls_mapping[name]

        # Novel classes are the unique keys in the registration set
        self.novel_classes = list(self.regis_cls2scans.keys())

    def _get_instances_from_class(self, class_id):
        """
        Retrieve novel instance samples for a given class.
        Returns a list of data dictionaries with binary novel masks.
        """
        instances = []
        for path in self.regis_cls2scans[class_id]:
            class_int = int(class_id)
            sample_name = os.path.basename(path)
            if self.cache:
                return shared_dict(f"pointcept-{sample_name}")
            sample_data = {}
            for asset in os.listdir(path):
                if asset.endswith(".npy") and asset[:-4] in self.VALID_ASSETS:
                    sample_data[asset[:-4]] = np.load(
                        os.path.join(path, asset)
                    )
            sample_data["name"] = sample_name
            sample_data["coord"] = sample_data["coord"].astype(np.float32)
            sample_data["color"] = sample_data["color"].astype(np.float32)
            sample_data["normal"] = sample_data["normal"].astype(np.float32)
            # Convert segment200 to a binary mask for the novel class
            sample_data["segment"] = sample_data.pop("segment200")
            sample_data["segment"] = (
                (sample_data["segment"] == class_int)
                .reshape([-1])
                .astype(np.int32)
            )
            sample_data["class_id"] = np.array(
                [self.regis_lookup_array[class_int]], dtype=np.int32
            )
            instances.append(sample_data)
        return instances

    def _augment_with_novel_samples(
        self, coords, features, normals, labels, novel_samples
    ):
        """
        Augment the training point cloud by inserting novel instance patches.
        Returns augmented data and a mask indicating original (1) vs. novel (0) points.
        """
        train_min = np.min(coords, axis=0)
        train_z_min = train_min[2]

        # Initialize augmented data with original training points
        aug_coords = coords
        aug_features = features
        aug_normals = normals
        aug_labels = labels
        orig_mask = np.ones(
            coords.shape[0], dtype=np.int32
        )  # 1 indicates original training points

        # Define available scene edge options for placement
        edges = ["bottom", "right", "top", "left"]

        for sample in novel_samples:
            samp_coords = sample["coord"]
            samp_features = sample["color"]
            samp_normals = sample["normal"]

            # Determine patch dimensions from the novel sample
            x_size = np.max(samp_coords[:, 0]) - np.min(samp_coords[:, 0])
            y_size = np.max(samp_coords[:, 1]) - np.min(samp_coords[:, 1])
            half_x = x_size / 2
            half_y = y_size / 2

            # Compute center based on target foreground points where segment == 1
            target = samp_coords[sample["segment"] == 1]
            x_center = (np.max(target[:, 0]) + np.min(target[:, 0])) / 2
            y_center = (np.max(target[:, 1]) + np.min(target[:, 1])) / 2

            # Create a square mask around the computed center
            mask_x = np.abs(samp_coords[:, 0] - x_center) <= half_x / 2
            mask_y = np.abs(samp_coords[:, 1] - y_center) <= half_y / 2
            square_mask = mask_x & mask_y

            # If too few points are selected, pick a random target point as new center
            if np.sum(square_mask & (sample["segment"] == 1)) < 0.1 * np.sum(
                sample["segment"] == 1
            ):
                rand_idx = np.random.choice(
                    np.where(sample["segment"] == 1)[0]
                )
                x_center, y_center = (
                    samp_coords[rand_idx, 0],
                    samp_coords[rand_idx, 1],
                )
                mask_x = np.abs(samp_coords[:, 0] - x_center) <= half_x / 2
                mask_y = np.abs(samp_coords[:, 1] - y_center) <= half_y / 2
                square_mask = mask_x & mask_y

            # Extract the patch from the novel sample
            patch_coords = samp_coords[square_mask]
            patch_features = samp_features[square_mask]
            patch_normals = samp_normals[square_mask]
            patch_labels = sample["segment"][square_mask]
            # Assign the novel class label to patch points (1 -> class_id; else background)
            patch_labels = np.where(patch_labels == 1, sample["class_id"], -1)

            # Randomly select an edge and compute translation vector
            sel_edge = random.choice(edges)
            edges.remove(sel_edge)
            if sel_edge == "bottom":
                base_point = coords[np.argmin(coords[:, 1])]  # Lowest
                inst_point = patch_coords[
                    np.argmax(patch_coords[:, 1])
                ]  # Highest
            elif sel_edge == "right":
                base_point = coords[np.argmax(coords[:, 0])]  # Rightmost
                inst_point = patch_coords[
                    np.argmin(patch_coords[:, 0])
                ]  # Leftmost
            elif sel_edge == "top":
                base_point = coords[np.argmax(coords[:, 1])]  # Highest
                inst_point = patch_coords[
                    np.argmin(patch_coords[:, 1])
                ]  # Lowest
            else:  # left edge
                base_point = coords[np.argmin(coords[:, 0])]  # Leftmost
                inst_point = patch_coords[
                    np.argmax(patch_coords[:, 0])
                ]  # Rightmost

            translation = [
                base_point[0] - inst_point[0],
                base_point[1] - inst_point[1],
                0,
            ]

            # Translate patch and adjust z-coordinate to align with scene floor
            translated_patch = patch_coords + translation
            translated_patch[:, 2] += train_z_min - np.min(
                translated_patch[:, 2]
            )

            # Append the translated patch to the augmented cloud
            aug_coords = np.vstack((aug_coords, translated_patch))
            aug_features = np.vstack((aug_features, patch_features))
            aug_normals = np.vstack((aug_normals, patch_normals))
            aug_labels = np.concatenate((aug_labels, patch_labels))
            # For the augmented points, mark them with 0 in the original mask
            orig_mask = np.concatenate(
                (orig_mask, np.zeros(len(translated_patch), dtype=np.int32))
            )

        return aug_coords, aug_features, aug_normals, aug_labels, orig_mask

    def get_data(self, idx):
        """
        Load primary point cloud, convert segmentation labels,
        sample novel instances, and augment the point cloud.
        """
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            return shared_dict(f"pointcept-{name}")

        data = {}
        for asset in os.listdir(data_path):
            if asset.endswith(".npy") and asset[:-4] in self.VALID_ASSETS:
                data[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data["name"] = name
        data["coord"] = data["coord"].astype(np.float32)
        data["color"] = data["color"].astype(np.float32)
        data["normal"] = data["normal"].astype(np.float32)
        data["segment"] = data.pop("segment200")
        data["segment"] = (
            self.convert_index(data["segment"]).reshape([-1]).astype(np.int32)
        )

        # Sample novel classes and obtain one instance per class
        selected = random.sample(self.novel_classes, self.nb_mix_blks)
        novel_samples = []
        for cls in selected:
            samples = self._get_instances_from_class(cls)
            assert len(samples) > 0
            novel_samples.extend(random.sample(samples, 1))

        # Augment training cloud with novel instance patches
        augmented = self._augment_with_novel_samples(
            data["coord"],
            data["color"],
            data["normal"],
            data["segment"],
            novel_samples,
        )
        (
            data["coord"],
            data["color"],
            data["normal"],
            data["segment"],
            data["mask"],
        ) = augmented
        return data


@DATASETS.register_module()
class ScanNet200Dataset_TEST(ScanNet200Dataset_GFS):
    """
    ScanNet200 dataset class for testing on base and novel classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_class_names = list(CLASS_LABELS_BASE)
        # Determine novel classes as those in CLASS_LABELS_BASE_NOVEL but not in base
        self.novel_class_names = [
            label
            for label in CLASS_LABELS_BASE_NOVEL
            if label not in CLASS_LABELS_BASE
        ]
        # Create a mapping from class name to new index for base and novel classes
        class_to_idx_map = {
            name: idx
            for idx, name in enumerate(
                self.base_class_names + self.novel_class_names
            )
        }
        # Build a lookup array to convert original 200 labels to the GFS label indices
        self.lookup_array = np.full(
            len(CLASS_LABELS_200), self.bg_label, dtype=np.int32
        )
        for idx, label in enumerate(CLASS_LABELS_200):
            if label in class_to_idx_map:
                self.lookup_array[idx] = class_to_idx_map[label]


@DATASETS.register_module()
class ScanNet200Dataset_TEST_vis(ScanNet200Dataset_TEST):
    """
    ScanNet200 dataset class for visualization.
    Extends the test dataset by preparing additional data for visualization,
    including coordinates and color information.
    """

    def prepare_test_data(self, idx):
        """
        Prepare test data with visualization details.

        Args:
            idx (int): Index of the data sample.

        Returns:
            dict: Dictionary containing segmentation, name, coord, color,
                  and a list of transformed fragments.
        """
        # Load and transform the sample
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        result_dict = {
            "segment": data_dict.pop("segment"),
            "name": data_dict.pop("name"),
            "coord": data_dict["coord"],
            "color": data_dict["color"],
        }
        if "origin_segment" in data_dict:
            assert "inverse" in data_dict
            result_dict["origin_segment"] = data_dict.pop("origin_segment")
            result_dict["inverse"] = data_dict.pop("inverse")

        # Apply augmentation transforms
        data_dict_list = [
            aug(deepcopy(data_dict)) for aug in self.aug_transform
        ]

        # Build fragments via voxelization and cropping
        fragment_list = []
        for data in data_dict_list:
            if self.test_voxelize is not None:
                data_part_list = self.test_voxelize(data)
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]
            for data_part in data_part_list:
                if self.test_crop is not None:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                fragment_list += data_part

        # Apply any post-processing transforms to all fragments
        result_dict["fragment_list"] = [
            self.post_transform(fragment) for fragment in fragment_list
        ]
        return result_dict


@DATASETS.register_module()
class ScanNet200Dataset_REGIS(ScanNet200Dataset_GFS):
    """
    ScanNet200 dataset class of registration set.
    Constructs a registration data list based on a registration file.
    If the registration file does not exist, it is generated automatically using a class-to-scans mapping.
    """

    def __init__(self, seed=None, k_shot=None, **kwargs):
        super().__init__(**kwargs)
        self.base_class_names = list(CLASS_LABELS_BASE)
        self.novel_class_names = [
            label
            for label in CLASS_LABELS_BASE_NOVEL
            if label not in CLASS_LABELS_BASE
        ]
        # Mapping from original class IDs to names
        self.orgid2name = {
            i: name.strip() for i, name in enumerate(CLASS_LABELS_200)
        }
        self.name2orgid = {
            name.strip(): i for i, name in enumerate(CLASS_LABELS_200)
        }
        # Define training (base) and testing (novel) classes based on original IDs
        self.train_classes = [
            self.name2orgid[name] for name in self.base_class_names
        ]
        self.test_classes = [
            self.name2orgid[name] for name in self.novel_class_names
        ]
        self.all_classes_orgid = self.train_classes + self.test_classes

        # Build lookup array to convert original 200-class labels to GFS labels
        class_to_idx_map = {
            name: idx
            for idx, name in enumerate(
                self.base_class_names + self.novel_class_names
            )
        }
        self.lookup_array = np.full(
            len(CLASS_LABELS_200), self.bg_label, dtype=np.int32
        )
        for idx, label in enumerate(CLASS_LABELS_200):
            if label in class_to_idx_map:
                self.lookup_array[idx] = class_to_idx_map[label]

        self.k_shot = k_shot
        self._create_data_list(seed)

    def _create_data_list(self, seed):
        """
        Create the registration data list from a registration file.
        If the registration file does not exist, generate it using a class-to-scans mapping.
        """
        registration_file = os.path.join(
            self.data_root, f"sc200_regis_{self.k_shot}_{seed}.txt"
        )
        if not os.path.exists(registration_file) and comm.is_main_process():
            class2scans_file = os.path.join(
                self.data_root, "sc200_class2trainscans.pkl"
            )
            if not os.path.exists(class2scans_file):
                self._create_class2scans(class2scans_file)
            with open(class2scans_file, "rb") as f:
                self.class2scans = pickle.load(f)

            np.random.seed(seed)
            random.seed(seed)
            # Create the registration file.
            open(registration_file, "w").close()

            used_scans = []
            # For each novel class, select k_shot unique scans.
            for novel_class in self.test_classes:
                available_scans = [
                    scan
                    for scan in self.class2scans[novel_class]
                    if scan not in used_scans
                ]
                selected_scans = np.random.choice(
                    available_scans, self.k_shot, replace=False
                )
                used_scans.extend(selected_scans)
                with open(registration_file, "a") as f:
                    for rel_scan in selected_scans:
                        f.write(f"{novel_class}\t{rel_scan}\n")
        comm.synchronize()

        # Load data list from registration file and convert relative paths to absolute paths.
        self.data_list = []
        with open(registration_file, "r") as f:
            for line in f.read().splitlines():
                class_id_str, rel_scan = line.split("\t")
                abs_scan = os.path.join(self.data_root, rel_scan)
                self.data_list.append((class_id_str, abs_scan))

    def _create_class2scans(self, class2scans_file):
        """
        Build a mapping from class ID to scan folders, filtering out scans with too few points.
        Paths are stored relative to self.data_root.
        """
        min_points = 100  # Minimum number of points to consider
        class2scans = {class_id: [] for class_id in self.all_classes_orgid}
        for scan_folder in glob.glob(
            os.path.join(self.data_root, self.split, "*")
        ):
            scan_name = os.path.basename(scan_folder)
            labels_path = os.path.join(scan_folder, "segment200.npy")
            labels = np.load(labels_path)
            unique_classes = np.unique(labels)
            unique_classes = unique_classes[
                unique_classes != -1
            ]  # Exclude background
            print(
                f"{scan_name} | shape: {labels.shape} | classes: {list(unique_classes)}"
            )
            for class_id in unique_classes:
                if class_id not in self.all_classes_orgid:
                    continue
                point_count = np.count_nonzero(labels == class_id)
                if point_count > min_points:
                    # Save the scan folder as a relative path.
                    rel_folder = os.path.relpath(scan_folder, self.data_root)
                    class2scans[class_id].append(rel_folder)

        print("==== Class-to-scans mapping completed ====")
        for class_id in self.all_classes_orgid:
            assert (
                len(class2scans[class_id]) > 0
            ), f"Class {class_id} ({self.orgid2name[class_id]}) has no data."
            print(
                f"\t Class {class_id} | min_points: {min_points} | Name: {self.orgid2name[class_id]} | Scans: {len(class2scans[class_id])}"
            )

        with open(class2scans_file, "wb") as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)

    def get_data(self, idx):
        """
        Load and process a registration sample.
        Converts segmentation labels into a binary mask for the specified class.
        """
        class_id, data_path = self.data_list[idx % len(self.data_list)]
        class_id = int(class_id)
        name = os.path.basename(data_path)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        for asset in os.listdir(data_path):
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["coord"] = data_dict["coord"].astype(np.float32)
        data_dict["color"] = data_dict["color"].astype(np.float32)
        data_dict["normal"] = data_dict["normal"].astype(np.float32)

        # Process segmentation labels: create binary mask for the current class.
        seg = data_dict.pop("segment200")
        seg = (seg == class_id).reshape([-1]).astype(np.int32)
        seg[seg == 0] = self.bg_label
        seg[seg == 1] = self.lookup_array[class_id]
        data_dict["segment"] = seg

        return data_dict


@DATASETS.register_module()
class ScanNetDataset_GFS(ScanNetDataset):
    """
    ScanNet dataset prototype for generalized few-shot learning.
    Loads the 'segment20' asset and converts labels using a lookup array.
    """

    VALID_ASSETS = ["coord", "color", "normal", "segment20"]

    def __init__(self, **kwargs):
        self.bg_label = -1
        # Initialize lookup array for fast label conversion.
        self.lookup_array = np.full(
            len(CLASS_LABELS_20), self.bg_label, dtype=np.int32
        )
        super().__init__(**kwargs)

    def convert_index(self, seg_array):
        """
        Convert the segmentation array using the lookup array.
        """
        converted = np.full(seg_array.shape, self.bg_label)
        valid = seg_array != -1
        converted[valid] = self.lookup_array[seg_array[valid]]
        return converted

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        sample_name = self.get_data_name(idx)
        if self.cache:
            return shared_dict(f"pointcept-{sample_name}")

        sample = {}
        for file in os.listdir(data_path):
            if file.endswith(".npy") and file[:-4] in self.VALID_ASSETS:
                sample[file[:-4]] = np.load(os.path.join(data_path, file))
        sample["name"] = sample_name
        # Convert to float32 for consistency
        sample["coord"] = sample["coord"].astype(np.float32)
        sample["color"] = sample["color"].astype(np.float32)
        sample["normal"] = sample["normal"].astype(np.float32)

        # Process segmentation: load 'segment20', convert and reshape.
        seg = sample.pop("segment20")
        sample["segment"] = (
            self.convert_index(seg).reshape(-1).astype(np.int32)
        )
        return sample


@DATASETS.register_module()
class ScanNetDataset_BASETrain(ScanNetDataset_GFS):
    """
    ScanNet dataset class for training on base classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Build a mapping from base class names to new indices.
        self.base_index_mapping = {
            name: idx for idx, name in enumerate(CLASS20_LABELS_BASE)
        }
        # Initialize lookup array for converting full 20-class labels to base indices.
        self.lookup_array = np.full(
            len(CLASS_LABELS_20), self.bg_label, dtype=np.int32
        )
        for idx, label in enumerate(CLASS_LABELS_20):
            if label in self.base_index_mapping:
                self.lookup_array[idx] = self.base_index_mapping[label]


@DATASETS.register_module()
class ScanNetDataset_REGISTrain(ScanNetDataset_BASETrain):
    """
    ScanNet dataset class for training on novel and base classes after registration.
    Novel instance patches are extracted from registration data and inserted along training scenes.
    """

    def __init__(self, nb_mix_blks=3, k_shot=5, seed=10, **kwargs):
        super().__init__(**kwargs)
        self.nb_mix_blks = nb_mix_blks
        reg_file = os.path.join(
            self.data_root, f"sc_regis_{k_shot}_{seed}.txt"
        )
        self.regis_cls2scans = defaultdict(list)
        with open(reg_file, "r") as f:
            for line in f.read().splitlines():
                cls_id, rel_scan = line.split("\t")
                abs_scan = os.path.join(self.data_root, rel_scan)
                self.regis_cls2scans[cls_id].append(abs_scan)

        self.base_class_names = list(CLASS20_LABELS_BASE)
        self.novel_class_names = [
            label
            for label in CLASS20_LABELS_BASE_NOVEL
            if label not in CLASS20_LABELS_BASE
        ]
        # Build lookup mapping for unified (base + novel) label space.
        cls_to_index = {
            name: idx
            for idx, name in enumerate(
                self.base_class_names + self.novel_class_names
            )
        }
        self.regis_lookup_array = np.full(
            len(CLASS_LABELS_20), self.bg_label, dtype=np.int32
        )
        for idx, label in enumerate(CLASS_LABELS_20):
            if label in cls_to_index:
                self.regis_lookup_array[idx] = cls_to_index[label]

        # Novel classes are the unique novel class IDs in the registration set.
        self.novel_classes = list(self.regis_cls2scans.keys())

    def _get_instances_from_class(self, class_id):
        """
        Retrieve novel instance samples for a given class.
        Returns a list of samples with binary masks.
        """
        instances = []
        for folder in self.regis_cls2scans[class_id]:
            class_int = int(class_id)
            sample_name = os.path.basename(folder)
            if self.cache:
                return shared_dict(f"pointcept-{sample_name}")
            sample = {}
            for file in os.listdir(folder):
                if file.endswith(".npy") and file[:-4] in self.VALID_ASSETS:
                    sample[file[:-4]] = np.load(os.path.join(folder, file))
            sample["name"] = sample_name
            sample["coord"] = sample["coord"].astype(np.float32)
            sample["color"] = sample["color"].astype(np.float32)
            sample["normal"] = sample["normal"].astype(np.float32)
            seg = sample.pop("segment20")
            sample["segment"] = (seg == class_int).reshape(-1).astype(np.int32)
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
        Returns the augmented coordinates, features, normals, labels, and a mask indicating
        original (1) versus novel (0) points.
        """
        # Determine the floor level (minimum z)
        scene_min = np.min(train_coords, axis=0)
        scene_z_min = scene_min[2]

        aug_coords = train_coords
        aug_feats = train_feats
        aug_normals = train_normals
        aug_labels = train_labels
        orig_mask = np.ones(
            train_coords.shape[0], dtype=np.int32
        )  # 1 indicates original points

        available_edges = ["bottom", "right", "top", "left"]

        for novel in novel_instances:
            inst_coords = novel["coord"]
            inst_feats = novel["color"]
            inst_normals = novel["normal"]

            # Compute the instance patch dimensions.
            x_size = np.max(inst_coords[:, 0]) - np.min(inst_coords[:, 0])
            y_size = np.max(inst_coords[:, 1]) - np.min(inst_coords[:, 1])
            half_x, half_y = x_size / 2, y_size / 2

            # Determine center from target points (where segment==1).
            target = inst_coords[novel["segment"] == 1]
            x_center = (np.max(target[:, 0]) + np.min(target[:, 0])) / 2
            y_center = (np.max(target[:, 1]) + np.min(target[:, 1])) / 2

            # Create a square mask around the computed center.
            mask_x = np.abs(inst_coords[:, 0] - x_center) <= half_x / 2
            mask_y = np.abs(inst_coords[:, 1] - y_center) <= half_y / 2
            square_mask = mask_x & mask_y

            # If too few target points are in the square, choose a random target point as center.
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
            # Set patch points with target label to the novel class id; others to background.
            patch_labels = np.where(patch_labels == 1, novel["class_id"], -1)

            # Choose an edge randomly and compute translation.
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
            if file.endswith(".npy") and file[:-4] in self.VALID_ASSETS:
                sample[file[:-4]] = np.load(os.path.join(data_path, file))
        sample["name"] = sample_name
        sample["coord"] = sample["coord"].astype(np.float32)
        sample["color"] = sample["color"].astype(np.float32)
        sample["normal"] = sample["normal"].astype(np.float32)

        # Load and convert segmentation labels.
        seg = sample.pop("segment20")
        sample["segment"] = (
            self.convert_index(seg).reshape(-1).astype(np.int32)
        )

        # Sample novel classes and select one instance per class.
        selected_novel = random.sample(self.novel_classes, self.nb_mix_blks)
        novel_samples = []
        for cls in selected_novel:
            insts = self._get_instances_from_class(cls)
            assert len(insts) > 0
            novel_samples.extend(random.sample(insts, 1))

        # Augment the training cloud with novel instance patches.
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
            novel_samples,
        )
        return sample


@DATASETS.register_module()
class ScanNetDataset_TEST(ScanNetDataset_GFS):
    """
    ScanNet dataset class for testing on base and novel classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define base and novel class names.
        self.base_class_names = list(CLASS20_LABELS_BASE)
        self.novel_class_names = [
            label
            for label in CLASS20_LABELS_BASE_NOVEL
            if label not in CLASS20_LABELS_BASE
        ]
        # Create mapping for unified labels.
        cls_to_idx = {
            name: idx
            for idx, name in enumerate(
                self.base_class_names + self.novel_class_names
            )
        }
        self.lookup_array = np.full(
            len(CLASS_LABELS_20), self.bg_label, dtype=np.int32
        )
        for idx, label in enumerate(CLASS_LABELS_20):
            if label in cls_to_idx:
                self.lookup_array[idx] = cls_to_idx[label]


@DATASETS.register_module()
class ScanNetDataset_REGIS(ScanNetDataset_GFS):
    """
    ScanNet dataset class of registration set.
    Constructs a registration data list based on a registration file.
    If the registration file does not exist, it is generated automatically using a class-to-scans mapping.
    """

    def __init__(self, seed=None, k_shot=None, **kwargs):
        super().__init__(**kwargs)
        # Ensure base and novel labels are subsets of CLASS_LABELS_20.
        assert set(CLASS20_LABELS_BASE).issubset(CLASS_LABELS_20)
        assert set(CLASS20_LABELS_BASE_NOVEL).issubset(CLASS_LABELS_20)

        self.base_class_names = list(CLASS20_LABELS_BASE)
        self.novel_class_names = [
            label
            for label in CLASS20_LABELS_BASE_NOVEL
            if label not in CLASS20_LABELS_BASE
        ]

        # Create mappings between original class IDs and names.
        self.orgid2name = {
            i: name.strip() for i, name in enumerate(CLASS_LABELS_20)
        }
        self.name2orgid = {
            name.strip(): i for i, name in enumerate(CLASS_LABELS_20)
        }
        self.train_classes = [
            self.name2orgid[name] for name in self.base_class_names
        ]
        self.test_classes = [
            self.name2orgid[name] for name in self.novel_class_names
        ]
        self.all_classes_orgid = self.train_classes + self.test_classes

        # Build a lookup array mapping original 20-class labels to unified labels.
        cls_to_idx = {
            name: idx
            for idx, name in enumerate(
                self.base_class_names + self.novel_class_names
            )
        }
        self.lookup_array = np.full(
            len(CLASS_LABELS_20), self.bg_label, dtype=np.int32
        )
        for idx, label in enumerate(CLASS_LABELS_20):
            if label in cls_to_idx:
                self.lookup_array[idx] = cls_to_idx[label]

        self.k_shot = k_shot
        self._create_data_list(seed)

    def _create_data_list(self, seed):
        """
        Create the data list from a registration file.
        If the registration file does not exist, generate it using a class-to-scans mapping.
        The registration file stores paths relative to self.data_root.
        """
        reg_file = os.path.join(
            self.data_root, f"sc_regis_{self.k_shot}_{seed}.txt"
        )
        if not os.path.exists(reg_file) and comm.is_main_process():
            class2scans_file = os.path.join(
                self.data_root, "sc_class2trainscans.pkl"
            )
            if not os.path.exists(class2scans_file):
                self._create_class2scans(class2scans_file)
            with open(class2scans_file, "rb") as f:
                self.class2scans = pickle.load(f)

            np.random.seed(seed)
            random.seed(seed)
            # Create (or clear) the registration file.
            open(reg_file, "w").close()

            used_scans = []
            # For each novel class, select k_shot unique scans.
            for novel_cls in self.test_classes:
                available = [
                    scan
                    for scan in self.class2scans[novel_cls]
                    if scan not in used_scans
                ]
                selected = np.random.choice(
                    available, self.k_shot, replace=False
                )
                used_scans.extend(selected)
                with open(reg_file, "a") as f:
                    # Write the novel class and the relative scan path.
                    for rel_scan in selected:
                        f.write(f"{novel_cls}\t{rel_scan}\n")
        comm.synchronize()

        # Load data list from the registration file and convert relative paths to absolute.
        self.data_list = []
        with open(reg_file, "r") as f:
            for line in f.read().splitlines():
                class_id_str, rel_scan = line.split("\t")
                abs_scan = os.path.join(self.data_root, rel_scan)
                self.data_list.append((class_id_str, abs_scan))

    def _create_class2scans(self, class2scans_file):
        """
        Build a mapping from class ID to scan folders.
        Filters out scans with too few points.
        The mapping is stored with paths relative to self.data_root.
        """
        min_points = 100  # Minimum number of points to consider
        class2scans = {cls: [] for cls in self.all_classes_orgid}
        for scan_folder in glob.glob(
            os.path.join(self.data_root, self.split, "*")
        ):
            scan_name = os.path.basename(scan_folder)
            labels_path = os.path.join(scan_folder, "segment20.npy")
            labels = np.load(labels_path)
            unique_classes = np.unique(labels)
            unique_classes = unique_classes[
                unique_classes != -1
            ]  # Exclude background
            print(
                f"{scan_name} | shape: {labels.shape} | classes: {list(unique_classes)}"
            )
            for cls in unique_classes:
                if cls not in self.all_classes_orgid:
                    continue
                point_count = np.count_nonzero(labels == cls)
                if point_count > min_points:
                    # Store the folder as a relative path.
                    rel_folder = os.path.relpath(scan_folder, self.data_root)
                    class2scans[cls].append(rel_folder)

        print("==== Class-to-scans mapping completed ====")
        for cls in self.all_classes_orgid:
            assert (
                len(class2scans[cls]) > 0
            ), f"Class {cls} ({self.orgid2name[cls]}) has no data."
            print(
                f"\t Class {cls} | min_points: {min_points} | Name: {self.orgid2name[cls]} | Scans: {len(class2scans[cls])}"
            )
        with open(class2scans_file, "wb") as f:
            pickle.dump(class2scans, f, pickle.HIGHEST_PROTOCOL)

    def get_data(self, idx):
        """
        Load and process a registration sample.
        Converts segmentation labels into a binary mask for the specified class.
        """
        class_id, data_path = self.data_list[idx % len(self.data_list)]
        class_id = int(class_id)
        sample_name = os.path.basename(data_path)
        if self.cache:
            cache_name = f"pointcept-{sample_name}"
            return shared_dict(cache_name)

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

        # Create binary mask for the current class.
        seg = sample.pop("segment20")
        seg = (seg == class_id).reshape(-1).astype(np.int32)
        seg[seg == 0] = self.bg_label
        seg[seg == 1] = self.lookup_array[class_id]
        sample["segment"] = seg
        return sample
