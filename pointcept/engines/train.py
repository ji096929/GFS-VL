"""
Trainer

Author: Zhaochong An (anzhaochong@outlook.com)
Please cite our work if the code is helpful to you.
"""

import gc
import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from functools import partial
import numpy as np
from collections import OrderedDict

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

from pcseg.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
)
from pathlib import Path
from pointcept.models.PLA.pcseg.models.text_networks import (
    load_text_embedding_from_encoder,
    load_text_embedding_from_path,
)
from pointcept.models.PLA.pcseg.models import (
    build_vision_network,
    build_text_network,
)
import torch.distributed as dist

TRAINERS = Registry("trainers")


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        st = getattr(self, "_filter_stats", None)
        if st is not None and self.cfg.get("use_margin_filter", False):
            n = st["ps_drop"] + st["margin_drop"] + st["keep"]
            if n - st.get("log_n", 0) >= 50 or n == 0:
                total = max(n, 1)
                self.logger.info(
                    f"[filter] n={n} ps_drop={st['ps_drop']} "
                    f"margin_drop={st['margin_drop']} keep={st['keep']} "
                    f"margin_rate={st['margin_drop']/total:.2%}"
                )
                st["log_n"] = n

        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


def _reserve_vram():
    """预占峰值显存（单方面防抢卡）。

    PyTorch 的缓存分配器在 del 之后不会把显存还给驱动，nvidia-smi 仍显示占用，
    但后续分配可以复用。开关：环境变量 GFS_RESERVE_GB（默认 0，关闭）。
    """
    gb = float(os.environ.get("GFS_RESERVE_GB", "0") or 0)
    if gb <= 0:
        return
    n = int(gb * (1024 ** 3) // 4)
    try:
        buf = torch.empty(n, dtype=torch.float32, device="cuda")
        del buf
        print("[reserve_vram] reserved %.1f GB into allocator cache" % gb, flush=True)
    except Exception as e:  # noqa: BLE001
        print("[reserve_vram] failed: %s" % e, flush=True)


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.epoch  # 原版误用 eval_epoch；历史配置两者相等，行为不变
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Reserve VRAM ...")
        _reserve_vram()
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(
                ">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>"
            )
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                # TODO: optimize to iteration based
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = (
            SummaryWriter(self.cfg.save_path)
            if comm.is_main_process()
            else None
        )
        self.logger.info(
            f"Tensorboard writer logging dir: {self.cfg.save_path}"
        )
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_data
            )
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=True,
            persistent_workers=(
                False if self.cfg.num_worker_per_gpu == 0 else True
            ),
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(
                    val_data
                )
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(
            self.cfg.optimizer, self.model, self.cfg.param_dicts
        )

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = (
            len(self.train_loader) * self.cfg.epoch  # 原版误用 eval_epoch：当 loop=epoch//eval_epoch 时总步数会被低估 loop 倍
        )
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler


@TRAINERS.register_module("GFS_VL_Trainer")
class GFS_VL_Trainer(Trainer):
    """
    Trainer for generalized few-shot 3D point cloud segmentation.
    """

    def __init__(self, cfg):
        """Initialize the trainer and build the 3D VLM."""
        super().__init__(cfg)
        if not os.environ.get("GFS_OFFLINE_PL", ""):
            self.build_zero_shot_model()
        else:
            # offline pseudo-labels: no VLM needed, saves ~3GB and 3min init
            self.VLM_3D = None
            self.VLM_3D_cfg = None

    def build_zero_shot_model(self):
        """
        Build the 3D VLM model used for pseudo-label generation.
        """
        cfg_file = "./pointcept/models/PLA/tools/cfgs/scannet200_models/zs/spconv_clip_caption_openscene.yaml"
        self.VLM_3D_cfg = cfg_from_yaml_file(cfg_file, cfg)

        # Combine base and novel class names for the 3D VLM model
        cfg.CLASS_NAMES = (
            self.train_loader.dataset.base_class_names
            + self.train_loader.dataset.novel_class_names
        )
        self.num_class = self.cfg.data.num_base_novels

        # Configure experiment tag and group path
        cfg.TAG = Path(cfg_file).stem
        cfg.EXP_GROUP_PATH = "/".join(cfg_file.split("/")[1:-1])

        # Update configuration for text embedding extraction
        cfg_from_list(["TEXT_ENCODER.EXTRACT_EMBED", True], cfg)

        logger = self.logger
        # Build text embedding if specified
        if cfg.get("TEXT_ENCODER", None) or cfg.MODEL.TASK_HEAD.get(
            "TEXT_EMBED", None
        ):
            text_encoder = build_text_network(cfg.TEXT_ENCODER).cuda()
            if (
                cfg.get("TEXT_ENCODER", None)
                and cfg.TEXT_ENCODER.EXTRACT_EMBED
            ):
                text_embed = load_text_embedding_from_encoder(
                    cfg.TEXT_ENCODER, text_encoder, logger
                )
            else:
                text_embed = load_text_embedding_from_path(
                    cfg.MODEL.TASK_HEAD.TEXT_EMBED, logger
                )
            cfg.MODEL.TASK_HEAD.TEXT_EMBED.CHANNEL = text_embed.shape[1]
            cfg.MODEL.TASK_HEAD.TEXT_EMBED.NUM_CLASS = text_embed.shape[0]
            if cfg.MODEL.get("ADAPTER", False):
                cfg.MODEL.ADAPTER.TEXT_DIM = text_embed.shape[1]
        else:
            text_embed = None

        # Build the 3D VLM model
        self.VLM_3D = build_vision_network(
            model_cfg=cfg.MODEL,
            num_class=cfg.CLASS_NAMES,
            dataset=cfg.DATA_CONFIG,
        )
        self.VLM_3D.load_params_from_file(
            filename=self.cfg.vlm_3d_weight,
            logger=logger,
            epoch_id=None,
            to_cpu=False,
        )

        # Set the classification head with text embeddings if available
        if text_embed is not None:
            self.VLM_3D.task_head.set_cls_head_with_text_embed(text_embed)

        self.VLM_3D.cuda()
        self.VLM_3D.eval()
        # Freeze parameters of the 3D VLM model
        for param in self.VLM_3D.parameters():
            param.requires_grad = False

    def extract_3d_vlm_proto(self, regis_val_loader):
        """
        Compute novel classes prototypes using the 3D VLM.
        """
        num_novel = self.cfg.data.num_base_novels - self.cfg.data.num_bases
        average_features = torch.zeros(num_novel, 512, device="cuda")
        class_counts = torch.zeros(num_novel, device="cuda")
        # 多原型：每个 shot 一个独立原型（修 5-shot 单均值坍塌）
        _K = max(int(getattr(self.cfg, "k_shot", 1) or 1), 1)
        multi_protos = torch.zeros(num_novel, _K, 512, device="cuda")
        multi_valid = torch.zeros(num_novel, _K, dtype=torch.bool, device="cuda")
        _shot_cnt = torch.zeros(num_novel, dtype=torch.long, device="cuda")

        if comm.is_main_process():
            for input_dict in regis_val_loader:
                # Transfer all tensor inputs to CUDA
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.cuda(non_blocking=True)

                with torch.no_grad():
                    # Construct input for 3D VLM model and obtain features
                    data_dict = self.construct_3D_vlm_input(input_dict)
                    output_dict = self.VLM_3D(data_dict)
                    vlm_feats = output_dict["all_feats"]

                    # Process each unique class (skip background: -1)
                    for class_id in input_dict["segment"].unique():
                        class_id = class_id.item()
                        if class_id == -1:
                            continue
                        assert class_id >= self.cfg.data.num_bases
                        class_mask = input_dict["segment"] == class_id
                        class_features = vlm_feats[class_mask]
                        idx = class_id - self.cfg.data.num_bases
                        average_features[idx] += class_features.sum(dim=0)
                        class_counts[idx] += class_features.size(0)
                        _k = int(_shot_cnt[idx])
                        if _k < _K:
                            multi_protos[idx, _k] = class_features.mean(dim=0)
                            multi_valid[idx, _k] = True
                            _shot_cnt[idx] += 1

            # Verify each novel class has at least one sample
            assert torch.all(class_counts > 0)
            # Normalize to get the average feature per class
            average_features = average_features / class_counts.unsqueeze(1)

        # Broadcast the computed prototypes to all processes in distributed training
        if comm.get_world_size() > 1:
            dist.broadcast(average_features, src=0)
        self.vlm_novel_protos = (multi_protos, multi_valid)
        return average_features

    def build_model(self):
        """
        Construct the model, convert it for DDP if needed,
        and load pre-trained weights.
        """
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        n_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        self.logger.info(f"Num params: {n_parameters}")

        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )

        # Load pre-trained weights if available
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # Remove 'module.' for single GPU
                else:
                    if comm.get_world_size() > 1:
                        key = (
                            "module." + key
                        )  # Prepend 'module.' for multi-GPU
                weight[key] = value
            strict = (
                self.cfg.load_strict if "load_strict" in self.cfg else True
            )
            missing_keys, unexpected_keys = model.load_state_dict(
                weight, strict=strict
            )
            self.logger.info(
                f"=> Loaded weight '{self.cfg.weight}' (epoch {checkpoint['epoch']})"
            )
            self.logger.info(
                f"=> missing_keys {missing_keys}, unexpected_keys {unexpected_keys}"
            )

        return model

    def train(self):
        """
        Main training loop for generalized few-shot learning. Iterates over multiple
        registration datasets and trains the segmentation model accordingly.
        """
        for self.regis_str in self.cfg.regis_train_list:
            # Retrieve registration dataset configuration
            if hasattr(self.cfg.data, self.regis_str):
                val_data_cfg = getattr(self.cfg.data, self.regis_str)
            else:
                raise AttributeError(
                    f"{self.regis_str} not found in self.cfg.data"
                )
            val_data = build_dataset(val_data_cfg)
            regis_val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=None,
                collate_fn=collate_fn,
            )

            # 多 split 训练时按 seed 切换离线伪标签目录
            _tmpl = os.environ.get("GFS_OFFLINE_PL_TMPL", "")
            if _tmpl:
                os.environ["GFS_OFFLINE_PL"] = _tmpl.replace(
                    "{seed}", str(val_data_cfg["seed"]))
                self.logger.info(
                    "=> Offline PL dir: " + os.environ["GFS_OFFLINE_PL"])
            # Extract 3D VLM feature prototypes on novel classes from current registration set
            if os.environ.get("GFS_OFFLINE_PL", ""):
                self.vlm_novel_proto = None
            else:
                self.vlm_novel_proto = self.extract_3d_vlm_proto(regis_val_loader)
            if getattr(self.cfg, "pl_mode", None) in ("dual_eye_and", "dual_eye_h1", "dual_eye_m1_vote", "psai_dec2_rank"):
                self._init_dec2_eye(regis_val_loader)

            # Update the training dataset configuration
            self.cfg.data.train.k_shot = self.cfg.k_shot
            self.cfg.data.train.seed = val_data_cfg["seed"]
            self.cfg.data.train.nb_mix_blks = self.cfg.nb_mix_blks
            self.train_loader = self.build_train_loader()
            if getattr(self.cfg, "pl_mode", None) == "dual_eye_h1":
                self._calibrate_h1_vlm_thresholds()

            # Resume: per-regis-split checkpoint restore (model/opt/sched/scaler)
            self.start_epoch = 0
            _resume_path = os.path.join(
                self.cfg.save_path, "model", f"{self.regis_str}_model_last.pth"
            )
            if os.path.exists(_resume_path) and not os.environ.get("GFS_NO_RESUME"):
                try:
                    _ck = torch.load(
                        _resume_path, map_location="cpu", weights_only=False
                    )
                    _ep = int(_ck.get("epoch", 0))
                    if 0 < _ep < self.max_epoch:
                        _miss, _unexp = self.model.load_state_dict(
                            _ck["state_dict"], strict=False)
                        self.logger.info(
                            "=> Resume: missing=%d unexpected=%d"
                            % (len(_miss), len(_unexp)))
                        if _ck.get("optimizer") is not None:
                            self.optimizer.load_state_dict(_ck["optimizer"])
                        if _ck.get("scheduler") is not None:
                            self.scheduler.load_state_dict(_ck["scheduler"])
                        if _ck.get("scaler") is not None and self.scaler is not None:
                            self.scaler.load_state_dict(_ck["scaler"])
                        self.start_epoch = _ep
                        self.logger.info(
                            "=> Resumed %s at epoch %d/%d"
                            % (_resume_path, _ep, self.max_epoch)
                        )
                except Exception as _e:  # noqa: BLE001
                    self.logger.info("=> Resume skipped: %s" % _e)
            elif len(self.cfg.regis_train_list) > 1:
                # 多 split 各自独立评测。若沿用上一个 split 训练完的权重，
                # split2..N 就是在已经见过 novel 类的模型上继续训 20 epoch，
                # 会把 5-split 均值系统性抬高，结果不能对标官方。
                self.logger.info(
                    "=> 多 split：重置模型/优化器/调度器，并重新注入 Stage-1 骨干"
                )
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
                self.model = self.build_model()
                for _h in self.hooks:
                    if _h.__class__.__name__ == "BackboneLoader":
                        _h.before_train()
                self.optimizer = self.build_optimizer()
                self.scheduler = self.build_scheduler()
                self.scaler = self.build_scaler()
                self.start_epoch = 0

            # Execute the training loop for the current registration dataset
            with EventStorage() as self.storage, ExceptionWriter():
                self.before_train()
                self.logger.info(
                    ">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>"
                )
                for self.epoch in range(self.start_epoch, self.max_epoch):
                    if comm.get_world_size() > 1:
                        self.train_loader.sampler.set_epoch(self.epoch)
                    self.model.train()
                    self.data_iterator = enumerate(self.train_loader)
                    self.before_epoch()
                    for (
                        self.comm_info["iter"],
                        self.comm_info["input_dict"],
                    ) in self.data_iterator:
                        self.before_step()
                        self.run_step()
                        self.after_step()
                    self.after_epoch()
                self.after_train()

        # If using multiple registration datasets, log the final evaluation metrics
        if comm.is_main_process() and len(self.cfg.regis_train_list) > 1:
            class_names = (
                self.train_loader.dataset.base_class_names
                + self.train_loader.dataset.novel_class_names
            )
            mean_mIoUs = self.hooks[-1].mean_mIoUs / len(
                self.hooks[-1].mean_iou_list
            )
            mIoU_val, base_mIoU, novel_mIoU, hm_mIoU = mean_mIoUs
            self.logger.info(
                f"Final Eval result from {self.cfg.regis_train_list}: mIoU: {mIoU_val:.4f}, "
                f"BASE: {base_mIoU:.4f}, NOVEL: {novel_mIoU:.4f}, hm_mIoU: {hm_mIoU:.4f}"
            )
            # Log class-wise IoU results
            stack_iou = np.mean(
                np.stack(self.hooks[-1].mean_iou_list, axis=0), axis=0
            )
            for i, iou in enumerate(stack_iou):
                self.logger.info(
                    f"Class_{i} - {class_names[i]} Final Result: iou {iou:.4f}"
                )

    def _update_elr_ema(self):
        ema_m = getattr(self, "_elr_ema_model", None)
        if ema_m is None:
            return
        decay = float(getattr(self.cfg, "elr_ema_decay", 0.995))
        raw_m = self.model.module if comm.get_world_size() > 1 else self.model
        with torch.no_grad():
            for ep, sp in zip(ema_m.parameters(), raw_m.parameters()):
                ep.mul_(decay).add_(sp.data, alpha=1.0 - decay)
            for eb, sb in zip(ema_m.buffers(), raw_m.buffers()):
                if eb.is_floating_point():
                    eb.mul_(decay).add_(sb.data, alpha=1.0 - decay)
                else:
                    eb.copy_(sb.data)

    def run_step(self):
        """
        Execute a single training step:
          - Calibrate target labels using the 3D VLM.
          - Forward pass through the segmentation model.
          - Compute loss and update model parameters.
        """
        input_dict = self.comm_info["input_dict"]
        # Move tensor inputs to GPU
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor):
                input_dict[key] = value.cuda(non_blocking=True)

        # Level 1: student_consensus 需要先 forward，再复用 detach 的 logits
        # 做仲裁。否则会多跑一次前向，在 train() 模式下双倍更新 SyncBN。
        orig_gt_seg = input_dict["segment"]
        _mode = getattr(self.cfg, "pl_mode", "ps_ai")
        _warm = int(getattr(self.cfg, "warmup_epochs", 0))
        _reuse = ((_mode == "student_consensus" and self.epoch >= _warm)
                  or (_mode == "entropy_q"
                      and bool(getattr(self.cfg, "use_3d_veto", False))
                      and self.epoch >= int(
                          getattr(self.cfg, "veto_warmup_epochs", 0))))

        if _reuse:
            with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
                _pre_out = self.model(input_dict)
                _pre_logits = _pre_out.pop("seg_logits")
                _pre_decorr = _pre_out.pop("decorr_loss", None)
                _pre_cross = _pre_out.pop("cross_logits", None)
            new_target = self.target_calibrate(
                input_dict, seg_logits=_pre_logits.detach())
        else:
            # Calibrate target labels based on 3D VLM predictions
            new_target = self.target_calibrate(input_dict)

        _elr_lam = float(getattr(self.cfg, "elr_lam", 0.0) or 0.0)
        _elr_warm = int(getattr(self.cfg, "elr_warmup_epochs", 1))
        _reg_tag = getattr(self, "regis_str", None)
        if getattr(self, "_elr_regis_tag", None) != _reg_tag:
            self._elr_regis_tag = _reg_tag
            self._elr_ema_model = None
        ema_logits = None
        if _elr_lam > 0.0 and self.epoch >= _elr_warm:
            if getattr(self, "_elr_ema_model", None) is None:
                raw_m = self.model.module if comm.get_world_size() > 1 else self.model
                self._elr_ema_model = build_model(self.cfg.model).cuda()
                self._elr_ema_model.load_state_dict(raw_m.state_dict(), strict=True)
                self._elr_ema_model.eval()
                for _p in self._elr_ema_model.parameters():
                    _p.requires_grad_(False)
                if comm.is_main_process():
                    self.logger.info(
                        "[ELR-EMA] initialized EMA teacher at ep=%d (decay=%.4f, lam=%.2f)"
                        % (self.epoch + 1, float(getattr(self.cfg, "elr_ema_decay", 0.995)), _elr_lam)
                    )
            _ema_inp = {k: v for k, v in input_dict.items() if k != "segment"}
            with torch.no_grad():
                ema_logits = self._elr_ema_model(_ema_inp)["seg_logits"].detach().float()

        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            if _reuse:
                output_dict = _pre_out
                seg_logits = _pre_logits
                decorr_loss = _pre_decorr
                cross_logits = _pre_cross
            else:
                output_dict = self.model(input_dict)
                seg_logits = output_dict.pop("seg_logits")
                decorr_loss = output_dict.pop("decorr_loss", None)
                cross_logits = output_dict.pop("cross_logits", None)
            criteria = (
                self.model.module.criteria
                if comm.get_world_size() > 1
                else self.model.criteria
            )
            _pw = (
                input_dict["pl_weight"].float()
                if "pl_weight" in input_dict else None
            )

            # ---- 训练期 Logit Adjustment (Menon et al., ICLR 2021) ----
            _use_la = bool(getattr(self.cfg, "use_la", False))
            if _use_la and not hasattr(self, "_la_bias"):
                _pi_path = getattr(self.cfg, "la_prior_path",
                                   "/data01/jh/data/pl_class_prior_full.npy")
                _pi = torch.tensor(np.load(_pi_path), dtype=torch.float32,
                                   device=seg_logits.device)
                _pi = _pi / _pi.sum()
                _tau = float(getattr(self.cfg, "la_tau", 0.5))
                self._la_bias = _tau * torch.log(_pi).unsqueeze(0)
                self.logger.info("[LA] tau=%.2f pi range %.6f~%.6f"
                                 % (_tau, float(_pi.min()), float(_pi.max())))
            if _use_la:
                seg_logits = seg_logits + self._la_bias.to(seg_logits.dtype)

            _gce_q = float(getattr(self.cfg, "pl_gce_q", 0.0) or 0.0)

            def _weighted_ce(logits):
                # 主 loss 与辅助 loss 共用同一口径，避免软权重模式下静默不一致
                if _pw is None:
                    if _gce_q > 0.0:
                        with torch.cuda.amp.autocast(enabled=False):
                            logits_f32 = logits.float()
                            valid = new_target != -1
                            if not valid.any():
                                return logits_f32.sum() * 0.0
                            log_p = torch.nn.functional.log_softmax(logits_f32[valid], dim=-1)
                            tgt_v = new_target[valid].long()
                            log_py = log_p.gather(1, tgt_v.unsqueeze(1)).squeeze(1)
                            is_pl_v = (orig_gt_seg[valid] == -1)
                            ce_v = -log_py
                            gce_v = (1.0 - torch.exp(_gce_q * log_py.clamp(min=-30.0))) / _gce_q
                            if not getattr(self, "_gce_logged", False) and comm.is_main_process():
                                self._gce_logged = True
                                n_gt = int((~is_pl_v).sum().item())
                                n_pl = int(is_pl_v.sum().item())
                                ce_gt = float(ce_v[~is_pl_v].mean().item()) if n_gt > 0 else 0.0
                                gce_pl = float(gce_v[is_pl_v].mean().item()) if n_pl > 0 else 0.0
                                ce_pl_ref = float(ce_v[is_pl_v].mean().item()) if n_pl > 0 else 0.0
                                self.logger.info(
                                    "[PL-GCE] q=%.2f valid=%d gt=%d pl=%d ce_gt=%.4f gce_pl=%.4f ce_pl_ref=%.4f"
                                    % (_gce_q, int(valid.sum().item()), n_gt, n_pl, ce_gt, gce_pl, ce_pl_ref)
                                )
                            return torch.where(is_pl_v, gce_v, ce_v).mean()
                    return criteria(logits, new_target)
                ce = torch.nn.functional.cross_entropy(
                    logits, new_target, reduction="none", ignore_index=-1)
                valid = new_target != -1
                return (ce * _pw)[valid].sum() / (_pw[valid].sum() + 1e-6)

            loss = _weighted_ce(seg_logits)
            if ema_logits is not None:
                with torch.cuda.amp.autocast(enabled=False):
                    _valid = new_target != -1
                    _is_pl = (orig_gt_seg == -1) & _valid
                    if _is_pl.any():
                        _p_prob = torch.nn.functional.softmax(seg_logits[_is_pl].float(), dim=-1)
                        _t_prob = torch.nn.functional.softmax(ema_logits[_is_pl], dim=-1)
                        _dot = (_p_prob * _t_prob).sum(dim=-1).clamp(min=1e-4, max=1.0 - 1e-4)
                        _elr_loss = torch.log(1.0 - _dot).sum() / (_valid.sum().float() + 1e-6)
                        if (self.comm_info.get("iter", 1) == 0 and (self.epoch + 1 in (2, 5, 10, 15, 20))
                                and comm.is_main_process()):
                            self.logger.info(
                                "[ELR-EMA ep=%d] ce=%.4f elr=%.4f (lam=%.1f) dot_pl=%.4f max_t_pl=%.4f n_pl=%d/%d"
                                % (self.epoch + 1, float(loss.detach()), float(_elr_loss.detach()), _elr_lam,
                                   float(_dot.mean().detach()), float(_t_prob.max(dim=-1).values.mean().detach()),
                                   int(_is_pl.sum().item()), int(_valid.sum().item()))
                            )
                        loss = loss + _elr_lam * _elr_loss
            if _pw is not None and os.environ.get("GFS_DEBUG_W") and not getattr(self, "_w_dbg", False):
                self._w_dbg = True
                valid = new_target != -1
                pw = _pw[valid]
                base = (new_target >= 0) & (new_target < 12)
                self.logger.info(
                    "[pl_weight] n=%d mean=%.3f min=%.3f max=%.3f | "
                    "base-GT w min=%.3f max=%.3f"
                    % (int(valid.sum()), float(pw.mean()), float(pw.min()),
                       float(pw.max()),
                       float(_pw[base].min()) if base.any() else -1.0,
                       float(_pw[base].max()) if base.any() else -1.0))
            if getattr(self.cfg, "use_lovasz", False):
                from pointcept.models.losses import LovaszLoss
                _lv = LovaszLoss(mode="multiclass", ignore_index=-1,
                                 loss_weight=float(getattr(self.cfg, "lovasz_w", 1.0)))
                _n = seg_logits.shape[0]
                _lo = _lv(seg_logits.view(_n, -1, 1, 1),
                          new_target.view(_n, 1, 1))
                if not getattr(self, "_lv_logged", False):
                    self._lv_logged = True
                    self.logger.info("[Lovasz] ce=%.4f lovasz=%.4f" % (float(loss), float(_lo)))
                loss = loss + _lo
            if decorr_loss is not None:
                loss = loss + decorr_loss
            if cross_logits is not None:
                _lam = float(getattr(self.cfg, "lam_cross", 0.3))
                loss = loss + _lam * _weighted_ce(cross_logits)
            output_dict["loss"] = loss

        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            _gn_probe(self)
            self.scaler.step(self.optimizer)
            # Adjust the scheduler if loss scaling factor remains unchanged
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
            self._update_elr_ema()
        else:
            loss.backward()
            _gn_probe(self)
            self.optimizer.step()
            self.scheduler.step()
            self._update_elr_ema()

        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def _compute_text_margin(self, proto, class_id):
        """margin = cos(proto, text_self) - max_{b in base} cos(proto, text_b).

        text_* = VLM classification head weights (already in memory).
        Low margin = mean feature sits close to a base class ("confidently wrong").
        """
        text = self.VLM_3D.task_head.cls_head.weight
        proto_n = proto / proto.norm().clamp(min=1e-8)
        cos_self = torch.dot(proto_n, text[class_id])
        cos_base = (text[: self.cfg.data.num_bases] @ proto_n).max()
        return float(cos_self - cos_base)

    def pseudo_label_selection(
        self, all_pred_cloud, all_feat_cloud, offset, ps_thresh=0.6
    ):
        """
        Filter raw 3D VLM predictions based on prototype similarity.

        Args:
            all_pred_cloud (Tensor): Raw 3D VLM predictions for all points.
            all_feat_cloud (Tensor): Feature vectors for all points.
            offset (list): Batch offsets indicating point indices.
            ps_thresh (float): Cosine similarity threshold for filtering.

        Returns:
            Tensor: Filtered 3D VLM predictions.
        """
        if not hasattr(self, "_filter_stats"):
            self._filter_stats = {"ps_drop": 0, "margin_drop": 0, "keep": 0, "log_n": 0}
        filtered_preds = []
        # Process each batch using provided offsets
        for start_idx, end_idx in zip([0] + offset[:-1], offset):
            class_prototypes = {}
            pred_cloud = all_pred_cloud[start_idx:end_idx].clone()
            feat_cloud = all_feat_cloud[start_idx:end_idx].clone()

            # Calculate mean feature for each predicted class in the batch
            for class_id in pred_cloud.unique():
                class_mask = pred_cloud == class_id
                class_prototypes[class_id.item()] = feat_cloud[
                    class_mask
                ].mean(dim=0)

            # Filter predictions based on similarity with 3D VLM prototypes
            for class_id, proto in class_prototypes.items():
                if class_id < self.cfg.data.num_bases:
                    # Ignore base classes
                    pred_cloud[pred_cloud == class_id] = -1
                else:
                    _ni = class_id - self.cfg.data.num_bases
                    similarity = None
                    if hasattr(self, "vlm_novel_protos"):
                        _protos, _valid = self.vlm_novel_protos
                        _p = _protos[_ni][_valid[_ni]]
                        if _p.numel() > 0:
                            # 取与该类 K 个 shot 原型中的最大相似度
                            similarity = torch.cosine_similarity(
                                proto.unsqueeze(0), _p, dim=1).max()
                    if similarity is None:
                        similarity = torch.cosine_similarity(
                            proto, self.vlm_novel_proto[_ni], dim=0
                        )
                    if similarity < ps_thresh:
                        self._filter_stats["ps_drop"] += 1
                        pred_cloud[pred_cloud == class_id] = -1
                    elif self.cfg.get("use_margin_filter", False):
                        # Margin filter: drop if mean feature sits too close to
                        # a base class text embedding (confidently-wrong signal).
                        margin = self._compute_text_margin(proto, class_id)
                        margin_thresh = self.cfg.get("margin_thresh", 0.023)
                        if margin < margin_thresh:
                            self._filter_stats["margin_drop"] += 1
                            pred_cloud[pred_cloud == class_id] = -1
                        else:
                            self._filter_stats["keep"] += 1
                    else:
                        self._filter_stats["keep"] += 1

            filtered_preds.append(pred_cloud)

        return torch.cat(filtered_preds, dim=0)

    @torch.no_grad()
    def target_calibrate(self, input_dict, seg_logits=None):
        """
        Calibrate the segmentation target labels using 3D VLM and few-shot data.

        seg_logits: 当前训练前向的 logits（已 detach）。student_consensus 模式下
        复用它做学生仲裁，避免再跑一次前向污染 SyncBN 统计。
        """
        if os.environ.get("GFS_OFFLINE_PL", ""):
            # offline pseudo-labels already merged into segment by the dataset
            return input_dict["segment"].clone()

        # ---- Oracle 上界实验（GFS_ORACLE_MODE）----
        _oracle = os.environ.get("GFS_ORACLE_MODE", "")
        if _oracle == "fullsup":
            # 全监督：直接用完整 GT（需 data.train.full_gt=True）
            return input_dict["segment"].clone()

        # L1-Pure 热身期：连冻结 VLM 都不跑，直接返回干净 GT
        _m0 = getattr(self.cfg, "pl_mode", "ps_ai")
        if (_m0 == "student_consensus"
                and self.epoch < int(getattr(self.cfg, "warmup_epochs", 4))
                and getattr(self.cfg, "warmup_mode", "none") == "none"):
            return input_dict["segment"].clone()

        data_dict = self.construct_3D_vlm_input(input_dict)
        ret_dict = self.VLM_3D(data_dict)

        label = input_dict["segment"].clone()
        bg_label_mask = label == -1  # Default background label

        # Oracle 完美过滤：需要 ret_dict 已生成
        if _oracle == "filter":
            return self._target_oracle_filter(ret_dict, input_dict)

        _mode = getattr(self.cfg, "pl_mode", "ps_ai")

        # ---- Level 1: 学生 × VLM Top-K 双向共识 ----
        if _mode == "student_consensus":
            _warm = int(getattr(self.cfg, "warmup_epochs", 4))
            if self.epoch < _warm:
                # 热身期：L1-Pure 完全不生成伪标签；L1-Hybrid 用 1C 的熵过滤
                if getattr(self.cfg, "warmup_mode", "none") == "entropy_q":
                    return self._target_entropy_quantile(
                        ret_dict, input_dict, label, bg_label_mask)
                return label
            return self._student_consensus_pl(input_dict, ret_dict, seg_logits)

        # ---- M1: 30 类全空间 argmax（不屏蔽基类）----
        # 实测：屏蔽 12 个基类会让未标注区的天花板/白墙失去吸收器，
        # smoke detector 在 712 chunk 上产生 982 万假阳性（GT 的 4400 倍）。
        # 改回 30 类竞争后同一统计只剩 15 点。
        if _mode in ("m1_argmax", "m1_cos"):
            return self._target_m1_argmax(
                ret_dict, input_dict, use_cos=(_mode == "m1_cos"))

        # ---- 1C: UGA UB-Filter 等价实现（pl_mode="entropy_q"）----
        # 跳过 PS/AI，直接对 VLM 原生 seg_scores 做 class-wise 熵分位过滤。
        if _mode == "entropy_q":
            return self._target_entropy_quantile(
                ret_dict, input_dict, label, bg_label_mask, seg_logits)
        if _mode == "global_entropy_q":
            return self._target_global_entropy_quantile(
                ret_dict, input_dict, label, bg_label_mask)
        if _mode == "dual_eye_and":
            return self._target_dual_eye_and(
                ret_dict, input_dict, label, bg_label_mask)
        if _mode == "dual_eye_h1":
            return self._target_dual_eye_h1(
                ret_dict, input_dict, label, bg_label_mask)
        if _mode == "dual_eye_m1_vote":
            return self._target_dual_eye_m1_vote(
                ret_dict, input_dict, label, bg_label_mask)

        prediction = ret_dict["seg_preds"].clone()
        # Filter predictions using prototype similarity
        prediction = self.pseudo_label_selection(
            prediction,
            ret_dict["all_feats"],
            input_dict["offset"].tolist(),
            ps_thresh=self.cfg.ps_thresh,
        )

        # Adaptive Infilling to further refine the pseudo-labels
        offset = input_dict["offset"].tolist()
        new_predictions = []
        # Process each batch of points
        for start_idx, end_idx in zip([0] + offset[:-1], offset):
            current_prototypes = {}
            pred_cloud = prediction[start_idx:end_idx].clone()
            feat_cloud = ret_dict["all_feats"][start_idx:end_idx].clone()
            cur_label = label[start_idx:end_idx].clone()
            cur_bg_mask = bg_label_mask[start_idx:end_idx]
            bg_pred_cloud = pred_cloud[cur_bg_mask]
            bg_feat_cloud = feat_cloud[cur_bg_mask]

            if "mask" in input_dict:
                noda_mask = input_dict["mask"][start_idx:end_idx].clone()

            # Compute prototypes for each novel class present in the background region
            for class_id in bg_pred_cloud.unique():
                if class_id < 0:
                    continue
                assert class_id >= self.cfg.data.num_bases
                class_features = bg_feat_cloud[bg_pred_cloud == class_id]
                current_prototypes[class_id.item()] = class_features.mean(
                    dim=0
                )

            updated_prototypes = []
            updated_class_ids = []
            # Prepare updated prototypes for all novel classes
            for class_id in range(
                self.cfg.data.num_bases, self.cfg.data.num_base_novels
            ):
                if class_id in current_prototypes:
                    updated_prototypes.append(current_prototypes[class_id])
                else:
                    updated_prototypes.append(
                        self.vlm_novel_proto[
                            class_id - self.cfg.data.num_bases
                        ].clone()
                    )
                updated_class_ids.append(class_id)
            updated_prototypes = torch.stack(updated_prototypes)

            # Identify points with unassigned labels (-1) for refinement
            if "mask" in input_dict:
                mask_neg_one = (
                    (pred_cloud == -1) & (cur_label == -1) & (noda_mask == 1)
                )
            else:
                mask_neg_one = (pred_cloud == -1) & (cur_label == -1)
            feat_neg_one = feat_cloud[mask_neg_one]

            # Compute cosine similarity between these features and the updated prototypes
            # 分块计算：大场景（原始点数）整场景都是 -1，M 极大，
            # (M,45,512) 广播会撑爆显存。分块不改变结果。
            _chunk = 16384
            _ms, _mc = [], []
            for _s in range(0, feat_neg_one.shape[0], _chunk):
                _f = feat_neg_one[_s:_s + _chunk]
                _sim = torch.cosine_similarity(
                    _f[:, None, :],
                    updated_prototypes[None, :, :],
                    dim=-1,
                )
                _a, _b = _sim.max(dim=1)
                _ms.append(_a)
                _mc.append(_b)
            if _ms:
                max_similarity = torch.cat(_ms)
                max_class_idx = torch.cat(_mc)
            else:
                max_similarity = torch.empty(0, device=feat_neg_one.device)
                max_class_idx = torch.empty(
                    0, dtype=torch.long, device=feat_neg_one.device)
            refined_labels = pred_cloud.clone()
            refined_labels[mask_neg_one] = torch.where(
                max_similarity >= self.cfg.ai_thresh,
                torch.tensor(updated_class_ids, device=pred_cloud.device)[
                    max_class_idx
                ],
                -1,  # Retain -1 if below threshold
            )
            new_predictions.append(refined_labels)

        # Merge predictions from all batches and update background regions in the original label
        prediction = torch.cat(new_predictions, dim=0)

        # ---- 1C-c: 在 PS+AI 之上再叠按类熵分位过滤（UGA 的配方）----
        if getattr(self.cfg, "pl_mode", "ps_ai") == "ps_ai_then_entropy":
            prediction = self._entropy_prune(
                prediction, ret_dict, input_dict, bg_label_mask, offset)
        if getattr(self.cfg, "pl_mode", "ps_ai") == "psai_dec2_rank":
            prediction = self._dec2_rank_prune(prediction, input_dict, bg_label_mask)

        label_one_hot = label.clone()
        label_one_hot[bg_label_mask] = prediction[bg_label_mask]

        return label_one_hot


    @torch.no_grad()
    @torch.no_grad()
    def _init_dec2_eye(self, regis_val_loader):
        """Initialize frozen Stage-1 PTv3 dec2 reference eye & extract K-shot novel prototypes."""
        from pointcept.models.utils.structure import Point
        import torch.nn.functional as F

        nb = int(self.cfg.data.num_bases)
        nn_cls = int(self.cfg.data.num_base_novels) - nb

        if not hasattr(self, "_dec2_bb"):
            mcfg = dict(self.cfg.model.backbone)
            mcfg["enable_flash"] = False
            bb = build_model(mcfg).cuda().eval()
            for p in bb.parameters():
                p.requires_grad_(False)
            ck = torch.load(self.cfg.backbone_weight, map_location="cpu", weights_only=False)
            sd = ck["state_dict"] if "state_dict" in ck else ck
            bb_sd = {}
            for k, v in sd.items():
                k2 = k.replace("module.", "", 1)
                if k2.startswith("backbone."):
                    bb_sd[k2[len("backbone."):]] = v
            miss, unexp = bb.load_state_dict(bb_sd, strict=True)
            assert len(miss) == 0 and len(unexp) == 0, f"dec2_eye load mismatch: miss={miss} unexp={unexp}"

            cache = {}
            n_dec = len(bb.dec._modules)
            for s in range(n_dec):
                bb.dec._modules[f"dec{s}"]._modules["up"].register_forward_pre_hook(
                    lambda m, inp, s=s: cache.__setitem__(f"inv{s}", inp[0]["pooling_inverse"])
                )
            bb.dec._modules["dec2"].register_forward_hook(
                lambda m, inp, out: cache.__setitem__("f2", out.feat)
            )
            self._dec2_bb = bb
            self._dec2_cache = cache
            self._dec2_Point = Point
            self.logger.info(
                f"=> [DualEyeAND] Loaded frozen Stage-1 dec2 eye from {self.cfg.backbone_weight} (strict=True)"
            )

        # Infer dec2 channel dim from backbone config (dec_channels[2] = 128 for PTv3)
        c_dec2 = int(self.cfg.model.backbone["dec_channels"][2])
        proto_sum = torch.zeros(nn_cls, c_dec2, dtype=torch.float32, device="cuda")
        counts = torch.zeros(nn_cls, dtype=torch.long, device="cuda")

        if comm.is_main_process():
            for batch in regis_val_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda(non_blocking=True)
                f2 = self._extract_dec2(batch)
                seg = batch["segment"]
                for cid in seg.unique():
                    cid_i = int(cid.item())
                    if cid_i < nb:
                        continue
                    idx = cid_i - nb
                    mask = seg == cid_i
                    proto_sum[idx] += f2[mask].sum(dim=0)
                    counts[idx] += int(mask.sum().item())
            assert torch.all(counts > 0), f"dec2_eye empty novel classes: {counts.tolist()}"
            protos = F.normalize(proto_sum, dim=-1)
        else:
            protos = torch.zeros(nn_cls, c_dec2, dtype=torch.float32, device="cuda")

        if comm.get_world_size() > 1 and dist.is_initialized():
            dist.broadcast(protos, src=0)
            dist.broadcast(counts, src=0)

        self._dec2_novel_proto = protos.contiguous()
        # Reset EMA thresholds and retention stats per registration split
        for attr in ("_dual_thr1", "_dual_thr2", "_dual_cls_keep", "_dual_cls_cand",
                     "_dual_bg_tot", "_dual_gt_tot", "_dual_step"):
            if hasattr(self, attr):
                delattr(self, attr)
        if comm.is_main_process():
            self.logger.info(
                f"=> [DualEyeAND] Extracted {nn_cls} novel dec2 prototypes (dim={c_dec2}, "
                f"min_pts={int(counts.min().item())}, max_pts={int(counts.max().item())})"
            )

    @torch.no_grad()
    def _extract_dec2(self, input_dict):
        """Extract L2-normalized FP32 dec2 features without mutating input_dict."""
        import torch.nn.functional as F

        pt_in = {}
        for k in ("coord", "feat", "grid_coord", "offset", "batch"):
            if k in input_dict and isinstance(input_dict[k], torch.Tensor):
                pt_in[k] = input_dict[k].detach()
        if "offset" not in pt_in and "batch" not in pt_in:
            pt_in["offset"] = torch.tensor([pt_in["coord"].shape[0]], dtype=torch.long, device=pt_in["coord"].device)
        if "grid_coord" not in pt_in:
            pt_in["grid_size"] = input_dict.get("grid_size", 0.02)

        self._dec2_cache.clear()
        with torch.cuda.amp.autocast(enabled=False):
            _ = self._dec2_bb(self._dec2_Point(pt_in))
            f = self._dec2_cache["f2"].float()
            f = f[self._dec2_cache["inv1"]]
            f = f[self._dec2_cache["inv0"]]
            f = F.normalize(f, dim=-1)
        self._dec2_cache.clear()
        return f

    @torch.no_grad()
    def _target_dual_eye_and(self, ret_dict, input_dict, label, bg_label_mask):
        """Dual-View Logical AND-Gate (VLM_S1 top-q1 AND Stage-1 dec2_proto top-q2) in strict FP32."""
        import torch.nn.functional as F

        feats = ret_dict.get("all_feats", None)
        if feats is None:
            raise RuntimeError("dual_eye_and: ret_dict missing all_feats")
        if not hasattr(self, "_dec2_novel_proto"):
            raise RuntimeError("dual_eye_and: _init_dec2_eye not called before training")

        nb = int(self.cfg.data.num_bases)
        nn_cls = int(self.cfg.data.num_base_novels) - nb
        q1 = float(getattr(self.cfg, "dual_eye_q1", 0.15))
        q2 = float(getattr(self.cfg, "dual_eye_q2", 0.15))
        ema = float(getattr(self.cfg, "dual_eye_ema", 0.9))
        score_mode = str(getattr(self.cfg, "dual_eye_score", "proto"))

        with torch.cuda.amp.autocast(enabled=False):
            W = F.normalize(self.VLM_3D.task_head.cls_head.weight.float(), dim=-1)
            f_norm = F.normalize(feats.float(), dim=-1)
            cos_nv = (f_norm @ W[nb:].t()).float()
            pred = cos_nv.argmax(dim=-1) + nb
            log_p = F.log_softmax(cos_nv, dim=-1)
            s1 = (log_p.exp() * log_p).sum(dim=-1)  # neg-entropy: higher is more confident
            assert s1.dtype == torch.float32

            f_dec2 = self._extract_dec2(input_dict)
            assert f_dec2.shape[0] == s1.shape[0], f"dec2 len {f_dec2.shape[0]} != {s1.shape[0]}"

            s1_bg = s1[bg_label_mask]
            n_bg = int(s1_bg.numel())
            if n_bg > 0:
                p_nv_bg = pred[bg_label_mask] - nb
                if score_mode == "margin_nv":
                    sim_nv = (f_dec2[bg_label_mask] @ self._dec2_novel_proto.t()).float()
                    spred = sim_nv.gather(1, p_nv_bg.unsqueeze(1)).squeeze(1)
                    sim_other = sim_nv.clone().scatter_(1, p_nv_bg.unsqueeze(1), -1e9)
                    s2_bg = spred - sim_other.max(dim=1).values
                else:
                    s2_bg = (f_dec2[bg_label_mask] * self._dec2_novel_proto[p_nv_bg]).sum(dim=-1)
                assert s2_bg.dtype == torch.float32
                k1 = max(1, min(n_bg - int(round(n_bg * q1)) + 1, n_bg))
                k2 = max(1, min(n_bg - int(round(n_bg * q2)) + 1, n_bg))
                thr1_loc = s1_bg.kthvalue(k1).values.detach().float()
                thr2_loc = s2_bg.kthvalue(k2).values.detach().float()
                has_bg = 1.0
            else:
                s2_bg = torch.empty(0, dtype=torch.float32, device=s1.device)
                thr1_loc = torch.zeros((), dtype=torch.float32, device=s1.device)
                thr2_loc = torch.zeros((), dtype=torch.float32, device=s1.device)
                has_bg = 0.0

        if comm.get_world_size() > 1 and dist.is_initialized():
            buf = torch.stack([
                thr1_loc, thr2_loc,
                torch.tensor(has_bg, dtype=torch.float32, device=s1.device),
            ])
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            if buf[2].item() > 0:
                cnt_r = buf[2].clamp_min(1.0)
                b_thr1 = float((buf[0] / cnt_r).item())
                b_thr2 = float((buf[1] / cnt_r).item())
            else:
                b_thr1, b_thr2 = None, None
        else:
            b_thr1 = float(thr1_loc.item()) if has_bg > 0 else None
            b_thr2 = float(thr2_loc.item()) if has_bg > 0 else None

        if b_thr1 is not None:
            if not hasattr(self, "_dual_thr1"):
                self._dual_thr1 = b_thr1
                self._dual_thr2 = b_thr2
            else:
                self._dual_thr1 = ema * self._dual_thr1 + (1.0 - ema) * b_thr1
                self._dual_thr2 = ema * self._dual_thr2 + (1.0 - ema) * b_thr2

        thr1 = getattr(self, "_dual_thr1", float("inf"))
        thr2 = getattr(self, "_dual_thr2", float("inf"))

        out = label.clone()
        if n_bg > 0:
            keep_bg = (s1_bg >= thr1) & (s2_bg >= thr2)
            bg_idx = torch.nonzero(bg_label_mask, as_tuple=False).squeeze(1)
            kept_idx = bg_idx[keep_bg]
            out[kept_idx] = pred[kept_idx]
        else:
            keep_bg = torch.zeros(0, dtype=torch.bool, device=s1.device)
            kept_idx = torch.zeros(0, dtype=torch.long, device=s1.device)

        # Per-novel-class retention tracker (printed at Ep1 and every 5 epochs)
        if not hasattr(self, "_dual_cls_keep"):
            self._dual_cls_keep = torch.zeros(nn_cls, dtype=torch.long, device=s1.device)
            self._dual_cls_cand = torch.zeros(nn_cls, dtype=torch.long, device=s1.device)
            self._dual_bg_tot = torch.zeros(1, dtype=torch.long, device=s1.device)
            self._dual_gt_tot = torch.zeros(1, dtype=torch.long, device=s1.device)
            self._dual_step = 0
        if n_bg > 0:
            self._dual_cls_cand += torch.bincount(pred[bg_label_mask] - nb, minlength=nn_cls)
            if kept_idx.numel() > 0:
                self._dual_cls_keep += torch.bincount(pred[kept_idx] - nb, minlength=nn_cls)
        self._dual_bg_tot += n_bg
        self._dual_gt_tot += int((~bg_label_mask).sum().item())
        self._dual_step += 1

        ep_len = len(self.train_loader) if hasattr(self, "train_loader") else 0
        if ep_len > 0 and (self._dual_step % ep_len == 0):
            cur_ep = self._dual_step // ep_len
            if cur_ep == 1 or (cur_ep % 5 == 0):
                ck = self._dual_cls_keep.clone()
                cc = self._dual_cls_cand.clone()
                bg_t = self._dual_bg_tot.clone()
                gt_t = self._dual_gt_tot.clone()
                if comm.get_world_size() > 1 and dist.is_initialized():
                    dist.all_reduce(ck)
                    dist.all_reduce(cc)
                    dist.all_reduce(bg_t)
                    dist.all_reduce(gt_t)
                if comm.is_main_process():
                    tot_k = int(ck.sum().item())
                    tot_b = int(bg_t.item())
                    self.logger.info(
                        "[DualEyeAND ep=%d] mode=%s thr1=%.6f thr2=%.4f q=(%.2f,%.2f) keep=%d/%d (%.2f%%) GT=%d"
                        % (cur_ep, score_mode, thr1, thr2, q1, q2, tot_k, tot_b,
                           100.0 * tot_k / max(tot_b, 1), int(gt_t.item()))
                    )
                    if cur_ep == 1:
                        names = getattr(self.cfg.data, "names", None)
                        for idx in range(nn_cls):
                            cid = nb + idx
                            nm = names[cid] if (names and cid < len(names)) else f"cls{cid}"
                            k_i = int(ck[idx].item())
                            c_i = int(cc[idx].item())
                            self.logger.info(
                                "  Class_%02d %-15s: keep=%7d / cand=%7d (%5.1f%%)"
                                % (cid, nm, k_i, c_i, 100.0 * k_i / max(c_i, 1))
                            )
            self._dual_cls_keep.zero_()
            self._dual_cls_cand.zero_()
            self._dual_bg_tot.zero_()
            self._dual_gt_tot.zero_()

        return out

    @torch.no_grad()
    def _calibrate_h1_vlm_thresholds(self):
        """Pre-calibrate cross-scene global Otsu(S1) & per-class top-q S1 thresholds in strict FP32."""
        import numpy as np
        import torch.nn.functional as F

        nb = int(self.cfg.data.num_bases)
        nn_cls = int(self.cfg.data.num_base_novels) - nb
        q_cls = float(getattr(self.cfg, "dual_eye_h1_q", 0.10))
        max_batches = int(getattr(self.cfg, "dual_eye_h1_cal_batches", 30))

        glob_samples = []
        cls_samples = [[] for _ in range(nn_cls)]
        gen = torch.Generator(device="cuda").manual_seed(2026 + comm.get_rank())

        with torch.cuda.amp.autocast(enabled=False):
            W = F.normalize(self.VLM_3D.task_head.cls_head.weight.float(), dim=-1)
            n_seen = 0
            for b_idx, batch in enumerate(self.train_loader):
                if b_idx >= max_batches:
                    break
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.cuda(non_blocking=True)
                bg_mask = (batch["segment"] == -1)
                if not bg_mask.any():
                    continue
                vd = self.construct_3D_vlm_input(batch)
                ret = self.VLM_3D(vd)
                f_norm = F.normalize(ret["all_feats"].float(), dim=-1)
                cos_nv = (f_norm @ W[nb:].t()).float()
                pred_nv = cos_nv.argmax(dim=-1)
                log_p = F.log_softmax(cos_nv, dim=-1)
                s1 = (log_p.exp() * log_p).sum(dim=-1)
                assert s1.dtype == torch.float32

                s1_bg = s1[bg_mask]
                p_bg = pred_nv[bg_mask]
                n_bg = int(s1_bg.numel())
                if n_bg > 8192:
                    idx = torch.randint(0, n_bg, (8192,), generator=gen, device=s1.device)
                    glob_samples.append(s1_bg[idx].cpu().numpy().astype(np.float32))
                else:
                    glob_samples.append(s1_bg.cpu().numpy().astype(np.float32))

                for c in range(nn_cls):
                    mc = (p_bg == c)
                    nc_i = int(mc.sum().item())
                    if nc_i == 0:
                        continue
                    sc = s1_bg[mc]
                    if nc_i > 2048:
                        idx_c = torch.randint(0, nc_i, (2048,), generator=gen, device=s1.device)
                        cls_samples[c].append(sc[idx_c].cpu().numpy().astype(np.float32))
                    else:
                        cls_samples[c].append(sc.cpu().numpy().astype(np.float32))
                n_seen += 1

        loc_glob = np.concatenate(glob_samples) if glob_samples else np.empty(0, dtype=np.float32)
        loc_cls = [np.concatenate(v) if v else np.empty(0, dtype=np.float32) for v in cls_samples]

        if comm.get_world_size() > 1 and dist.is_initialized():
            gathered = comm.gather((loc_glob, loc_cls), dst=0)
        else:
            gathered = [(loc_glob, loc_cls)]

        thr_buf = torch.zeros(1 + nn_cls, dtype=torch.float32, device="cuda")
        if comm.is_main_process():
            all_g = np.concatenate([g[0] for g in gathered if len(g[0]) > 0]).astype(np.float64)
            # Adaptive [p0.1, p99.9] 512-bin FP32/FP64 Otsu
            lo, hi = np.percentile(all_g, [0.1, 99.9])
            hist, edges = np.histogram(np.clip(all_g, lo, hi), bins=512, range=(lo, hi))
            hist = hist.astype(np.float64)
            centers = 0.5 * (edges[:-1] + edges[1:])
            w0 = np.cumsum(hist)
            w1 = hist.sum() - w0
            m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-12)
            m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1e-12)
            var_b = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1]) ** 2
            otsu_thr = float(edges[int(np.argmax(var_b)) + 1])
            otsu_pct = 100.0 * float((all_g >= otsu_thr).mean())
            thr_buf[0] = otsu_thr

            for c in range(nn_cls):
                arrs = [g[1][c] for g in gathered if len(g[1][c]) > 0]
                if arrs:
                    ac = np.concatenate(arrs).astype(np.float64)
                    thr_buf[1 + c] = float(np.quantile(ac, 1.0 - q_cls))
                else:
                    thr_buf[1 + c] = -1e9
            self.logger.info(
                f"=> [DualEyeH1] Calibrated on {n_seen}x{comm.get_world_size()} batches ({len(all_g)} pts): "
                f"Otsu(S1)={otsu_thr:.6f} (top {otsu_pct:.2f}%), range=[{lo:.6f}, {hi:.6f}], per_cls_q={q_cls:.2f}"
            )

        if comm.get_world_size() > 1 and dist.is_initialized():
            dist.broadcast(thr_buf, src=0)

        self._h1_otsu_thr = float(thr_buf[0].item())
        self._h1_cls_thr = thr_buf[1:].contiguous()
        for attr in ("_h1_cls_keep", "_h1_cls_cand", "_h1_bg_tot", "_h1_gt_tot", "_h1_step"):
            if hasattr(self, attr):
                delattr(self, attr)

    @torch.no_grad()
    def _target_dual_eye_h1(self, ret_dict, input_dict, label, bg_label_mask):
        """H1: Global Otsu(S1) floor AND Per-Class S1 top-q AND dec2 geometric vote (dec2_m_nv > 0) in FP32."""
        import torch.nn.functional as F

        feats = ret_dict.get("all_feats", None)
        if feats is None:
            raise RuntimeError("dual_eye_h1: ret_dict missing all_feats")
        if not hasattr(self, "_dec2_novel_proto") or not hasattr(self, "_h1_cls_thr"):
            raise RuntimeError("dual_eye_h1: _init_dec2_eye / _calibrate_h1_vlm_thresholds not called")

        nb = int(self.cfg.data.num_bases)
        nn_cls = int(self.cfg.data.num_base_novels) - nb

        with torch.cuda.amp.autocast(enabled=False):
            W = F.normalize(self.VLM_3D.task_head.cls_head.weight.float(), dim=-1)
            f_norm = F.normalize(feats.float(), dim=-1)
            cos_nv = (f_norm @ W[nb:].t()).float()
            pred = cos_nv.argmax(dim=-1) + nb
            log_p = F.log_softmax(cos_nv, dim=-1)
            s1 = (log_p.exp() * log_p).sum(dim=-1)
            assert s1.dtype == torch.float32

            f_dec2 = self._extract_dec2(input_dict)
            assert f_dec2.shape[0] == s1.shape[0]

            s1_bg = s1[bg_label_mask]
            n_bg = int(s1_bg.numel())
            out = label.clone()
            if n_bg > 0:
                p_nv_bg = pred[bg_label_mask] - nb
                sim_nv = (f_dec2[bg_label_mask] @ self._dec2_novel_proto.t()).float()
                spred = sim_nv.gather(1, p_nv_bg.unsqueeze(1)).squeeze(1)
                sim_other = sim_nv.clone().scatter_(1, p_nv_bg.unsqueeze(1), -1e9)
                m_nv_bg = spred - sim_other.max(dim=1).values
                assert m_nv_bg.dtype == torch.float32

                keep_bg = (
                    (s1_bg >= self._h1_otsu_thr)
                    & (s1_bg >= self._h1_cls_thr[p_nv_bg])
                    & (m_nv_bg > 0.0)
                )
                bg_idx = torch.nonzero(bg_label_mask, as_tuple=False).squeeze(1)
                kept_idx = bg_idx[keep_bg]
                out[kept_idx] = pred[kept_idx]
            else:
                kept_idx = torch.zeros(0, dtype=torch.long, device=s1.device)

        if not hasattr(self, "_h1_cls_keep"):
            self._h1_cls_keep = torch.zeros(nn_cls, dtype=torch.long, device=s1.device)
            self._h1_cls_cand = torch.zeros(nn_cls, dtype=torch.long, device=s1.device)
            self._h1_bg_tot = torch.zeros(1, dtype=torch.long, device=s1.device)
            self._h1_gt_tot = torch.zeros(1, dtype=torch.long, device=s1.device)
            self._h1_step = 0
        if n_bg > 0:
            self._h1_cls_cand += torch.bincount(pred[bg_label_mask] - nb, minlength=nn_cls)
            if kept_idx.numel() > 0:
                self._h1_cls_keep += torch.bincount(pred[kept_idx] - nb, minlength=nn_cls)
        self._h1_bg_tot += n_bg
        self._h1_gt_tot += int((~bg_label_mask).sum().item())
        self._h1_step += 1

        ep_len = len(self.train_loader) if hasattr(self, "train_loader") else 0
        if ep_len > 0 and (self._h1_step % ep_len == 0):
            cur_ep = self._h1_step // ep_len
            if cur_ep == 1 or (cur_ep % 5 == 0):
                ck = self._h1_cls_keep.clone()
                cc = self._h1_cls_cand.clone()
                bg_t = self._h1_bg_tot.clone()
                gt_t = self._h1_gt_tot.clone()
                if comm.get_world_size() > 1 and dist.is_initialized():
                    dist.all_reduce(ck)
                    dist.all_reduce(cc)
                    dist.all_reduce(bg_t)
                    dist.all_reduce(gt_t)
                if comm.is_main_process():
                    tot_k = int(ck.sum().item())
                    tot_b = int(bg_t.item())
                    self.logger.info(
                        "[DualEyeH1 ep=%d] otsu=%.6f keep=%d/%d (%.2f%%) GT=%d"
                        % (cur_ep, self._h1_otsu_thr, tot_k, tot_b,
                           100.0 * tot_k / max(tot_b, 1), int(gt_t.item()))
                    )
                    if cur_ep == 1:
                        names = getattr(self.cfg.data, "names", None)
                        for idx in range(nn_cls):
                            cid = nb + idx
                            nm = names[cid] if (names and cid < len(names)) else f"cls{cid}"
                            k_i = int(ck[idx].item())
                            c_i = int(cc[idx].item())
                            self.logger.info(
                                "  Class_%02d %-15s: keep=%7d / cand=%7d (%5.1f%%)"
                                % (cid, nm, k_i, c_i, 100.0 * k_i / max(c_i, 1))
                            )
            self._h1_cls_keep.zero_()
            self._h1_cls_cand.zero_()
            self._h1_bg_tot.zero_()
            self._h1_gt_tot.zero_()

        return out

    @torch.no_grad()

    def _dec2_rank_prune(self, prediction, input_dict, bg_label_mask):
        """Prune official PS+AI pseudo-labels whose Stage-1 dec2 rank exceeds ceil(alpha * N_novel)."""
        import math
        if not hasattr(self, "_dec2_novel_proto"):
            raise RuntimeError("psai_dec2_rank: _init_dec2_eye not called before training")

        nb = int(self.cfg.data.num_bases)
        nn_cls = int(self.cfg.data.num_base_novels) - nb
        alpha = float(getattr(self.cfg, "dual_eye_rank_alpha", 0.10))
        k_max = max(1, int(math.ceil(alpha * nn_cls)))

        with torch.cuda.amp.autocast(enabled=False):
            cand = bg_label_mask & (prediction >= nb)
            out = prediction.clone()
            n_bg = int(bg_label_mask.sum().item())
            n_cand = int(cand.sum().item())
            if n_cand > 0:
                f_dec2 = self._extract_dec2(input_dict)
                assert f_dec2.shape[0] == prediction.shape[0]
                p_nv = prediction[cand].long() - nb
                sim_nv = (f_dec2[cand] @ self._dec2_novel_proto.t()).float()
                spred = sim_nv.gather(1, p_nv.unsqueeze(1))
                ranks = 1 + (sim_nv > spred).sum(dim=1)
                cand_idx = torch.nonzero(cand, as_tuple=False).squeeze(1)
                out[cand_idx[ranks > k_max]] = -1
                kept_mask = ranks <= k_max
                kept_p_nv = p_nv[kept_mask]
            else:
                kept_p_nv = torch.zeros(0, dtype=torch.long, device=prediction.device)

        if not hasattr(self, "_pdr_cls_keep"):
            self._pdr_cls_keep = torch.zeros(nn_cls, dtype=torch.long, device=prediction.device)
            self._pdr_cls_cand = torch.zeros(nn_cls, dtype=torch.long, device=prediction.device)
            self._pdr_bg_tot = torch.zeros(1, dtype=torch.long, device=prediction.device)
            self._pdr_gt_tot = torch.zeros(1, dtype=torch.long, device=prediction.device)
            self._pdr_step = 0

        if n_cand > 0:
            self._pdr_cls_cand += torch.bincount(p_nv, minlength=nn_cls)
            if kept_p_nv.numel() > 0:
                self._pdr_cls_keep += torch.bincount(kept_p_nv, minlength=nn_cls)
        self._pdr_bg_tot += n_bg
        self._pdr_gt_tot += int((~bg_label_mask).sum().item())
        self._pdr_step += 1

        ep_len = len(self.train_loader) if hasattr(self, "train_loader") else 0
        if ep_len > 0 and (self._pdr_step % ep_len == 0):
            cur_ep = self._pdr_step // ep_len
            if cur_ep == 1 or (cur_ep % 5 == 0):
                ck = self._pdr_cls_keep.clone()
                cc = self._pdr_cls_cand.clone()
                bg_t = self._pdr_bg_tot.clone()
                gt_t = self._pdr_gt_tot.clone()
                if comm.get_world_size() > 1 and dist.is_initialized():
                    dist.all_reduce(ck)
                    dist.all_reduce(cc)
                    dist.all_reduce(bg_t)
                    dist.all_reduce(gt_t)
                if comm.is_main_process():
                    tot_k = int(ck.sum().item())
                    tot_c = int(cc.sum().item())
                    tot_b = int(bg_t.item())
                    self.logger.info(
                        "[PSAI+Dec2Rank ep=%d] alpha=%.2f k=%d/%d cand=%d (%.2f%%) keep=%d/%d (%.2f%%) GT=%d"
                        % (cur_ep, alpha, k_max, nn_cls, tot_c, 100.0 * tot_c / max(tot_b, 1),
                           tot_k, tot_b, 100.0 * tot_k / max(tot_b, 1), int(gt_t.item()))
                    )
                    if cur_ep == 1:
                        ds = self.train_loader.dataset
                        true_names = list(ds.base_class_names) + list(ds.novel_class_names)
                        parts = []
                        for ci in range(nn_cls):
                            nm = true_names[nb + ci] if (nb + ci) < len(true_names) else f"c{nb+ci}"
                            parts.append(f"{nm}={int(ck[ci])}/{int(cc[ci])}")
                        self.logger.info("[PSAI+Dec2Rank ep=1 per_cls] " + ", ".join(parts))
            self._pdr_cls_keep.zero_()
            self._pdr_cls_cand.zero_()
            self._pdr_bg_tot.zero_()
            self._pdr_gt_tot.zero_()

        return out

    def _target_dual_eye_m1_vote(self, ret_dict, input_dict, label, bg_label_mask):
        """Zero-hyperparam, zero-quota dual-view consensus: M1 full-space VLM novel hit AND Stage-1 dec2_m_nv > margin."""
        import torch.nn.functional as F

        feats = ret_dict.get("all_feats", None)
        if feats is None:
            raise RuntimeError("dual_eye_m1_vote: ret_dict missing all_feats")
        if not hasattr(self, "_dec2_novel_proto"):
            raise RuntimeError("dual_eye_m1_vote: _init_dec2_eye not called before training")

        nb = int(self.cfg.data.num_bases)
        nn_cls = int(self.cfg.data.num_base_novels) - nb
        margin_thr = float(getattr(self.cfg, "dual_eye_m1_margin", 0.0))

        with torch.cuda.amp.autocast(enabled=False):
            W = F.normalize(self.VLM_3D.task_head.cls_head.weight.float(), dim=-1)
            f_norm = F.normalize(feats.float(), dim=-1)
            pred_all = (f_norm @ W.t()).argmax(dim=-1)
            m1_bg = bg_label_mask & (pred_all >= nb)

            out = label.clone()
            n_bg = int(bg_label_mask.sum().item())
            n_m1 = int(m1_bg.sum().item())
            if n_m1 > 0:
                f_dec2 = self._extract_dec2(input_dict)
                assert f_dec2.shape[0] == pred_all.shape[0]
                p_nv = pred_all[m1_bg] - nb
                sim_nv = (f_dec2[m1_bg] @ self._dec2_novel_proto.t()).float()
                spred = sim_nv.gather(1, p_nv.unsqueeze(1)).squeeze(1)
                sim_other = sim_nv.clone().scatter_(1, p_nv.unsqueeze(1), -1e9)
                m_nv = spred - sim_other.max(dim=1).values
                assert m_nv.dtype == torch.float32

                agree = (m_nv > margin_thr)
                m1_idx = torch.nonzero(m1_bg, as_tuple=False).squeeze(1)
                kept_idx = m1_idx[agree]
                out[kept_idx] = pred_all[kept_idx]
            else:
                kept_idx = torch.zeros(0, dtype=torch.long, device=pred_all.device)

        if not hasattr(self, "_m1v_cls_keep"):
            self._m1v_cls_keep = torch.zeros(nn_cls, dtype=torch.long, device=pred_all.device)
            self._m1v_cls_cand = torch.zeros(nn_cls, dtype=torch.long, device=pred_all.device)
            self._m1v_bg_tot = torch.zeros(1, dtype=torch.long, device=pred_all.device)
            self._m1v_gt_tot = torch.zeros(1, dtype=torch.long, device=pred_all.device)
            self._m1v_step = 0
        if n_m1 > 0:
            self._m1v_cls_cand += torch.bincount(pred_all[m1_bg] - nb, minlength=nn_cls)
            if kept_idx.numel() > 0:
                self._m1v_cls_keep += torch.bincount(pred_all[kept_idx] - nb, minlength=nn_cls)
        self._m1v_bg_tot += n_bg
        self._m1v_gt_tot += int((~bg_label_mask).sum().item())
        self._m1v_step += 1

        ep_len = len(self.train_loader) if hasattr(self, "train_loader") else 0
        if ep_len > 0 and (self._m1v_step % ep_len == 0):
            cur_ep = self._m1v_step // ep_len
            if cur_ep == 1 or (cur_ep % 5 == 0):
                ck = self._m1v_cls_keep.clone()
                cc = self._m1v_cls_cand.clone()
                bg_t = self._m1v_bg_tot.clone()
                gt_t = self._m1v_gt_tot.clone()
                if comm.get_world_size() > 1 and dist.is_initialized():
                    dist.all_reduce(ck)
                    dist.all_reduce(cc)
                    dist.all_reduce(bg_t)
                    dist.all_reduce(gt_t)
                if comm.is_main_process():
                    tot_k = int(ck.sum().item())
                    tot_c = int(cc.sum().item())
                    tot_b = int(bg_t.item())
                    self.logger.info(
                        "[DualEyeM1Vote ep=%d] margin=%.3f m1_cand=%d (%.2f%%) keep=%d/%d (%.2f%%) GT=%d"
                        % (cur_ep, margin_thr, tot_c, 100.0 * tot_c / max(tot_b, 1),
                           tot_k, tot_b, 100.0 * tot_k / max(tot_b, 1), int(gt_t.item()))
                    )
                    if cur_ep == 1:
                        names = getattr(self.cfg.data, "names", None)
                        for idx in range(nn_cls):
                            cid = nb + idx
                            nm = names[cid] if (names and cid < len(names)) else f"cls{cid}"
                            k_i = int(ck[idx].item())
                            c_i = int(cc[idx].item())
                            self.logger.info(
                                "  Class_%02d %-15s: keep=%7d / m1_cand=%7d (%5.1f%%)"
                                % (cid, nm, k_i, c_i, 100.0 * k_i / max(c_i, 1))
                            )
            self._m1v_cls_keep.zero_()
            self._m1v_cls_cand.zero_()
            self._m1v_bg_tot.zero_()
            self._m1v_gt_tot.zero_()

        return out

    def _target_global_entropy_quantile(self, ret_dict, input_dict, label,
                                        bg_label_mask):
        """Cross-scene global EMA entropy quantile filter in strict FP32."""
        import torch.distributed as dist

        feats = ret_dict.get("all_feats", None)
        if feats is None:
            raise RuntimeError("global_entropy_q: ret_dict missing all_feats")

        nb = int(self.cfg.data.num_bases)
        nn_cls = int(self.cfg.data.num_base_novels) - nb
        rho = float(getattr(self.cfg, "global_entropy_rho", 0.02))
        ema = float(getattr(self.cfg, "global_entropy_ema", 0.9))

        with torch.cuda.amp.autocast(enabled=False):
            W = self.VLM_3D.task_head.cls_head.weight.float()
            f_norm = torch.nn.functional.normalize(feats.float(), dim=-1)
            cos_nv = (f_norm @ W[nb:].t()).float()
            pred = cos_nv.argmax(dim=-1) + nb
            log_p = torch.nn.functional.log_softmax(cos_nv, dim=-1)
            ent = -(log_p.exp() * log_p).sum(dim=-1)
            assert ent.dtype == torch.float32

        eu_bg = ent[bg_label_mask]
        if eu_bg.numel() > 0:
            k = max(1, min(int(round(eu_bg.numel() * rho)), eu_bg.numel()))
            thr_local = eu_bg.kthvalue(k).values.detach().float()
            has_bg = 1.0
        else:
            thr_local = torch.zeros((), dtype=torch.float32, device=ent.device)
            has_bg = 0.0

        if comm.get_world_size() > 1 and dist.is_initialized():
            buf = torch.stack([thr_local, torch.tensor(has_bg, dtype=torch.float32, device=ent.device)])
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            thr_batch = float((buf[0] / buf[1].clamp_min(1.0)).item()) if buf[1].item() > 0 else None
        else:
            thr_batch = float(thr_local.item()) if has_bg > 0 else None

        if thr_batch is not None:
            if not hasattr(self, "_global_ent_thr"):
                self._global_ent_thr = thr_batch
            else:
                self._global_ent_thr = ema * self._global_ent_thr + (1.0 - ema) * thr_batch

        thr = getattr(self, "_global_ent_thr", float("-inf"))
        keep_mask = bg_label_mask & (ent <= thr)

        out = label.clone()
        out[keep_mask] = pred[keep_mask]

        # Per-novel-class retention tracker (printed at end of Ep1 and every 5 epochs)
        if not hasattr(self, "_geq_cls_keep"):
            self._geq_cls_keep = torch.zeros(nn_cls, dtype=torch.long, device=ent.device)
            self._geq_cls_cand = torch.zeros(nn_cls, dtype=torch.long, device=ent.device)
            self._geq_bg_tot = torch.zeros(1, dtype=torch.long, device=ent.device)
            self._geq_gt_tot = torch.zeros(1, dtype=torch.long, device=ent.device)
            self._geq_step = 0
        nv_cand = pred[bg_label_mask] - nb
        nv_keep = pred[keep_mask] - nb
        if nv_cand.numel() > 0:
            self._geq_cls_cand += torch.bincount(nv_cand, minlength=nn_cls)
        if nv_keep.numel() > 0:
            self._geq_cls_keep += torch.bincount(nv_keep, minlength=nn_cls)
        self._geq_bg_tot += int(bg_label_mask.sum().item())
        self._geq_gt_tot += int((~bg_label_mask).sum().item())
        self._geq_step += 1

        ep_len = len(self.train_loader) if hasattr(self, "train_loader") else 0
        if ep_len > 0 and (self._geq_step % ep_len == 0):
            cur_ep = self._geq_step // ep_len
            if cur_ep == 1 or (cur_ep % 5 == 0):
                ck = self._geq_cls_keep.clone()
                cc = self._geq_cls_cand.clone()
                bg_t = self._geq_bg_tot.clone()
                gt_t = self._geq_gt_tot.clone()
                if comm.get_world_size() > 1 and dist.is_initialized():
                    dist.all_reduce(ck)
                    dist.all_reduce(cc)
                    dist.all_reduce(bg_t)
                    dist.all_reduce(gt_t)
                if comm.is_main_process():
                    tot_k = int(ck.sum().item())
                    tot_b = int(bg_t.item())
                    self.logger.info(
                        "[GlobalEntQ ep=%d] thr=%.6f rho=%.3f keep=%d/%d (%.2f%%) GT=%d"
                        % (cur_ep, thr, rho, tot_k, tot_b,
                           100.0 * tot_k / max(tot_b, 1), int(gt_t.item()))
                    )
                    if cur_ep == 1:
                        names = getattr(self.cfg.data, "names", None)
                        for idx in range(nn_cls):
                            cid = nb + idx
                            nm = names[cid] if (names and cid < len(names)) else f"cls{cid}"
                            k_i = int(ck[idx].item())
                            c_i = int(cc[idx].item())
                            self.logger.info(
                                "  Class_%02d %-15s: keep=%7d / cand=%7d (%5.1f%%)"
                                % (cid, nm, k_i, c_i, 100.0 * k_i / max(c_i, 1))
                            )
            self._geq_cls_keep.zero_()
            self._geq_cls_cand.zero_()
            self._geq_bg_tot.zero_()
            self._geq_gt_tot.zero_()

        return out

    @torch.no_grad()
    def _target_entropy_quantile(self, ret_dict, input_dict, label,
                                 bg_label_mask, seg_logits=None):
        """UGA UB-Filter：class-wise 熵分位过滤，跳过 PS 与 AI。

        熵用 VLM 原生 seg_scores（已乘预训练 logit_scale），无额外温度。
        分位在 batch 内每类独立计算：保留熵最低的 entropy_keep_rho 比例。
        """
        # 用余弦 logits 算熵：原始 seg_scores = cls_head(feats)，feats 未归一化
        # 实测 Spearman(||f||, entropy) = -0.928，未归一化的熵在按特征模长排序，
        # 不是语义不确定性。cls_head 行范数=1.0，所以余弦化不改变 argmax。
        feats = ret_dict.get("all_feats", None)
        if feats is None:
            raise RuntimeError("1C: ret_dict 缺少 all_feats")
        W = self.VLM_3D.task_head.cls_head.weight
        cos_logits = torch.nn.functional.normalize(
            feats.float(), dim=-1) @ W.t()

        nb = int(self.cfg.data.num_bases)
        rho = float(getattr(self.cfg, "entropy_keep_rho", 0.1))
        # 1C-g: 全局绝对阈值（跨类统一）。设为 float 时忽略 rho，
        # 只在 ent <= entropy_abs_thresh 处写入伪标签。
        # 理由：每类 top-k 保留率是常数，对正样本率 < 保留率的稀疏类
        # （smoke 2.6%、paper 1.2%）纯度上界被锁死，排序再准也拿不到点。
        # 熵的绝对尺度类间差异只有 0.0004，远小于类内动态范围 0.0021，
        # 因此跨类统一阈值在数学上是合法的。
        _abs_thr = getattr(self.cfg, "entropy_abs_thresh", None)
        _abs_thr = None if _abs_thr is None else float(_abs_thr)

        # ---- 3D veto 通用参数（Exp B-fix1/fix2）----
        # veto_warmup_epochs: 前 N 轮不开 veto，给 novel 头热身。
        # Epoch 0 时 novel_seg_head 是 Kaiming 随机初始化（std~0.125），
        # 而 base 头来自 800 轮预训练，所以 base_p 接近 1，冷启动必杀。
        # veto_stage: "pre" = 先否决再取分位（不占用 top-k 配额）；
        #             "post" = 先取分位再否决（Exp B 原顺序）。
        _warm_veto = int(getattr(self.cfg, "veto_warmup_epochs", 0))
        _tau3 = float(getattr(self.cfg, "veto_tau", 0.7))
        _use_veto = (seg_logits is not None
                     and bool(getattr(self.cfg, "use_3d_veto", False))
                     and self.epoch >= _warm_veto)
        _veto_pre = str(getattr(self.cfg, "veto_stage", "post")) == "pre"

        # 只在 novel 子集里 argmax。实测：GT novel 点的 Top-1 有 49.5% 被
        # 12 个 base 类抢走，屏蔽后 Top-1 从 0.4734 跳到 0.7943（+32.1pp）。
        # bg_label_mask 区域（label==-1）理论上不存在 base 类，所以屏蔽零成本。
        cos_nv = cos_logits[:, nb:]               # (N, num_novel)
        pred = cos_nv.argmax(dim=-1) + nb

        # fix2 (veto_stage="pre")：先把高置信 base 点从候选里剔掉，
        # 再按熵取分位。这样被否决的点不再占用该类 top-k 的配额。
        if _use_veto and _veto_pre:
            _sp = torch.softmax(seg_logits.float(), dim=-1)
            _base_p = _sp[:, :nb].max(dim=-1).values
            _vm = (_base_p > _tau3) & bg_label_mask
            pred[_vm] = -1
            if not hasattr(self, "_preveto_stats"):
                self._preveto_stats = dict(veto=0, cand=0, n=0)
            st = self._preveto_stats
            st["veto"] += int(_vm.sum())
            st["cand"] += int(bg_label_mask.sum())
            st["n"] += 1
            if st["n"] % 20 == 0:
                self.logger.info(
                    "[VetoPre] tau=%.2f ep=%d 累计 veto_bg=%d / cand_bg=%d (%.1f%%)"
                    % (_tau3, self.epoch, st["veto"], st["cand"],
                       100.0 * st["veto"] / max(st["cand"], 1)))

        p = torch.softmax(cos_nv, dim=-1)
        ent = -(p * (p + 1e-8).log()).sum(dim=-1)

        offset = [int(x) for x in input_dict["offset"].tolist()]
        out = []
        n_keep = n_drop = 0
        for s, e in zip([0] + offset[:-1], offset):
            pc = pred[s:e].clone()
            eu = ent[s:e]
            for c in pc.unique():
                ci = int(c.item())
                if ci < nb:
                    continue
                # 只在伪标签真正写入的区域算分位。
                # 否则 GT base（60%）与 support GT（17%）会在 cos_nv 上
                # 低熵地映射到某个 novel 类，吃掉该类 k=int(n*rho) 的配额。
                m = (pc == c) & bg_label_mask[s:e]
                n = int(m.sum())
                if n <= 1:
                    continue
                if _abs_thr is not None:
                    drop = m & (eu > _abs_thr)
                else:
                    k = max(1, int(n * rho))
                    thr = torch.topk(eu[m], k, largest=False).values[-1]
                    drop = m & (eu > thr)
                nd = int(drop.sum())
                n_drop += nd
                n_keep += n - nd
                pc[drop] = -1
            out.append(pc)
        pred = torch.cat(out, dim=0)

        if not hasattr(self, "_ent_stats"):
            self._ent_stats = dict(pl_keep=0, pl_drop=0, gt_keep=0, n_log=0)
        self._ent_stats["pl_keep"] += n_keep
        self._ent_stats["pl_drop"] += n_drop
        self._ent_stats["gt_keep"] += int((~bg_label_mask).sum())
        self._ent_stats["n_log"] += 1
        if self._ent_stats["n_log"] % 20 == 0:
            self.logger.info(
                "[1C] 累计 keep=%d drop=%d (%s) | GT 保留=%d"
                % (self._ent_stats["pl_keep"], self._ent_stats["pl_drop"],
                   ("abs=%.6f" % _abs_thr) if _abs_thr is not None
                   else ("topk rho=%.2f" % rho),
                   self._ent_stats["gt_keep"]))

        # fix1 (veto_stage="post")：分位过滤之后再否决。
        # 统计口径修正：Exp B 的旧版没乘 bg_label_mask，把 60% 的 GT base
        # 点和 support 点也计入了 veto，导致 veto_share 虚高到 0.84。
        # 真正有诊断价值的指标是「bg 区域内伪标签被否决的比例」。
        if _use_veto and not _veto_pre:
            _sp = torch.softmax(seg_logits.float(), dim=-1)
            _base_p = _sp[:, :nb].max(dim=-1).values
            _veto = (_base_p > _tau3) & (pred >= nb)
            _pl_before = int(((pred >= nb) & bg_label_mask).sum())
            _veto_bg = int((_veto & bg_label_mask).sum())
            pred[_veto] = -1
            _pl_after = int(((pred >= nb) & bg_label_mask).sum())
            if not hasattr(self, "_veto_stats"):
                self._veto_stats = dict(veto=0, veto_bg=0, pl_b=0, pl_a=0, n=0)
            st = self._veto_stats
            st["veto"] += int(_veto.sum())
            st["veto_bg"] += _veto_bg
            st["pl_b"] += _pl_before
            st["pl_a"] += _pl_after
            st["n"] += 1
            if st["n"] % 20 == 0:
                self.logger.info(
                    "[Veto] tau=%.2f ep=%d 伪标签 %d->%d (否决 %.1f%%) | 全域veto=%d"
                    % (_tau3, self.epoch, st["pl_b"], st["pl_a"],
                       100.0 * (1.0 - st["pl_a"] / max(st["pl_b"], 1)),
                       st["veto"]))

        out_label = label.clone()
        out_label[bg_label_mask] = pred[bg_label_mask]
        return out_label

    @torch.no_grad()
    def _entropy_prune(self, prediction, ret_dict, input_dict, bg_label_mask, offset):
        """对 PS+AI 已生成的预测做按类熵分位裁剪。

        与 _target_entropy_quantile 的区别：不重新 argmax、不屏蔽 base，
        只在 bg 区域按「已预测的类」分组，保留熵最低的 entropy_keep_rho 比例。
        """
        feats = ret_dict.get("all_feats", None)
        if feats is None:
            return prediction
        W = self.VLM_3D.task_head.cls_head.weight
        cos_logits = torch.nn.functional.normalize(
            feats.float(), dim=-1) @ W.t()
        nb = int(self.cfg.data.num_bases)
        rho = float(getattr(self.cfg, "entropy_keep_rho", 0.1))

        cos_nv = cos_logits[:, nb:]
        p = torch.softmax(cos_nv, dim=-1)
        ent = -(p * (p + 1e-8).log()).sum(-1)

        out = prediction.clone()
        n_keep = n_drop = 0
        for s, e in zip([0] + offset[:-1], offset):
            pc = out[s:e]
            eu = ent[s:e]
            bg = bg_label_mask[s:e]
            for c in pc.unique():
                ci = int(c.item())
                if ci < nb:
                    continue
                m = (pc == c) & bg
                n = int(m.sum())
                if n <= 1:
                    continue
                k = max(1, int(n * rho))
                thr = torch.topk(eu[m], k, largest=False).values[-1]
                drop = m & (eu > thr)
                nd = int(drop.sum())
                n_drop += nd
                n_keep += n - nd
                pc[drop] = -1

        _k = getattr(self, "_cc_keep", 0) + n_keep
        _d = getattr(self, "_cc_drop", 0) + n_drop
        _n = getattr(self, "_cc_n", 0) + 1
        self._cc_keep, self._cc_drop, self._cc_n = _k, _d, _n
        if _n % 20 == 0:
            self.logger.info(
                "[1C-c] PS+AI 后累计 keep=%d drop=%d (rho=%.2f, keep_ratio=%.2f)"
                % (_k, _d, rho, _k / max(_k + _d, 1)))
        return out

    @torch.no_grad()
    def _student_consensus_pl(self, input_dict, ret_dict, seg_logits=None):
        """学生 × VLM Top-K 双向共识。

        只采纳同时满足三个条件的点：学生置信、学生判 novel、学生预测落在
        VLM 的 Top-K 候选内。学生判 base 的点自然被排除（s_pred < nb）。
        """
        nb = int(self.cfg.data.num_bases)
        tau = float(getattr(self.cfg, "stu_tau", 0.4))
        K = int(getattr(self.cfg, "stu_topk", 2))

        label = input_dict["segment"].clone()
        bg = label == -1

        feats = ret_dict.get("all_feats", None)
        if feats is None:
            return label
        W = self.VLM_3D.task_head.cls_head.weight
        cos_logits = torch.nn.functional.normalize(
            feats.float(), dim=-1) @ W.t()
        cos_nv = cos_logits[:, nb:]
        kk = min(K, cos_nv.shape[1])
        cand = cos_nv.topk(kk, dim=-1).indices + nb          # (N, K)

        if seg_logits is None:
            s_out = self.model(input_dict)
            s_logits = s_out["seg_logits"] if isinstance(s_out, dict) else s_out
        else:
            s_logits = seg_logits
        s_prob = torch.softmax(s_logits.float(), dim=-1)
        s_conf, s_pred = s_prob.max(dim=-1)

        in_cand = (s_pred.unsqueeze(1) == cand).any(dim=1)
        keep = (s_conf > tau) & (s_pred >= nb) & in_cand & bg

        pl = torch.full_like(s_pred, -1)
        pl[keep] = s_pred[keep]

        out = label.clone()
        out[bg] = pl[bg]

        n_bg = int(bg.sum())
        n_keep = int(keep.sum())
        if not hasattr(self, "_sc_stats"):
            self._sc_stats = dict(keep=0, bg=0, n=0)
        self._sc_stats["keep"] += n_keep
        self._sc_stats["bg"] += n_bg
        self._sc_stats["n"] += 1
        if self._sc_stats["n"] % 20 == 0:
            s = self._sc_stats
            self.logger.info(
                "[L1] 累计 keep=%d bg=%d (%.1f%%) tau=%.2f K=%d ep=%d"
                % (s["keep"], s["bg"], 100.0 * s["keep"] / max(s["bg"], 1),
                   tau, K, self.epoch))
        return out

    @torch.no_grad()
    def _target_m1_argmax(self, ret_dict, input_dict, use_cos=False):
        """M1：30 类全空间 argmax，只保留 novel 命中；可选按类绝对余弦阈值。"""
        feats = ret_dict.get("all_feats", None)
        if feats is None:
            return input_dict["segment"].clone()
        nb = int(self.cfg.data.num_bases)
        W = self.VLM_3D.task_head.cls_head.weight
        cos = torch.nn.functional.normalize(feats.float(), dim=-1) @ W.t()
        pred = cos.argmax(dim=-1)
        novel = pred >= nb
        if use_cos:
            if not hasattr(self, "_tau_c"):
                _p = getattr(self.cfg, "tau_path",
                             "/data01/jh/data/step0/tau_c_p50.npy")
                self._tau_c = torch.from_numpy(np.load(_p)).to(pred.device)
            maxc = cos.gather(1, pred.unsqueeze(1)).squeeze(1)
            _t = self._tau_c[pred]
            novel = novel & ((_t <= 0) | (maxc > _t))
        label = input_dict["segment"].clone()
        bg = label == -1
        pl = torch.full_like(pred, -1)
        sel = novel & bg
        pl[sel] = pred[sel]
        out = label.clone()
        out[bg] = pl[bg]
        if not hasattr(self, "_m1_stats"):
            self._m1_stats = dict(keep=0, bgc=0, n=0)
        self._m1_stats["keep"] += int(sel.sum())
        self._m1_stats["bgc"] += int(bg.sum())
        self._m1_stats["n"] += 1
        if self._m1_stats["n"] % 20 == 0:
            st = self._m1_stats
            self.logger.info(
                "[M1] keep=%d bg=%d (%.1f%%) cos=%s ep=%d" % (
                    st["keep"], st["bgc"],
                    100.0 * st["keep"] / max(st["bgc"], 1),
                    use_cos, self.epoch))
        return out

    @torch.no_grad()
    def _target_oracle_filter(self, ret_dict, input_dict):
        """完美过滤上界：只保留 PL 正确覆盖的原场景 novel 点。

        base GT 与 support GT 无条件保留；原场景 novel 点只有在 PL 预测与 GT
        一致时才保留，其余设 -1（模拟「所有错误 PL 都被过滤掉」）。
        """
        nb = int(self.cfg.data.num_bases)
        label = input_dict["segment"].clone()
        feats = ret_dict.get("all_feats", None)
        if feats is None:
            return label
        W = self.VLM_3D.task_head.cls_head.weight
        cos = torch.nn.functional.normalize(feats.float(), dim=-1) @ W.t()
        mode = getattr(self.cfg, "pl_mode", "oracle_m1")
        if mode == "oracle_m0":
            pred = cos[:, nb:].argmax(-1) + nb
        else:
            p30 = cos.argmax(-1)
            pred = torch.where(p30 >= nb, p30, torch.full_like(p30, -1))
        novel_gt = label >= nb
        supp = input_dict.get("mask", None)
        if supp is not None:
            supp = supp == 0
        else:
            supp = torch.zeros_like(label, dtype=torch.bool)
        wrong = (pred != label) & novel_gt & (~supp)
        label[wrong] = -1
        if not hasattr(self, "_or_stats"):
            self._or_stats = dict(keep=0, drop=0, n=0)
        self._or_stats["keep"] += int((novel_gt & (~wrong)).sum())
        self._or_stats["drop"] += int(wrong.sum())
        self._or_stats["n"] += 1
        if self._or_stats["n"] % 20 == 0:
            st = self._or_stats
            self.logger.info("[Oracle] %s keep=%d drop=%d" % (
                mode, st["keep"], st["drop"]))
        return label

    def construct_3D_vlm_input(self, input_dict):
        """
        Construct the input dictionary required for 3D VLM.
        """
        xyz = input_dict["coord"]
        rgb = input_dict["color"]  # Colors normalized to [-1, 1]

        # Adjust RGB if not normalized
        if not self.VLM_3D_cfg.DATA_CONFIG.DATA_PROCESSOR.rgb_norm:
            rgb = (rgb + 1) * 127.5

        data_dict = {
            "points_xyz": xyz,
            "rgb": rgb,
            "pc_count": xyz.shape[0],
        }
        # Scale coordinates for voxelization
        xyz_voxel_scale = (
            xyz * self.VLM_3D_cfg.DATA_CONFIG.DATA_PROCESSOR.voxel_scale
        )
        xyz_voxel_scale = xyz_voxel_scale - xyz_voxel_scale.min(0).values
        data_dict["points_xyz_voxel_scale"] = xyz_voxel_scale

        # Prepare features: use RGB and/or coordinates as needed
        if self.VLM_3D_cfg.DATA_CONFIG.DATA_PROCESSOR.rgb_as_feat:
            data_dict["feats"] = data_dict["rgb"]

        if self.VLM_3D_cfg.DATA_CONFIG.DATA_PROCESSOR.xyz_as_feat:
            if "feats" in data_dict:
                data_dict["feats"] = torch.cat(
                    (data_dict["feats"], data_dict["points_xyz"]), dim=1
                )
            else:
                data_dict["feats"] = data_dict["points_xyz"]

        # Process offsets to generate indexed voxel coordinates
        result = []
        start = 0
        for i, end in enumerate(input_dict["offset"]):
            slice_xyz = data_dict["points_xyz_voxel_scale"][start:end]
            # Prepend batch index to each point coordinate
            indexed_points = torch.cat(
                [
                    torch.full(
                        (slice_xyz.shape[0], 1),
                        i,
                        dtype=torch.int64,
                        device=slice_xyz.device,
                    ),
                    slice_xyz.to(torch.int64),
                ],
                dim=-1,
            )
            result.append(indexed_points)
            start = end

        data_dict["points_xyz_voxel_scale"] = torch.cat(result, dim=0)

        # Determine the spatial shape for the voxel grid
        data_dict["spatial_shape"] = torch.clamp(
            (data_dict["points_xyz_voxel_scale"].max(0).values[1:] + 1),
            min=self.VLM_3D_cfg.DATA_CONFIG.MIN_SPATIAL_SCALE,
        )
        data_dict["batch_idxs"] = data_dict["points_xyz_voxel_scale"][:, 0].to(
            torch.int32
        )

        batch_size = len(input_dict["offset"])
        if batch_size == 1:
            data_dict["offsets"] = torch.tensor(
                [0, data_dict["batch_idxs"].shape[0]], dtype=torch.int32
            )
        else:
            data_dict["offsets"] = torch.cumsum(
                torch.bincount(data_dict["batch_idxs"] + 1).to(torch.int32),
                dim=0,
            )
            assert len(data_dict["offsets"]) == batch_size + 1

        data_dict["batch_size"] = batch_size
        return data_dict


def _gn_probe(trainer):
    """XL-1 诊断：每 20 iter 记录一次 dec0/dec1/dec2/other 的梯度 L2 与 RMS。

    * 只在 iter % 20 == 0 时做实际统计，其余 iter 只自增计数器
    * AMP 下 p.grad 是 scaled 的，先除以 scaler.get_scale() 还原真实量级
    * 全程在 GPU 上累加标量，循环结束只调一次 .item()
    * 绝不影响训练：整体 try/except
    """
    if not bool(getattr(trainer.cfg, "log_grad_norm", False)):
        return
    try:
        n = int(getattr(trainer, "_gn_iter", 0)) + 1
        trainer._gn_iter = n
        if n % 20 != 0:
            return
        scale = 1.0
        if getattr(trainer, "scaler", None) is not None:
            sc = float(trainer.scaler.get_scale())
            if sc > 0:
                scale = 1.0 / sc
        acc = {g: [torch.zeros((), device="cuda"),
                   torch.zeros((), device="cuda")]
               for g in ("dec0", "dec1", "dec2", "other")}
        for nm, prm in trainer.model.named_parameters():
            if prm.grad is None:
                continue
            g = prm.grad.detach().float() * scale
            grp = "other"
            if "dec0." in nm:
                grp = "dec0"
            elif "dec1." in nm:
                grp = "dec1"
            elif "dec2." in nm:
                grp = "dec2"
            acc[grp][0] += (g * g).sum()
            acc[grp][1] += float(g.numel())
        parts = []
        for grp in ("dec0", "dec1", "dec2", "other"):
            s2, cnt = acc[grp]
            l2 = float(torch.sqrt(s2).item())
            rms = l2 / max(cnt ** 0.5, 1.0)
            parts.append("%s L2=%.3e RMS=%.3e" % (grp, l2, rms))
        trainer.logger.info(
            "[GradNorm] ep=%d it=%d invscale=%.2e | %s"
            % (trainer.epoch, n, scale, " | ".join(parts)))
    except Exception:
        pass
