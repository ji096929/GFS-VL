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

            # Update the training dataset configuration
            self.cfg.data.train.k_shot = self.cfg.k_shot
            self.cfg.data.train.seed = val_data_cfg["seed"]
            self.cfg.data.train.nb_mix_blks = self.cfg.nb_mix_blks
            self.train_loader = self.build_train_loader()

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

            def _weighted_ce(logits):
                # 主 loss 与辅助 loss 共用同一口径，避免软权重模式下静默不一致
                if _pw is None:
                    return criteria(logits, new_target)
                ce = torch.nn.functional.cross_entropy(
                    logits, new_target, reduction="none", ignore_index=-1)
                valid = new_target != -1
                return (ce * _pw)[valid].sum() / (_pw[valid].sum() + 1e-6)

            loss = _weighted_ce(seg_logits)
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
            self.scaler.step(self.optimizer)
            # Adjust the scheduler if loss scaling factor remains unchanged
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

        label_one_hot = label.clone()
        label_one_hot[bg_label_mask] = prediction[bg_label_mask]

        return label_one_hot

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
