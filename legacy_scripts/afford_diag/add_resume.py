# -*- coding: utf-8 -*-
p = '/data01/jh/code/GFS-VL/pointcept/engines/train.py'
s = open(p, encoding='utf-8').read()

old = '''            self.train_loader = self.build_train_loader()

            # Execute the training loop for the current registration dataset
            with EventStorage() as self.storage, ExceptionWriter():'''

new = '''            self.train_loader = self.build_train_loader()

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
                        self.model.load_state_dict(_ck["state_dict"], strict=False)
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

            # Execute the training loop for the current registration dataset
            with EventStorage() as self.storage, ExceptionWriter():'''

assert s.count(old) == 1, 'anchor count = %d' % s.count(old)
s = s.replace(old, new)
open(p, 'w', encoding='utf-8').write(s)
print('resume inserted')
