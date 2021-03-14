import logging
import os
from typing import Union, List, Any, Callable, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dataset import PartDataset
from dataset.dataset import gen_train_data, gen_test_data, idx_to_energy
from engine.running_loss import RunningLoss


class Controller(pl.LightningModule):
    module: torch.nn.Module

    def __init__(self, cfg):
        super().__init__()
        pl.seed_everything(cfg.seed)
        self.module = cfg.module_factory()
        self.cfg = cfg
        self._seed = cfg.seed

        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.loss_energy = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(1)
        self.a = 1.

        self.running_loss = RunningLoss()
        self._optim = None

        self._train_ds = None
        self._test_ds = None

        for key, item in cfg.items():
            logging.info(f'{key}\t{item}')

    def forward(self, *args, **kwargs):
        return self.module(*args, *kwargs)

    def on_train_start(self):
        if self.cfg.get('path_to_checkpoint'):
            self.load()

    def training_step(self, batch, batch_idx: int, optimizer_idx=0):
        pred_cls, pred_energy, rev_cls, rev_energy = self(batch['img'])
        loss1 = self.loss_cls(pred_cls, batch['cls']) + self.loss_energy(rev_cls, batch['energy'])
        loss2 = self.loss_energy(pred_energy, batch['energy']) + self.loss_cls(rev_energy, batch['cls'])
        loss = loss1 + self.a * loss2
        self.running_loss(loss.item())
        return loss

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        optimizer.step(closure=optimizer_closure)

    def optimizer_zero_grad(
        self, epoch: int, batch_idx: int, optimizer: Optimizer, optimizer_idx: int
    ):
        optimizer.zero_grad()

    def validation_step(self, batch, batch_idx: int, dalaloader_idx=0):
        pred_cls, pred_energy = self(batch['img'])
        return self.softmax(pred_cls), self.softmax(pred_energy), batch['cls'], batch['energy']

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        for index, tag in enumerate(['VAL', 'TEST']):
            metrics = [torch.cat(i, dim=0).data.cpu().numpy() for i in zip(*outputs[index])]
            acc, mae, roc_auc, quality_metric = self._calculate_metrics(*metrics)

            logging.info(f'\nEPOCH {self.current_epoch}\t{tag}\tAccuracy\t{acc}'
                         f'\nEPOCH {self.current_epoch}\t{tag}\tMAE Energy\t{mae}'
                         f'\nEPOCH {self.current_epoch}\t{tag}\tROC AUC\t{roc_auc}'
                         f'\nEPOCH {self.current_epoch}\t{tag}\tQuality Metric\t{quality_metric}')

    def training_epoch_end(self, outputs: List[Any]) -> None:
        logging.info(f'\nEPOCH {self.current_epoch}\tLoss {self.running_loss.loss}' + '\n' * 3)
        self.save()

    # Configuration
    def configure_optimizers(self):
        if self._optim is None:
            opt = [self.cfg.optim_factory(self.module.parameters(), 0.01)]
            self._optim = opt
        else:
            opt = self._optim[0]
        lr_sched = [self.cfg.lr_sched_factory(opt[i], self.current_epoch - 1) for i in range(len(opt))]
        return opt, lr_sched

    def train_dataloader(self) -> DataLoader:
        ds = self.train_ds
        train_indices = list(np.random.RandomState(self._seed + 1).choice(len(ds),
                                                                          int(len(ds) * 0.8),
                                                                          replace=False))
        ds = PartDataset(ds, train_indices)
        return torch.utils.data.DataLoader(ds, self.cfg.batch_size, num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        ds = self.train_ds
        train_indices = list(np.random.RandomState(self._seed + 1).choice(len(ds),
                                                                          int(len(ds) * 0.8),
                                                                          replace=False))

        val_indices = [i for i in range(len(ds)) if i not in train_indices]

        ds = [PartDataset(ds, val_indices), self.test_ds]
        return [torch.utils.data.DataLoader(ds[i], self.cfg.batch_size, num_workers=4) for i in range(len(ds))]

    def save(self):
        os.makedirs(self.cfg.path_to_results, exist_ok=True)
        state = {
            'model': self.module.state_dict(),
            'optimizer': [i.state_dict() for i in self.trainer.optimizers],
            'epoch': self.current_epoch
        }
        torch.save(state, os.path.join(self.cfg.path_to_results, f'check_{self.current_epoch}.pth'))

    def load(self):
        state = torch.load(self.cfg.path_to_checkpoint, map_location='cpu')
        if self._optim is None:
            self.configure_optimizers()
        self.module.load_state_dict(state)
        for i in range(len(state['optimizer'])):
            self._optim[i].load_state_dict(state['optimizer'][i])
        if self.trainer:
            self.trainer.current_epoch = state['epoch']

    @staticmethod
    def _calculate_metrics(pred_cls, pred_energy, cls, energy):
        acc = ((np.argmax(pred_cls, axis=1) == cls) & (np.argmax(pred_energy, axis=1) == energy)).mean()
        pred_energy = [idx_to_energy[i] for i in np.argmax(pred_energy, axis=1)]
        energy = [idx_to_energy[i] for i in energy]
        mae = np.abs(np.array(pred_energy) - np.array(energy)).mean()
        roc_auc = roc_auc_score(cls, np.argmax(pred_cls, axis=1))
        quality_metric = (roc_auc - mae) * 1000
        return acc, mae, roc_auc, quality_metric

    @property
    def train_ds(self):
        if self._train_ds is None:
            self._train_ds = gen_train_data(self.cfg.path, self.cfg.transform)
        return self._train_ds

    @property
    def test_ds(self):
        if self._test_ds is None:
            self._test_ds = gen_test_data(self.cfg.path, self.cfg.transform)
        return self._test_ds
