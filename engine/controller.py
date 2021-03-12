import logging
import os
from typing import Union, List, Tuple, Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import ImgDataset, PartDataset
from engine.running_loss import RunningLoss


class Controller(pl.LightningModule):
    module: torch.nn.Module

    def __init__(self, cfg):
        super().__init__()
        pl.seed_everything(cfg.seed)
        self.module = cfg.module_factory()
        self.cfg = cfg
        self._seed = cfg.seed
        self._dataset = ImgDataset(cfg.path, cfg.transform)
        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.loss_energy = torch.nn.CrossEntropyLoss()
        self.a = 1.
        self.running_loss = RunningLoss()
        self._optim = None
        for key, item in cfg.items():
            logging.info(f'{key}\t{item}')

    def forward(self, *args, **kwargs):
        return self.module(*args, *kwargs)

    def on_train_start(self):
        if self.cfg.get('path_to_checkpoint'):
            self.load()

    def training_step(self, batch, batch_idx: int, optimizer_idx=0):
        pred_cls, pred_energy = self(batch['img'])
        loss1 = self.loss_cls(pred_cls, batch['cls'])
        loss2 = self.loss_energy(pred_energy, batch['energy'])
        loss = loss1 + self.a * loss2
        self.running_loss(loss.item())
        return loss

    def validation_step(self, batch, batch_idx: int, dalaloader_idx=0):
        pred_cls, pred_energy = self(batch['img'])
        return pred_cls, pred_energy, batch['cls'], batch['energy']

    def validation_epoch_end(self, outputs: List[Tuple[torch.Tensor, torch.Tensor, List, List]]) -> None:
        pred_cls, pred_energy, cls, energy = [torch.cat(i, dim=0) for i in zip(*outputs)]

        # TODO: add threshold
        acc = ((torch.argmax(pred_cls, dim=1) == cls) & (torch.argmax(pred_energy, dim=1) == energy)).float().mean()
        # mae_energy = torch.abs(torch.argmax(pred_energy, dim=1) - energy).mean()
        # TODO other metrics
        logging.info(f'\nEPOCH {self.current_epoch}\tAccuracy = {acc}')
        # logging.info(f'\nEPOCH {self.current_epoch}\tMAE Energy = {mae_energy}')

    def training_epoch_end(self, outputs: List[Any]) -> None:
        logging.info(f'\nEPOCH {self.current_epoch}\tLoss {self.running_loss.loss}' + '\n' * 3)
        self.save()

    # Configuration
    def configure_optimizers(self):
        if self._optim is None:
            opt = self.cfg.optim_factory(self.module.parameters())
            self._optim = [opt]
        else:
            opt = self._optim[0]
        lr_sched = self.cfg.lr_sched_factory(opt, self.current_epoch - 1)
        return [opt], [lr_sched]

    def train_dataloader(self) -> DataLoader:
        train_indices = list(np.random.RandomState(self._seed).choice(len(self._dataset),
                                                                      int(len(self._dataset) * 0.8),
                                                                      replace=False))
        ds = PartDataset(self._dataset, train_indices)
        return torch.utils.data.DataLoader(ds, self.cfg.batch_size, num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        train_indices = set(np.random.RandomState(self._seed).choice(len(self._dataset),
                                                                     int(len(self._dataset) * 0.8),
                                                                     replace=False))

        val_indices = [i for i in range(len(self._dataset)) if i not in train_indices]

        ds = PartDataset(self._dataset, val_indices)
        return torch.utils.data.DataLoader(ds, self.cfg.batch_size, num_workers=4)

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
