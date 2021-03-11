import logging
from typing import Union, List, Tuple, Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import ImgDataset, PartDataset
from engine.running_loss import RunningLoss


class Controller(pl.LightningModule):
    module: torch.nn.Module

    def __init__(self, module, cfg, seed=123):
        super().__init__()
        pl.seed_everything(seed)
        self.module = module()
        self.cfg = cfg
        self._seed = seed
        self._dataset = ImgDataset(cfg.path, cfg.transform)
        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.loss_energy = torch.nn.CrossEntropyLoss()
        self.a = 1.
        self.running_loss = RunningLoss()

    def forward(self, *args, **kwargs):
        return self.module(*args, *kwargs)

    def training_step(self, batch, batch_idx: int, optimizer_idx=0):
        pred_cls, pred_energy = self(batch['img'])
        loss1 = self.loss_cls(pred_cls, batch['cls'])
        loss2 = self.loss_energy(pred_energy, batch['energy'])
        loss = loss1 + self.a * loss2
        self.running_loss(loss.item())
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        logging.info(f'EPOCH {self.current_epoch}\tLoss {self.running_loss.loss}')

    def validation_step(self, batch, batch_idx: int, dalaloader_idx=0):
        pred_cls, pred_energy = self(batch['img'])
        return pred_cls, pred_energy, batch['cls'], batch['energy']

    def validation_epoch_end(self, outputs: List[Tuple[torch.Tensor, torch.Tensor, List, List]]) -> None:
        pred_cls, pred_energy, cls, energy = zip(*outputs)

        # TODO: add threshold
        acc = ((torch.argmax(pred_cls, dim=1) == cls) & (torch.argmax(pred_energy, dim=1) == energy)).float().mean()
        mae_energy = torch.abs(torch.argmax(pred_energy, dim=1) - energy).mean()
        # TODO other metrics
        logging.info(f'EPOCH {self.current_epoch}\tAccuracy = {acc}')
        logging.info(f'EPOCH {self.current_epoch}\tMAE Energy = {mae_energy}')

    def configure_optimizers(
            self,
    ):
        opt = torch.optim.SGD(self.module.parameters(), 0.1, 0.9, weight_decay=0.00001)
        lr_sched = torch.optim.lr_scheduler.StepLR(opt, 100, 0.1)
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
