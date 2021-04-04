import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Any

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import PartDataset, ConcatDataset
from dataset import gen_train_data, gen_test_data, idx_to_energy, gen_dataset
from engine.running_loss import RunningLoss


class Controller(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        pl.seed_everything(cfg.seed)
        self.module: torch.nn.Module = cfg.module_factory()
        self.cfg = cfg
        self._seed = cfg.seed

        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.loss_energy = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(1)
        self.a = 1.
        self.b = 0.5

        self.running_loss = RunningLoss()
        self._optim = None

        self._train_ds = None
        self._test_ds = None

        for key, item in cfg.items():
            logging.info(f'{key}\t{item}')

    def forward(self, *args, **kwargs):
        return self.module(*args, *kwargs)

    def on_fit_start(self):
        if self.cfg.get('path_to_checkpoint'):
            self.load()

    def training_step(self, batch, batch_idx: int, optimizer_idx=0):
        pred_cls, pred_energy, rev_cls, rev_energy = self(batch['img'], optimizer_idx)
        labeled_flag = batch['cls'] != -1
        unlabeled_flag = batch['cls'] == -1
        batch['cls'][unlabeled_flag] = torch.argmax(self.softmax(pred_cls[unlabeled_flag]), dim=1)
        batch['energy'][unlabeled_flag] = torch.argmax(self.softmax(pred_cls[unlabeled_flag]), dim=1)

        if optimizer_idx == 0:
            pred_cls_labeled = pred_cls[labeled_flag]
            rev_cls_labeled = rev_cls[labeled_flag]
            pred_cls_unlabeled = pred_cls[unlabeled_flag]
            rev_cls_unlabeled = rev_cls[unlabeled_flag]
            loss1 = self.loss_cls(pred_cls_labeled, batch['cls'][labeled_flag]) + \
                    self.loss_energy(rev_cls_labeled, batch['energy'][labeled_flag])
            loss1 += self.b * (self.loss_cls(pred_cls_unlabeled, batch['cls'][unlabeled_flag]) +
                               self.loss_energy(rev_cls_unlabeled, batch['energy'][unlabeled_flag]))
            loss2 = 0.
        elif optimizer_idx == 1:
            pred_energy_labeled = pred_energy[labeled_flag]
            rev_energy_labeled = rev_energy[labeled_flag]
            pred_energy_unlabeled = pred_energy[unlabeled_flag]
            rev_energy_unlabeled = rev_energy[unlabeled_flag]
            loss2 = self.loss_energy(pred_energy_labeled, batch['energy'][labeled_flag]) + \
                    self.loss_cls(rev_energy_labeled, batch['cls'][labeled_flag])
            loss2 += (self.loss_energy(pred_energy_unlabeled, batch['energy'][unlabeled_flag]) +
                      self.loss_cls(rev_energy_unlabeled, batch['cls'][unlabeled_flag])) * self.b
            loss1 = 0.
        else:
            raise ValueError('Invalid optimizer index')

        loss = loss1 + self.a * loss2
        self.running_loss(loss.item())
        return loss

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

    def test_step(self, batch, batch_idx: int, dalaloader_idx=0):
        pred_cls, pred_energy = self(batch['img'])
        return self.softmax(pred_cls), self.softmax(pred_energy), batch['path']

    def test_epoch_end(self, outputs: List[Any]) -> None:
        counters = []
        for index in range(len(outputs)):
            counter = defaultdict(int)
            data = tuple(zip(*outputs[index]))
            paths = []
            for i in data[-1]:
                for j in i:
                    paths.append(str(Path(j).resolve().name))
            cls, energy = [torch.cat(i, dim=0).data.cpu().numpy() for i in data[:-1]]
            cls = np.argmax(cls, axis=1)
            energy = [idx_to_energy[i] for i in np.argmax(energy, axis=1)]
            for part in zip(paths, cls, energy):
                counter[part[1], part[2]] += 1
                print(*part, sep=',')
                if self.cfg.get('results_file'):
                    print(*part, sep=',', file=self.cfg.results_file)
            counters.append(counter)
        print(*counters, sep='\n')

    # Configuration
    def configure_optimizers(self):
        if self._optim is None:
            opt = [self.cfg.optim_factory(list(self.module.ext1.parameters()) +
                                          list(self.module.cls1.parameters()) +
                                          list(self.module.lin1_extra.parameters())),
                   self.cfg.optim_factory(list(self.module.ext2.parameters()) +
                                          list(self.module.cls2.parameters()) +
                                          list(self.module.lin2_extra.parameters()))]
            self._optim = opt
        else:
            opt = self._optim
        lr_sched = [self.cfg.lr_sched_factory(opt[i], self.current_epoch - 1) for i in range(len(opt))]
        return opt, lr_sched

    def train_dataloader(self) -> DataLoader:
        ds1 = self.train_ds
        train_indices = list(np.random.RandomState(self._seed + 1).choice(len(ds1),
                                                                          int(len(ds1) * 0.8),
                                                                          replace=False))
        ds1 = PartDataset(ds1, train_indices)
        ds2 = gen_dataset(self.cfg.test_path1, self.cfg.transform)
        ds3 = gen_dataset(self.cfg.test_path2, self.cfg.transform)
        ds = ConcatDataset(ds1, ds2, ds3)
        return torch.utils.data.DataLoader(ds, self.cfg.batch_size, num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        ds = self.train_ds
        train_indices = list(np.random.RandomState(self._seed + 1).choice(len(ds),
                                                                          int(len(ds) * 0.8),
                                                                          replace=False))

        val_indices = [i for i in range(len(ds)) if i not in train_indices]

        return [torch.utils.data.DataLoader(PartDataset(ds, val_indices), self.cfg.batch_size, num_workers=4),
                torch.utils.data.DataLoader(self.test_ds, self.cfg.batch_size)]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return [torch.utils.data.DataLoader(i, self.cfg.batch_size, num_workers=4) for i in
                [gen_dataset(self.cfg.test_path1, self.cfg.transform),
                 gen_dataset(self.cfg.test_path2, self.cfg.transform)]]

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
        self.module.load_state_dict(state['model'])
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
