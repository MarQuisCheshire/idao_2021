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

from dataset import PartDataset, energy_indices, ConcatDataset
from dataset import gen_train_data, gen_test_data, idx_to_energy, gen_dataset
from dataset.dataset import test_cases
from engine.running_loss import RunningLoss
from losses.unlabeledloss import UnlabeledLossCLS, UnlabeledLossEnergy


class Controller(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        pl.seed_everything(cfg.seed)
        self.module: torch.nn.Module = cfg.module_factory()
        self.cfg = cfg
        self._seed = cfg.seed

        self.loss_cls = torch.nn.CrossEntropyLoss()
        self.loss_energy = torch.nn.MSELoss()
        self.softmax = torch.nn.Softmax(1)

        self.unlabeled_cls_loss = UnlabeledLossCLS(0.1)
        self.unlabeled_energy_loss = UnlabeledLossEnergy(0.1)

        self.kldiv = torch.nn.KLDivLoss()
        self.norm_dist = torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.))

        self.a = 0.1
        self.b = 0.1

        self.running_loss = RunningLoss()
        self._optim = None

        self._train_ds = None
        self._test_ds = None

        for key, item in cfg.items():
            logging.info(f'{key}\t{item}')

    def forward(self, *args, **kwargs):
        return self.module(*args, *kwargs)

    def on_fit_start(self):
        # if self.cfg.get('path_to_checkpoint'):
        #     self.load()
        # self.module.load_state_dict(torch.load('result_state_1_2.pth'))
        self.module.load_state_dict(
            torch.load('D:\\IDAO\\results\\regression\\120x120\\unlabeled_3\\2\\check_27.pth')['model']
        )

    def training_step(self, batch, batch_idx: int, optimizer_idx=0):
        pred_cls, pred_energy, rev_cls, rev_energy = self(batch['img'], optimizer_idx)
        labeled = batch['cls'] != -1
        # need real energy for regression
        true_energy = torch.tensor([idx_to_energy[i] for i in batch['energy'][labeled].data.cpu().numpy()],
                                   device=self.device).float()
        if optimizer_idx == 0:
            loss = self.loss_cls(pred_cls[labeled], batch['cls'][labeled]) + self.a * \
                   self.loss_cls(rev_cls[labeled], batch['energy'][labeled])
            # loss += self.unlabeled_cls_loss(pred_cls[~labeled])
            # loss += 0.1 * self.kldiv(self.module.last1,
            #                          self.norm_dist.sample(self.module.last1.shape).to(self.module.last1.device))
        elif optimizer_idx == 1:
            loss = self.loss_energy(pred_energy.flatten()[labeled], true_energy) + self.b * \
                   self.loss_cls(rev_energy[labeled], batch['cls'][labeled])
            # loss += self.unlabeled_energy_loss(pred_energy[~labeled])
            # loss += 0.1 * self.kldiv(self.module.last2,
            #                          self.norm_dist.sample(self.module.last2.shape).to(self.module.last2.device))
        self.running_loss(loss.item())
        return loss

    def validation_step(self, batch, batch_idx: int, dalaloader_idx=0):
        pred_cls, pred_energy = self(batch['img'])
        return self.softmax(pred_cls), pred_energy, batch['cls'], batch['energy']

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        for index, tag in enumerate(['VAL', 'TEST']):
            metrics = [torch.cat(i, dim=0).data.cpu().numpy() for i in zip(*outputs[index])]
            acc, mae, roc_auc, quality_metric = self._calculate_metrics(*metrics)

            logging.info(f'\nEPOCH {self.current_epoch}\t{tag}\tAccuracy\t{acc}'
                         f'\nEPOCH {self.current_epoch}\t{tag}\tMAE Energy\t{mae}'
                         f'\nEPOCH {self.current_epoch}\t{tag}\tROC AUC\t{roc_auc}'
                         f'\nEPOCH {self.current_epoch}\t{tag}\tQuality Metric\t{quality_metric}')
        counters = []
        for index in [2, 3]:
            counter = defaultdict(int)
            data = tuple(zip(*outputs[index]))
            cls, energy = [torch.cat(i, dim=0).data.cpu().numpy() for i in data[:-2]]
            cls = np.argmax(cls, axis=1)
            # cls = (np.argmax(cls, axis=1) - 1) * (-1)
            energy = [idx_to_energy[np.argmin([abs(i - j) for j in energy_indices.keys()])] for i in energy]
            for part in zip(cls, energy):
                counter[part[0], part[1]] += 1
            counters.append(counter)
        print(*counters, sep='\n')
        print(sum([item for key, item in counters[0].items() if key in test_cases]))
        print(sum([item for key, item in counters[1].items() if key not in test_cases]))

    def training_epoch_end(self, outputs: List[Any]) -> None:
        logging.info(f'\nEPOCH {self.current_epoch}\tLoss {self.running_loss.loss}' + '\n' * 3)
        self.save()

    def test_step(self, batch, batch_idx: int, dalaloader_idx=0):
        pred_cls, pred_energy = self(batch['img'])
        return self.softmax(pred_cls), pred_energy, batch['path']

    def test_epoch_end(self, outputs: List[Any]) -> None:
        f = None
        if self.cfg.get('results_file'):
            f = open(self.cfg['results_file'], 'w')
            print('id,classification_predictions,regression_predictions', file=f)
        counters = []
        for index in range(len(outputs)):
            counter = defaultdict(int)
            data = tuple(zip(*outputs[index]))
            paths = []
            for i in data[-1]:
                for j in i:
                    paths.append(str(Path(j).resolve().name)[:-4])
            cls, energy = [torch.cat(i, dim=0).data.cpu().numpy() for i in data[:-1]]
            # cls = np.argmax(cls, axis=1)
            cls = (np.argmax(cls, axis=1) - 1) * (-1)
            energy = [idx_to_energy[np.argmin([abs(i - j) for j in energy_indices.keys()])] for i in energy]
            for part in zip(paths, cls, energy):
                counter[part[1], part[2]] += 1
                print(*part, sep=',')
                if f:
                    print(*part, sep=',', file=f)
            counters.append(counter)
        if f:
            f.close()
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
        ds1 = torch.utils.data.Subset(ds1, np.random.choice(len(ds1), 48, replace=False))
        ds = ConcatDataset(ds1, self.test_ds)
        print('LEN', len(ds1), len(ds))
        return torch.utils.data.DataLoader(ds, self.cfg.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        ds = self.train_ds
        train_indices = list(np.random.RandomState(self._seed + 1).choice(len(ds),
                                                                          int(len(ds) * 0.8),
                                                                          replace=False))

        val_indices = [i for i in range(len(ds)) if i not in train_indices]

        return [torch.utils.data.DataLoader(PartDataset(ds, val_indices), 1000, num_workers=2),
                torch.utils.data.DataLoader(self.test_ds, 1000),
                torch.utils.data.DataLoader(gen_dataset(self.cfg.test_path2, self.cfg.transform), 1000, num_workers=2),
                torch.utils.data.DataLoader(gen_dataset(self.cfg.test_path1, self.cfg.transform), 1000, num_workers=2)]

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
        pred_energy = np.array(
            [idx_to_energy[np.argmin([abs(i - j) for j in energy_indices.keys()])] for i in pred_energy])
        energy = np.array([idx_to_energy[i] for i in energy])
        acc = ((np.argmax(pred_cls, axis=1) == cls) & (pred_energy == energy)).mean()
        # energy = [idx_to_energy[np.argmin([abs(i - j) for j in energy_indices.keys()])] for i in energy]
        mae = np.abs(pred_energy - energy).mean()
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
