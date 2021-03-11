from typing import Union, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataset import ImgDataset, PartDataset


class Controller(pl.LightningModule):
    module: torch.nn.Module

    def __init__(self, module, cfg, seed=123):
        super().__init__()
        self.module = module
        self.cfg = cfg
        self.seed = seed
        self._dataset = ImgDataset(cfg.path, cfg.transform)
        pl.seed_everything(seed)
        pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def training_step(self, batch, batch_idx: int, optimizer_idx=0):
        out = self(**batch)
        return out

    def validation_step(self, batch, batch_idx: int, dalaloader_idx=0):
        tensor = self(**batch)
        return tensor

    # def validation_epoch_end(self, outputs: List[Any]) -> None:

    def train_dataloader(self) -> DataLoader:
        ds = PartDataset(self._dataset, 0, int(len(self._dataset) * 0.8))
        return torch.utils.data.DataLoader(ds, self.cfg.batch_size, num_workers=4, shuffle=True, drop_last=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        ds = PartDataset(self._dataset, int(len(self._dataset) * 0.8), len(self._dataset))
        return torch.utils.data.DataLoader(ds, self.cfg.batch_size, num_workers=4)
