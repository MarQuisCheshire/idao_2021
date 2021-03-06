import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning
import torch
import torchvision
from pytorch_lightning import Trainer

from engine.controller6 import Controller
from models import MiniDoubleMobile
from models.final import MiniDoubleMobileVersion2
from models.mobilenetv2_twolittle import DoubleMobile


class DictWrapper:
    def __init__(self, d):
        self.d = d
        for i, j in d.items():
            self.__setattr__(i, j)

    def __getitem__(self, item):
        return self.d[item]

    def __len__(self):
        return len(self.d)

    def __getattr__(self, item):
        return getattr(self.d, item)

    def __iter__(self):
        return iter(self.d)


def init_logging(log_path):
    for h in logging.root.handlers[:]:
        logging.root.removeHandler(h)

    log_path = log_path if isinstance(log_path, Path) else Path(log_path)

    log_format = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)15s()] %(message)s"
    logging.basicConfig(level=logging.DEBUG,
                        format=log_format,
                        datefmt='%d.%m.%y %H:%M:%S',
                        filename=str(log_path / 'log.txt'))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


def main():
    cfg = DictWrapper({
        # 'optim_factory': lambda p: torch.optim.SGD(p, 0.000001, momentum=0.9),
        'optim_factory': lambda p: torch.optim.Adam(p, 0.000001),
        'lr_sched_factory': lambda opt, last_epoch: torch.optim.lr_scheduler.StepLR(opt, 50, 0.1, last_epoch),
        'path': 'D:\\IDAO\\data\\train',
        'path_to_checkpoint': None,
        'path_to_results': 'D:\\IDAO\\results\\regression\\120x120\\unlabeled_3\\3',
        'batch_size': 16,
        'transform': torchvision.transforms.Compose([
            torchvision.transforms.Lambda(np.array),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(120),
        ]),
        'seed': 123,
        # 'module_factory': lambda: MobileNetV2(first_channels=20)
        'module_factory': lambda: MiniDoubleMobileVersion2(first_channels=20, rev_alpha=1., emb_size=256),
        'results_file': None,
        'test_path1': 'D:\\IDAO\\data\\public_test',
        'test_path2': 'D:\\IDAO\\data\\private_test',
    })

    pytorch_lightning.seed_everything(cfg.seed)
    os.makedirs(cfg.path_to_results, exist_ok=True)
    init_logging(cfg.path_to_results)

    controller = Controller(cfg)
    trainer = Trainer(gpus=1,
                      logger=False,
                      checkpoint_callback=False,
                      num_sanity_val_steps=0,
                      max_epochs=60,
                      )
    trainer.fit(controller)


if __name__ == '__main__':
    main()
