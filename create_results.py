import numpy as np
import torch
import torchvision
from pytorch_lightning import Trainer

from engine import Controller
from main import DictWrapper
from models.mobilenetv2_twolittle import DoubleMobile

if __name__ == '__main__':
    cfg = DictWrapper({
        'optim_factory': lambda p, lr=0.01: torch.optim.SGD(p, lr, 0.9, weight_decay=0.00001),
        'lr_sched_factory': lambda opt, last_epoch: torch.optim.lr_scheduler.StepLR(opt, 10, 0.1, last_epoch),
        'path': 'D:\\IDAO\\data\\train',
        'path_to_checkpoint': r'D:\IDAO\results\6\check_12.pth',
        'path_to_results': 'D:\\IDAO\\results\\7',
        'batch_size': 100,
        'transform': torchvision.transforms.Compose([
            torchvision.transforms.Lambda(np.array),
            torchvision.transforms.ToTensor()
        ]),
        'seed': 123,
        # 'module_factory': lambda: MobileNetV2(first_channels=20)
        'module_factory': lambda: DoubleMobile(first_channels=20, rev_alpha=1., emb_size=128),
        'results_file': None,
        'test_path1': 'D:\\IDAO\\data\\public_test',
        'test_path2': 'D:\\IDAO\\data\\private_test'
    })

    controller = Controller(cfg)
    trainer = Trainer(gpus=1,
                      logger=False,
                      checkpoint_callback=False,
                      num_sanity_val_steps=0,
                      max_epochs=50,
                      )
    trainer.test(controller)
