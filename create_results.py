import numpy as np
import torch
import torchvision
from pytorch_lightning import Trainer

from engine.controller3 import Controller
from main import DictWrapper
from models import MiniDoubleMobile

if __name__ == '__main__':
    cfg = DictWrapper({
        'optim_factory': lambda p, lr=0.01: torch.optim.SGD(p, lr, momentum=0.9),
        # 'optim_factory': lambda p, lr=0.01: torch.optim.Adam(p, lr),
        'lr_sched_factory': lambda opt, last_epoch: torch.optim.lr_scheduler.StepLR(opt, 10, 0.1, last_epoch),
        'path': 'D:\\IDAO\\data\\train',
        # 'path_to_checkpoint': r'D:\IDAO\results\6\check_13.pth',
        # 'path_to_checkpoint': r'D:\IDAO\results\13\check_7.pth',
        # 'path_to_checkpoint': r'D:\IDAO\results\regression\1\check_22.pth',
        # 'path_to_checkpoint': r'D:\IDAO\results\regression\120x120\1\check_19.pth', # release 1.0
        'path_to_checkpoint': 'result_state.pth', # release 2.0
        'path_to_results': 'D:\\IDAO\\results\\7',
        'batch_size': 100,
        'transform': torchvision.transforms.Compose([
            torchvision.transforms.Lambda(np.array),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.CenterCrop(120)
        ]),
        'seed': 123,
        # 'module_factory': lambda: MobileNetV2(first_channels=20)
        # 'module_factory': lambda: DoubleMobile(first_channels=20, rev_alpha=1., emb_size=128), # 6
        # 'module_factory': lambda: DoubleMobile(first_channels=20, rev_alpha=0.8, emb_size=128, dropout_p=0.4), # 13
        'module_factory': lambda: MiniDoubleMobile(first_channels=20, rev_alpha=1., emb_size=256, dropout_p=0.0), # release 1.0

        'results_file': 'submission_A.csv',
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
