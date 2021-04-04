from typing import List, Any, Union

import numpy as np
import torch
import torchvision
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from dataset import idx_to_energy, energy_indices, PartDataset, gen_test_data, gen_train_data
from models.final import MiniDoubleMobileVersion2


def val_dataloader(path, transform) -> Union[DataLoader, List[DataLoader]]:
    ds = gen_train_data(path, transform)
    train_indices = list(np.random.RandomState(123 + 1).choice(len(ds),
                                                               int(len(ds) * 0.8),
                                                               replace=False))

    val_indices = [i for i in range(len(ds)) if i not in train_indices]
    return [torch.utils.data.DataLoader(PartDataset(ds, val_indices), 100, num_workers=4),
            torch.utils.data.DataLoader(gen_test_data(path, transform), 100)]


def calculate_metrics(pred_cls, pred_energy, cls, energy):
    pred_energy = np.array([idx_to_energy[np.argmin([abs(i - j) for j in energy_indices.keys()])] for i in pred_energy])
    energy = np.array([idx_to_energy[i] for i in energy])
    acc = ((np.argmax(pred_cls, axis=1) == cls) & (pred_energy == energy)).mean()
    # energy = [idx_to_energy[np.argmin([abs(i - j) for j in energy_indices.keys()])] for i in energy]
    mae = np.abs(pred_energy - energy).mean()
    roc_auc = roc_auc_score(cls, np.argmax(pred_cls, axis=1))
    quality_metric = (roc_auc - mae) * 1000
    return acc, mae, roc_auc, quality_metric


def val_step(model, batch):
    pred_cls, pred_energy = model(batch['img'].to(device))
    pred_cls = pred_cls.cpu()
    pred_energy = pred_energy.cpu()
    return torch.nn.Softmax(dim=1)(pred_cls), pred_energy, batch['cls'], batch['energy']


def val_epoch_end(outputs: List[Any]) -> None:
    for index, tag in enumerate(['VAL', 'TEST']):
        metrics = [torch.cat(i, dim=0).data.cpu().numpy() for i in zip(*outputs[index])]
        acc, mae, roc_auc, quality_metric = calculate_metrics(*metrics)

        print(f'{tag}\tAccuracy\t{acc}'
              f'\n{tag}\tMAE Energy\t{mae}'
              f'\n{tag}\tROC AUC\t{roc_auc}'
              f'\n{tag}\tQuality Metric\t{quality_metric}')


if __name__ == '__main__':
    device = 'cuda:0'
    model = MiniDoubleMobileVersion2(first_channels=20, rev_alpha=1., emb_size=256, dropout_p=0.0).to(device)
    # 1.0 (domain adaptation 0.6 0.8)
    # model.load_state_dict(torch.load('check_19.pth', map_location='cpu')['model'])
    # 1.1 (previous classifier and better adapted regression (1.5))
    # cls = torch.load('check_19.pth', map_location='cpu')['model']
    # energy = torch.load('check_7.pth', map_location='cpu')['model']
    # state = OrderedDict()
    # for key, item in cls.items():
    #     if any([key.startswith(i) for i in ['ext1.', 'cls1.', 'lin1_extra.']]):
    #         state[key] = item
    #
    # for key, item in energy.items():
    #     if any([key.startswith(i) for i in ['ext2.', 'cls2.', 'lin2_extra.']]):
    #         state[key] = item
    #
    # model.load_state_dict(state)
    # torch.save(model.state_dict(), 'result_state.pth')

    # model.load_state_dict(torch.load('result_state.pth'))

    # 1.2 (used pseudo flag and variance reduction)
    # cls = torch.load('result_state.pth', map_location='cpu')
    # state = OrderedDict()
    # for key, item in cls.items():
    #     if any([key.startswith(i) for i in ['ext1.', 'cls1.', 'lin1_extra.']]):
    #         state[key] = item
    # energy = torch.load(r'D:\IDAO\results\regression\120x120\unlabeled\5\check_12.pth')['model']
    # for key, item in energy.items():
    #     if any([key.startswith(i) for i in ['ext2.', 'cls2.', 'lin2_extra.']]):
    #         state[key] = item
    # model.load_state_dict(state)
    # torch.save(model.state_dict(), 'result_state_1_2.pth')

    # 1.3 (declined) (used pseudo flag and kldiv after extractor)
    # cls = torch.load('result_state.pth', map_location='cpu')
    # state = OrderedDict()
    # for key, item in cls.items():
    #     if any([key.startswith(i) for i in ['ext1.', 'cls1.', 'lin1_extra.']]):
    #         state[key] = item
    # energy = torch.load( r'D:\IDAO\results\regression\120x120\unlabeled\8\check_1.pth')['model']
    # for key, item in energy.items():
    #     if any([key.startswith(i) for i in ['ext2.', 'cls2.', 'lin2_extra.']]):
    #         state[key] = item
    # model.load_state_dict(state)
    # torch.save(model.state_dict(), 'result_state_1_3.pth')
    # model.load_state_dict(torch.load('result_state_1_3.pth'))

    # 1.4 (finetuned 1.2 model)
    # 'D:\\IDAO\\results\\regression\\120x120\\unlabeled_3\\2\\check_27.pth'
    model.load_state_dict(torch.load('result_state_1_4.pth'))

    model.eval()
    torch.set_grad_enabled(False)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(np.array),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(120)
    ])
    d1, d2 = val_dataloader('D:\\IDAO\\data\\train', transform)

    outp_lst = []
    for ds_index, ds in enumerate((d1, d2)):
        outp_lst.append([])
        for batch_index, batch in enumerate(ds):
            outp = val_step(model, batch)
            outp_lst[-1].append(outp)

    val_epoch_end(outp_lst)
