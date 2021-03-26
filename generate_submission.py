from collections import defaultdict
from pathlib import Path
from typing import List, Any

import numpy as np
import torch
import torchvision

from dataset import gen_dataset, idx_to_energy, energy_indices
from models import MiniDoubleMobile


def test_step(model, batch):
    pred_cls, pred_energy = model(batch['img'])
    return torch.nn.Softmax(dim=1)(pred_cls), pred_energy, batch['path']


def test_epoch_end(outputs: List[Any], results_file=None) -> None:
    f = None
    if results_file:
        f = open(results_file, 'w')
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
        cls = (np.argmax(cls, axis=1) - 1) * (-1)
        energy = [idx_to_energy[np.argmin([abs(i - j) for j in energy_indices.keys()])] for i in energy]
        for part in zip(paths, cls, energy):
            counter[part[1], part[2]] += 1
            if f:
                print(*part, sep=',', file=f)
        counters.append(counter)
    if f:
        f.close()


if __name__ == '__main__':
    model = MiniDoubleMobile(first_channels=20, rev_alpha=1., emb_size=256, dropout_p=0.0)
    model.load_state_dict(torch.load('check_19.pth', map_location='cpu')['model'])
    model.eval()
    torch.set_grad_enabled(False)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(np.array),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(120)
    ])
    d1 = torch.utils.data.DataLoader(gen_dataset(str(Path('tests/public_test').resolve()), transform), 10)
    d2 = torch.utils.data.DataLoader(gen_dataset(str(Path('tests/private_test').resolve()), transform), 10)

    outp_lst = []
    for ds_index, ds in enumerate((d1, d2)):
        outp_lst.append([])
        for batch_index, batch in enumerate(ds):
            outp = test_step(model, batch)
            outp_lst[-1].append(outp)

    test_epoch_end(outp_lst, 'submission.csv')
