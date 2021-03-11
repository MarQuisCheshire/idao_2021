import os
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImgDataset(Dataset):
    path: Path
    paths: list
    cls: list
    energy: list

    def __init__(self, path: Union[str, Path], transform=None):
        self.path = path if isinstance(path, Path) else Path(path)
        self.transform = transform
        self.search()

    def search(self):
        self.paths = []
        self.cls = []
        self.energy = []
        cls_indices = dict(ER=0, He_NR=1)

        for path, dirs, files in os.walk(str(self.path.resolve())):
            for file in filter(lambda x: x.endswith('.png'), files):
                splitted = file.split('CYGNO_60_40_')[1].split('_keV')[0].split('_')
                if len(splitted) == 2:
                    cls, energy = splitted
                else:
                    assert len(splitted) == 3, "Invalid data"
                    cls = f'{splitted[0]}_{splitted[1]}'
                    energy = splitted[2]
                self.energy.append(int(energy))
                self.cls.append(cls_indices[cls])
                self.paths.append(os.path.join(path, file))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        img = Image.open(path)

        if self.transform:
            img = self.transform(img)
        return dict(img=img, cls=self.cls[item], energy=self.energy[item])


class PartDataset(Dataset):
    def __init__(self, dataset, left, right):
        self.dataset = dataset
        self.left = left
        self.right = right

    def __len__(self):
        return self.right - self.left

    def __getitem__(self, item):
        assert item < len(self)
        return self.dataset[self.left + item]


if __name__ == '__main__':
    ds = ImgDataset('D:\\IDAO\\data\\train', np.array)
    dl = torch.utils.data.DataLoader(ds, 100, shuffle=True, drop_last=True, num_workers=4)
    from tqdm import tqdm

    c = 0
    for i, batch in tqdm(enumerate(dl)):
        c += len(batch['cls'])
    print(len(ds), c)
