import os
from pathlib import Path
from typing import Union

import numpy as np
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
        self._search()

    def _search(self):
        self.paths = []
        self.cls = []
        self.energy = []
        cls_indices = dict(ER=0, He_NR=1)
        energy_indices = {1: 0, 3: 1, 6: 2, 10: 3, 20: 4, 30: 5}

        for path, dirs, files in os.walk(str(self.path.resolve())):
            for file in filter(lambda x: x.endswith('.png'), files):
                splitted = file.split('CYGNO_60_40_')[1].split('_keV')[0].split('_')
                if len(splitted) == 2:
                    cls, energy = splitted
                else:
                    assert len(splitted) == 3, "Invalid data"
                    cls = f'{splitted[0]}_{splitted[1]}'
                    energy = splitted[2]
                self.energy.append(energy_indices[int(energy)])
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
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


if __name__ == '__main__':
    ds = ImgDataset('D:\\IDAO\\data\\train', np.array)
    print(ds[0]['img'].shape)
    # dl = torch.utils.data.DataLoader(ds, 100, shuffle=True, drop_last=True, num_workers=4)
    # from tqdm import tqdm
    #
    # c = 0
    # for i, batch in tqdm(enumerate(dl)):
    #     c += len(batch['cls'])
    # print(len(ds), c)
