import torch

from models.blocks import Swish, BuildingBlock, SEAttention, RevGrad

normalizations = {
    'BatchNorm': torch.nn.BatchNorm2d,
    'InstanceNorm': torch.nn.InstanceNorm2d,
}

activations = {
    'PReLU': torch.nn.PReLU,
    'ReLU': lambda _: torch.nn.ReLU(True),
    'Swish': lambda _: Swish(),
}


class MobileNetV2(torch.nn.Module):
    def __init__(self,
                 img_channels=1,
                 first_channels=32,
                 normalization='BatchNorm',
                 activation='PReLU'):
        super().__init__()
        normalization = normalizations[normalization]
        activation = activations[activation]

        self.extractor = torch.nn.Sequential(
            torch.nn.Conv2d(img_channels, first_channels, 9, bias=False, padding=4, stride=3),
            normalization(first_channels),
            activation(first_channels),
            BuildingBlock(first_channels, first_channels, attention=SEAttention, norm=normalization,
                          activation=activation),
            # Stage 1
            BuildingBlock(first_channels, first_channels * 2, attention=SEAttention, norm=normalization,
                          activation=activation, stride=2),
            BuildingBlock(first_channels * 2, first_channels * 2, attention=SEAttention, norm=normalization,
                          activation=activation),
            BuildingBlock(first_channels * 2, first_channels * 2, attention=SEAttention, norm=normalization,
                          activation=activation),
            # Stage 2
            BuildingBlock(first_channels * 2, first_channels * 4, attention=SEAttention, norm=normalization,
                          activation=activation, stride=2),
            BuildingBlock(first_channels * 4, first_channels * 4, attention=SEAttention, norm=normalization,
                          activation=activation),
            BuildingBlock(first_channels * 4, first_channels * 4, attention=SEAttention, norm=normalization,
                          activation=activation),
            # Stage 4
            BuildingBlock(first_channels * 4, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation, stride=2),
            BuildingBlock(first_channels * 8, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation, stride=2),
            BuildingBlock(first_channels * 8, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation, r=4),

            torch.nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, img):
        emb = self.extractor(img)
        return emb


class DoubleMobile(torch.nn.Module):

    def __init__(self, emb_size=512, first_channels=32, rev_alpha=0.01, dropout_p=0., *args, **kwargs):
        super().__init__()
        self.ext1 = MobileNetV2(first_channels=first_channels, *args, **kwargs)
        self.ext2 = MobileNetV2(first_channels=first_channels, *args, **kwargs)

        self.cls1 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(dropout_p, True),
            torch.nn.Linear(first_channels * 8, max(first_channels * 8, 1024)),
            torch.nn.ReLU(True),
            torch.nn.Linear(max(first_channels * 8, 1024), emb_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(emb_size, 2)
        )
        self.cls2 = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(dropout_p, True),
            torch.nn.Linear(first_channels * 8, max(first_channels * 8, 1024)),
            torch.nn.ReLU(True),
            torch.nn.Linear(max(first_channels * 8, 1024), emb_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(emb_size, 1)
        )

        self.lin1_extra = torch.nn.Sequential(
            torch.nn.Flatten(),
            RevGrad(rev_alpha),
            torch.nn.Dropout(dropout_p, True),
            torch.nn.Linear(first_channels * 8, max(first_channels * 8, 1024)),
            torch.nn.ReLU(True),
            torch.nn.Linear(max(first_channels * 8, 1024), emb_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(emb_size, 1)
        )
        self.lin2_extra = torch.nn.Sequential(
            torch.nn.Flatten(),
            RevGrad(rev_alpha),
            torch.nn.Dropout(dropout_p, True),
            torch.nn.Linear(first_channels * 8, max(first_channels * 8, 1024)),
            torch.nn.ReLU(True),
            torch.nn.Linear(max(first_channels * 8, 1024), emb_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(emb_size, 2)
        )

    def forward(self, x, net_idx=None):
        if net_idx is None:
            cls_, adapt_cls = self._inner_call(x, self.ext1, self.cls1, self.lin1_extra)
            energy_, adapt_energy = self._inner_call(x, self.ext2, self.cls2, self.lin2_extra)
            if not self.training:
                return cls_, energy_
            return cls_, energy_, adapt_cls, adapt_energy
        elif net_idx == 0:
            cls_, adapt_cls = self._inner_call(x, self.ext1, self.cls1, self.lin1_extra)
            return cls_, None, adapt_cls, None
        elif net_idx == 1:
            energy_, adapt_energy = self._inner_call(x, self.ext2, self.cls2, self.lin2_extra)
            return None, energy_, None, adapt_energy

    @staticmethod
    def _inner_call(x, ext, cls, extra_cls):
        x = ext(x)
        return cls(x), extra_cls(x)


if __name__ == '__main__':
    # Test size
    net = MobileNetV2(first_channels=20).cuda()
    img = torch.rand(24, 1, 576, 576, device='cuda:0')

    cls, energy = net(img)
    ((cls ** 2).mean() + (energy ** 2).mean()).backward()
    del net

    from thop import profile, clever_format

    net = MobileNetV2(first_channels=20)
    print(net)
    # print(net.params_full)
    x = torch.rand(1, 1, 576, 576)
    flops, params = profile(net, inputs=(x,))

    flops, params = clever_format([flops, params], "%.3f")
    print(f'flops:{flops}, params:{params}')
