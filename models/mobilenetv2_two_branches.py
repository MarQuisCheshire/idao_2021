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


class MobileNetV2TwoBranches(torch.nn.Module):
    def __init__(self,
                 img_channels=1,
                 first_channels=32,
                 emb_size=512,
                 normalization='BatchNorm',
                 activation='PReLU',
                 rev_alpha=0.01):
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
            # Stage 3
            BuildingBlock(first_channels * 4, first_channels * 6, attention=SEAttention, norm=normalization,
                          activation=activation, stride=2),
            BuildingBlock(first_channels * 6, first_channels * 6, attention=SEAttention, norm=normalization,
                          activation=activation),
            BuildingBlock(first_channels * 6, first_channels * 6, attention=SEAttention, norm=normalization,
                          activation=activation),

        )

        self.cls1 = torch.nn.Sequential(
            # Stage 4
            BuildingBlock(first_channels * 6, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation, stride=2),
            BuildingBlock(first_channels * 8, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation),
            BuildingBlock(first_channels * 8, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(first_channels * 8, max(first_channels * 8, 1024)),
            torch.nn.ReLU(True),
            torch.nn.Linear(max(first_channels * 8, 1024), emb_size),
            torch.nn.ReLU(True)
        )

        self.cls2 = torch.nn.Sequential(
            # Stage 4
            BuildingBlock(first_channels * 6, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation, stride=2),
            BuildingBlock(first_channels * 8, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation),
            BuildingBlock(first_channels * 8, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(first_channels * 8, max(first_channels * 8, 1024)),
            torch.nn.ReLU(True),
            torch.nn.Linear(max(first_channels * 8, 1024), emb_size),
            torch.nn.ReLU(True)
        )

        self.lin1 = torch.nn.Linear(emb_size, 2)
        self.lin2 = torch.nn.Linear(emb_size, 6)

        self.lin1_extra = torch.nn.Linear(emb_size, 6)
        self.lin2_extra = torch.nn.Linear(emb_size, 2)
        self.reversal1 = RevGrad(rev_alpha)
        self.reversal2 = RevGrad(rev_alpha)

    def forward(self, img):
        emb = self.extractor(img)
        emb1 = self.cls1(emb)
        emb2 = self.cls2(emb)

        if self.training:
            return (self.lin1(emb1),
                    self.lin2(emb2),
                    self.reversal1(self.lin1_extra(emb1)),
                    self.reversal2(self.lin2_extra(emb2)))
        return self.lin1(emb1), self.lin2(emb2)

    def get_params1(self):
        lst = []
        for name, p in self.named_parameters():
            if 'lin2.' not in name:
                lst.append(p)
        return lst

    def get_params2(self):
        lst = []
        for name, p in self.named_parameters():
            if 'lin1.' not in name:
                lst.append(p)
        return lst


if __name__ == '__main__':
    # Test size
    net = MobileNetV2TwoBranches(first_channels=20).cuda()
    img = torch.rand(24, 1, 576, 576, device='cuda:0')

    cls, energy = net(img)
    ((cls ** 2).mean() + (energy ** 2).mean()).backward()
    del net

    from thop import profile, clever_format

    net = MobileNetV2TwoBranches(first_channels=20)
    print(net)
    # print(net.params_full)
    x = torch.rand(1, 1, 576, 576)
    flops, params = profile(net, inputs=(x,))

    flops, params = clever_format([flops, params], "%.3f")
    print(f'flops:{flops}, params:{params}')
