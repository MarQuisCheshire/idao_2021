import torch

from models.blocks import Swish, BuildingBlock, SEAttention

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
                 emb_size=512,
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
            # Stage 3
            BuildingBlock(first_channels * 4, first_channels * 6, attention=SEAttention, norm=normalization,
                          activation=activation, stride=2),
            BuildingBlock(first_channels * 6, first_channels * 6, attention=SEAttention, norm=normalization,
                          activation=activation),
            BuildingBlock(first_channels * 6, first_channels * 6, attention=SEAttention, norm=normalization,
                          activation=activation),
            # Stage 4
            BuildingBlock(first_channels * 6, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation, stride=2),
            BuildingBlock(first_channels * 8, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation),
            BuildingBlock(first_channels * 8, first_channels * 8, attention=SEAttention, norm=normalization,
                          activation=activation, r=4),

            torch.nn.AdaptiveAvgPool2d(1)
        )

        self.cls = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(first_channels * 8, max(first_channels * 8, 1024)),
            torch.nn.ReLU(True),
            torch.nn.Linear(max(first_channels * 8, 1024), emb_size),
            torch.nn.ReLU(True)
        )

        self.lin1 = torch.nn.Linear(emb_size, 2)
        self.lin2 = torch.nn.Linear(emb_size, 6)

    def forward(self, img):
        emb = self.extractor(img)
        emb = self.cls(emb)

        return self.lin1(emb), self.lin2(emb)


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
