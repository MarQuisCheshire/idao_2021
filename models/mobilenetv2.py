import torch


class BuildingBlock(torch.nn.Module):
    def __init__(self, inp, outp,
                 r=2, stride=1,
                 attention=None,
                 norm=torch.nn.BatchNorm2d,
                 activation=torch.nn.PReLU):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(inp, outp * r, 1, bias=False),
            norm(outp * r),
            activation(outp * r),
            torch.nn.Conv2d(outp * r, outp * r, 3, groups=outp * r, bias=False, padding=1, stride=stride),
            norm(outp * r),
            activation(outp * r),
            torch.nn.Conv2d(outp * r, outp, 1, bias=False),
            norm(outp),
            attention(outp) if attention is not None else torch.nn.Identity()
        )
        self.downsample = stride == 1 and inp == outp

    def forward(self, x):
        out = self.seq(x)
        if self.downsample:
            out += x
        return out


class SEAttention(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channel, channel // reduction, 1, bias=False),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(channel // reduction, channel, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.seq(x)
        return x * y


normalizations = {
    'BatchNorm': torch.nn.BatchNorm2d
}

activations = {
    'PReLU': torch.nn.PReLU
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
    net = MobileNetV2(first_channels=20).cuda()
    img = torch.rand(30, 1, 576, 576, device='cuda:0')

    cls, energy = net(img)
