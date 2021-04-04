import torch


class UnlabeledLossCLS(torch.nn.Module):

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.softmax = torch.nn.Softmax(dim=1)
        self.inner_loss = torch.nn.CrossEntropyLoss()

    def forward(self, cls):
        labels = torch.argmax(self.softmax(cls), dim=1)
        return self.alpha * self.inner_loss(cls, labels)


# class UnlabeledLossEnergy(torch.nn.Module):
#
#     def __init__(self, alpha=1.):
#         super().__init__()
#         self.alpha = alpha
#         self.inner_loss = torch.nn.KLDivLoss()
#         tmp = torch.tensor([1., 3., 6., 10., 20., 30.])
#         self.dist = torch.distributions.Normal(torch.mean(tmp), torch.std(tmp))
#
#     def forward(self, energy):
#         return self.alpha * self.inner_loss(energy, self.dist.sample(energy.shape).to(energy.device))

class UnlabeledLossEnergy(torch.nn.Module):

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.var = torch.tensor([1., 3., 6., 10., 20., 30.]).var()

    def forward(self, energy):
        return self.alpha * energy.var()
