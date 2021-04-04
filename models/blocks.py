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


class Swish(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x) * x


class RevGradFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


class RevGrad(torch.nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return RevGradFunc.apply(input_, self._alpha)
