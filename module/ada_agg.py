import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange

class AdaAggLayer(nn.Module):
    r"""Applies an adaptive aggregate conv2d to the incoming data:.`
    """
    __constants__ = ['in_planes', 'out_planes', 'kernel_size', 'experts']

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, experts=5, align=True, lite=False):
        super(AdaAggLayer, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.experts = experts
        self.align = align
        self.lite = lite
        self.m = 0.1

        self.weight = nn.Parameter(torch.randn(experts, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(experts, out_planes))
        else:
            self.bias = None

        # channel-wise align
        if self.align and self.kernel_size > 1:
            align_conv = torch.zeros(self.experts * out_planes, out_planes, 1, 1)

            for i in range(self.experts):
                for j in range(self.out_planes):
                    align_conv[i * self.out_planes + j, j, 0, 0] = 1.0

            self.align_conv = nn.Parameter(align_conv, requires_grad=True)
        else:
            self.align = False

        # lite version
        if self.lite:
            self.register_buffer('lite_attention', torch.zeros(self.experts))

        # attention layer
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, in_planes // 4 + 1, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // 4 + 1, experts, 1, bias=True),
            nn.Flatten(),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.attention:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for i in range(self.experts):
            nn.init.kaiming_uniform_(self.weight[i])

    def forward(self, x):
        sigmoid_attention = self.attention(x)  # batch_size * experts

        batch_size = x.shape[0]

        # lite version
        if self.lite:
            if self.training:
                sigmoid_attention = sigmoid_attention.mean(0)
                self.lite_attention = (1 - self.m) * self.lite_attention + self.m * sigmoid_attention
            else:
                sigmoid_attention = self.lite_attention

            sigmoid_attention = sigmoid_attention.unsqueeze(0).repeat(batch_size, 1)

        # x = x.view(1, -1, height, width)   # 1 * BC * H * W
        x = rearrange(x, '(d b) c h w->d (b c) h w', d=1)

        # channel-wise align
        if self.align:
            weight = rearrange(self.weight, '(d e) o i j k->d (e o) i (j k)', d=1)
            # weight = self.weight.view(1, self.experts * self.out_planes, self.in_planes, self.kernel_size * self.kernel_size)
            weight = F.conv2d(weight, weight=self.align_conv, bias=None, stride=1, padding=0, dilation=1, groups=self.experts)
            weight = rearrange(weight, 'd (e o) i (j k)->(d e) o i j k', e=self.experts, j=self.kernel_size)
        else:
            weight = self.weight

        # weight = self.weight
        aggregate_weight = rearrange(
            torch.einsum('be,eoijk->boijk', sigmoid_attention, weight),
            'b o i j k->(b o) i j k'
        )

        if self.bias is not None:
            aggregate_bias = torch.einsum('be,eo->bo', sigmoid_attention, self.bias).view(-1)
            y = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)
        else:
            y = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        # y = y.view(batch_size, self.out_planes, y.size(-2), y.size(-1))
        y = rearrange(y, 'd (b o) h w->(d b) o h w', d=1, b=batch_size)
        
        return y
