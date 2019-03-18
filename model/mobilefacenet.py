import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import torch


def init_block(in_channels, out_channels, stride):
    """Builds the first block of the MobileFaceNet"""
    
    return nn.Sequential(
        
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

class InvertedResidual(nn.Module):
    """Implementation of the modified Inverted residual block"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio,outp_size=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.inv_block = nn.Sequential(
            #pointwise
            nn.Conv2d(in_channels, in_channels * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU(inplace=True),

            #depthwise conv3x3
            nn.Conv2d(in_channels * expand_ratio, in_channels * expand_ratio, 3, stride, 1,
                      groups=in_channels * expand_ratio, bias=False),
            nn.BatchNorm2d(in_channels * expand_ratio),
            nn.ReLU(inplace=True),
            
            #pointwise linear
            nn.Conv2d(in_channels * expand_ratio, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.inv_block(x)

        return self.inv_block(x)

class AngleSimpleLinear(nn.Module):
    """Computes cos of angles between input vectors and weights vectors"""
    def __init__(self, in_features, out_features):
        super(AngleSimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        cos_theta = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return cos_theta.clamp(-1, 1)


class MobileFaceNet(nn.Module):
    """Implements modified MobileFaceNet"""
    def __init__(self, num_classes=33, width_multiplier=1., feature=True):
        super(MobileFaceNet, self).__init__()
        
        assert num_classes > 0
        assert width_multiplier > 0
        self.feature = feature

        # Set up of inverted residual blocks
        self.inverted_residual_setting = [
            #bottleneck
            # t, c, n, s
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 1],
            [4, 128, 1, 2],
            [2, 128, 2, 1]
        ]

        first_channel_num = 64
        last_channel_num = 512

        ##first layer--conv3x3
        self.features = [init_block(1, first_channel_num, 2)]
        ##second layer--depthwise conv3x3
        self.features.append(nn.Conv2d(first_channel_num, first_channel_num, 3, 1, 1,
                                       groups=first_channel_num, bias=False))
        self.features.append(nn.BatchNorm2d(64))
        self.features.append(nn.ReLU(inplace=True))
        ##Bottleneck
        ##Inverted Residual Blocks
        in_channel_num = first_channel_num
        #size_h, size_w = MobileFaceNet.get_input_res()
        #size_h, size_w = size_h // 2, size_w // 2
        for t, c, n, s in self.inverted_residual_setting:
            output_channel = int(c * width_multiplier)
            for i in range(n):
                if i == 0:
                    #size_h, size_w = size_h // s, size_w // s
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          s, t))
                else:
                    self.features.append(InvertedResidual(in_channel_num, output_channel,
                                                          1, t))
                in_channel_num = output_channel

        # 1x1 expand block
        self.features.append(nn.Sequential(nn.Conv2d(in_channel_num, last_channel_num, 1, 1, 0, bias=False),
                                           nn.BatchNorm2d(last_channel_num),
                                           nn.ReLU(inplace=True)))
        self.features = nn.Sequential(*self.features)

        # Depth-wise pooling
        k_size = (MobileFaceNet.get_input_res()[0] // 16, MobileFaceNet.get_input_res()[1] // 16)
        self.dw_pool = nn.Conv2d(last_channel_num, last_channel_num, k_size,
                                 groups=last_channel_num, bias=False)
        self.dw_bn = nn.BatchNorm2d(last_channel_num)
        #the last layer
        self.conv1_extra = nn.Conv2d(last_channel_num,128, 1, stride=1, padding=0, bias=False)
        
        if self.feature:
            self.fc_angular = AngleSimpleLinear(128, 33)
        # self.classifier = nn.Linear(128,33)
        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.dw_bn(self.dw_pool(x))
        x = self.conv1_extra(x)

        # if self.feature or not self.training:
        #     return x

        x = x.view(x.size(0), -1)
        x = self.fc_angular(x)

        return x

    @staticmethod
    def get_input_res():
        return 64,64

    def set_dropout_ratio(self, ratio):
        assert 0 <= ratio < 1.

    def init_weights(self):
        """Initializes weights of the model before training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
