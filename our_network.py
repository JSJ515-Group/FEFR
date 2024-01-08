from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
from torch.nn import init
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_classes=1000):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6
        #block = Bottleneck if depth >=44 else BasicBlock
        block = BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32
        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class FFL_ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        super(FFL_ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        #block = Bottleneck if depth >=44 else BasicBlock

        block = BasicBlock
       # block = Bottleneck if depth >= 44 else BasicBlock
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)

        fix_inplanes = self.inplanes
        self.layer3_1 = self._make_layer(block, 64, n, stride=2)
        self.inplanes = fix_inplanes  ##reuse self.inplanes
        self.layer3_2 = self._make_layer(block, 64, n, stride=2)
        self.conv2=nn.Conv2d(16,32,kernel_size=2,stride=2)
        self.conv3=nn.Conv2d(32,64,kernel_size=2,stride=2)
        self.conv4=nn.Conv2d(32,32,kernel_size=2,stride=2)
        self.fc_adjust=nn.Linear(96,64)





        self.avgpool = nn.AvgPool2d(8)



        self.classfier3_1 = nn.Linear(64 * block.expansion, num_classes)
        self.classfier3_2 = nn.Linear(64 * block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    #
    # def _upsample_add(self, x, y):
    #
    #     _,_,H,W = y.shape
    #     return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):

        fmap = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x1 = self.layer1(x)  # 32x32
        x2= self.layer2(x1)  # 16x16

        x1_adjust = self.conv2(x1)
        x_add1 = x2 + x1_adjust
        x_half1 = x_add1 / 2

        x_3_1 = self.layer3_1(x2)  # 8x8
        x2_adjust = self.conv3(x2)
        x_add2 = x2_adjust + x_3_1
        x_half2 = x_add2 / 2
        x_half1_adjust = self.conv4(x_half1)
        fus1 = torch.cat((x_half1_adjust, x_half2), dim=1)
        fus1a = fus1[:, :24, :, :]
        fus1b = fus1[:, 24:48, :, :]
        fus1b_1 = fus1b[:, :12, :, :]
        fus1b_2 = fus1b[:, 12:24, :, :]
        fus1ab_2 = torch.cat((fus1a, fus1b_2), dim=1)
        fus1c = fus1[:, 48:72, :, :]
        fus1bc = torch.cat((fus1b_1, fus1c), dim=1)
        fus1bc_1 = fus1bc[:, 0:18, :, :]
        fus1bc_2 = fus1bc[:, 18:36, :, :]
        fus1ab_2c_2 = torch.cat((fus1ab_2, fus1bc_2), dim=1)
        fus1d = fus1[:, 72:96, :, :]
        fus1cd = torch.cat((fus1bc_1, fus1d), dim=1)
        fusi = torch.cat((fus1ab_2c_2, fus1cd), dim=1)

        x_3_2 = self.layer3_2(x2)
        x2_adjust1 = self.conv3(x2)
        x_add3 = x2_adjust1 + x_3_2
        x_half3 = x_add3 / 2
        fus2 = torch.cat((x_half1_adjust, x_half3), dim=1)
        fus2a = fus2[:, :24, :, :]
        fus2b = fus2[:, 24:48, :, :]
        fus2b_1 = fus2b[:, :12, :, :]
        fus2b_2 = fus2b[:, 12:24, :, :]
        fus2ab_2 = torch.cat((fus2a, fus2b_2), dim=1)
        fus2c = fus2[:, 48:72, :, :]
        fus2bc = torch.cat((fus2b_1, fus2c), dim=1)
        fus2bc_1 = fus2bc[:, 0:18, :, :]
        fus2bc_2 = fus2bc[:, 18:36, :, :]
        fus2ab_2c_2 = torch.cat((fus2ab_2, fus2bc_2), dim=1)
        fus2d = fus2[:, 72:96, :, :]
        fus2cd = torch.cat((fus2bc_1, fus2d), dim=1)
        fusj = torch.cat((fus2ab_2c_2, fus2cd), dim=1)
        model = cbam_block(96)
        model.to('cuda')
        fusi = model(fusi)
        fusj = model(fusj)

        fmap.append(fusi)
        fmap.append(fusj)




        x_3_1 = self.avgpool(x_3_1)
        x_3_1 = x_3_1.view(x_3_1.size(0), -1)

        #x_3_1=torch.unsqueeze(x_3_1,dim=0)
        #x_3_1=self.fc_adjust(x_3_1)
        x_3_2 = self.avgpool(x_3_2)
        x_3_2 = x_3_2.view(x_3_2.size(0), -1)
        #x_3_2 = torch.unsqueeze(x_3_2, dim=0)
        #x_3_2 = self.fc_adjust(x_3_2)



        x_3_1 = self.classfier3_1(x_3_1)


        x_3_2 = self.classfier3_2(x_3_2)



        return x_3_1,x_3_2,fmap

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x
class Fusion_module(nn.Module):
    def __init__(self,channel,numclass,sptial):
        super(Fusion_module, self).__init__()
        self.fc2   = nn.Linear(channel, numclass)
        self.conv1 =  nn.Conv2d(channel*2, channel*2, kernel_size=3, stride=1, padding=1, groups=channel*2, bias=False)
        self.bn1 = nn.BatchNorm2d(channel * 2)
        self.conv1_1 = nn.Conv2d(channel*2, channel, kernel_size=1, groups=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channel)
        self.conv_adjust=nn.Conv2d(192,128,kernel_size=1)


        self.sptial = sptial


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #self.avg = channel
    def forward(self, x,y):
        bias = False
        atmap = []
        input = torch.cat((x,y),1)
        input=self.conv_adjust(input)

        x = F.relu(self.bn1((self.conv1(input))))
        x = F.relu(self.bn1_1(self.conv1_1(x)))

        atmap.append(x)
        x = F.avg_pool2d(x, self.sptial)
        x = x.view(x.size(0), -1)

        out = self.fc2(x)
        atmap.append(out)

        return out




def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

def ffl_resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return FFL_ResNet(**kwargs)


if __name__ == '__main__':
    x = torch.rand((1,3,32,32))
    model = FFL_ResNet(depth=32,num_classes=100)
    y = model(x)
    # print(y.shape)
















