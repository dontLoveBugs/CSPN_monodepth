# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:32:49 2018
@author: Xinjing Cheng
@email : chengxinjing@baidu.com
"""

"""
    Modified by WangXin
    Updated on 16:58:37 19/05/19
    Replace upsample with conv_transpose2d to implement up_pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from network.post_process import CSPN_ours as post_process

# memory analyze

__all__ = ['ResNet_prediction', 'ResNet_completion', 'resnet18_prediction', 'resnet50_prediction']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_path = {
    'resnet18': 'pretrained/resnet18.pth',
    'resnet50': 'pretrained/resnet50.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class MyBlock(nn.Module):
    def __init__(self, oheight=0, owidth=0):
        super(MyBlock, self).__init__()

        self.oheight = oheight
        self.owidth = owidth

    def _up_pooling(self, x, scale):
        N, C, H, W = x.size()

        num_channels = C
        weights = torch.zeros(num_channels, 1, scale, scale)
        if torch.cuda.is_available():
            weights = weights.cuda()
        weights[:, :, 0, 0] = 1
        y = F.conv_transpose2d(x, weights, stride=scale, groups=num_channels)
        del weights

        if self.oheight != scale * H or self.owidth != scale * W:
            y = y[:, :, 0:self.oheight, 0:self.owidth]

        return y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)


class UpProj_Block(MyBlock):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(UpProj_Block, self).__init__(oheight, owidth)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Simple_Gudi_UpConv_Block(MyBlock):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block, self).__init__(oheight, owidth)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        return out


class Simple_Gudi_UpConv_Block_Last_Layer(MyBlock):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Simple_Gudi_UpConv_Block_Last_Layer, self).__init__(oheight, owidth)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.conv1(x)
        return out


class Gudi_UpProj_Block(MyBlock):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block, self).__init__(oheight, owidth)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class Gudi_UpProj_Block_Cat(MyBlock):
    def __init__(self, in_channels, out_channels, oheight=0, owidth=0):
        super(Gudi_UpProj_Block_Cat, self).__init__(oheight, owidth)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1_1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sc_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.sc_bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, side_input):
        x = self._up_pooling(x, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        out = torch.cat((out, side_input), 1)
        out = self.relu(self.bn1_1(self.conv1_1(out)))
        out = self.bn2(self.conv2(out))
        short_cut = self.sc_bn1(self.sc_conv1(x))
        out += short_cut
        out = self.relu(out)
        return out


class ResNet_completion(nn.Module):

    def __init__(self, block, layers, up_proj_block):
        self.inplanes = 64
        super(ResNet_completion, self).__init__()
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.mid_channel = 256 * block.expansion
        self.conv2 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.post_process_layer = self._make_post_process_layer()

        self.gud_up_proj_layer1 = self._make_gud_up_conv_layer(Gudi_UpProj_Block, 2048, 1024, 15, 19)
        self.gud_up_proj_layer2 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 1024, 512, 29, 38)
        self.gud_up_proj_layer3 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 512, 256, 57, 76)
        self.gud_up_proj_layer4 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 256, 64, 114, 152)
        self.gud_up_proj_layer5 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 1, 228, 304)
        self.gud_up_proj_layer6 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 8, 228, 304)

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

    def _make_up_conv_layer(self, up_proj_block, in_channels, out_channels):
        return up_proj_block(in_channels, out_channels)

    def _make_gud_up_conv_layer(self, up_proj_block, in_channels, out_channels, oheight, owidth):
        return up_proj_block(in_channels, out_channels, oheight, owidth)

    def _make_post_process_layer(self):
        return post_process.AffinityPropagate_completion()

    def forward(self, x):
        [batch_size, channel, height, width] = x.size()
        sparse_depth = x.narrow(1, 3, 1).clone()  # get sparse depth

        x = self.conv1_1(x)
        skip4 = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        skip3 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(self.conv2(x))
        x = self.gud_up_proj_layer1(x)

        x = self.gud_up_proj_layer2(x, skip2)
        x = self.gud_up_proj_layer3(x, skip3)
        x = self.gud_up_proj_layer4(x, skip4)

        blur_depth = self.gud_up_proj_layer5(x)
        guidance = self.gud_up_proj_layer6(x)
        x = self.post_process_layer(guidance, blur_depth, sparse_depth)

        return x


class ResNet_prediction(nn.Module):

    def __init__(self, block, layers, up_proj_block):
        self.inplanes = 64
        super(ResNet_prediction, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.mid_channel = 256 * block.expansion
        self.conv2 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.conv3 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.post_process_layer = self._make_post_process_layer()

        self.gud_up_proj_layer1 = self._make_gud_up_conv_layer(Gudi_UpProj_Block, 2048, 1024, 15, 19)
        self.gud_up_proj_layer2 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 1024, 512, 29, 38)
        self.gud_up_proj_layer3 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 512, 256, 57, 76)
        self.gud_up_proj_layer4 = self._make_gud_up_conv_layer(Gudi_UpProj_Block_Cat, 256, 64, 114, 152)
        self.gud_up_proj_layer5 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 1, 228, 304)
        self.gud_up_proj_layer6 = self._make_gud_up_conv_layer(Simple_Gudi_UpConv_Block_Last_Layer, 64, 9, 228, 304)

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

    def _make_up_conv_layer(self, up_proj_block, in_channels, out_channels):
        return up_proj_block(in_channels, out_channels)

    def _make_gud_up_conv_layer(self, up_proj_block, in_channels, out_channels, oheight, owidth):
        return up_proj_block(in_channels, out_channels, oheight, owidth)

    def _make_post_process_layer(self):
        return post_process.AffinityPropagate_prediction()

    def forward(self, x):
        x = self.conv1_1(x)
        skip4 = x

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        skip3 = x

        x = self.layer2(x)
        skip2 = x

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(self.conv2(x))
        x = self.gud_up_proj_layer1(x)

        x = self.gud_up_proj_layer2(x, skip2)
        x = self.gud_up_proj_layer3(x, skip3)
        x = self.gud_up_proj_layer4(x, skip4)

        blur_depth = self.gud_up_proj_layer5(x)
        guidance = self.gud_up_proj_layer6(x)
        x = self.post_process_layer(guidance, blur_depth)

        return [x, guidance]


def resnet18_prediction(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_prediction(BasicBlock, [2, 2, 2, 2], UpProj_Block, **kwargs)
    if pretrained:
        print('==> Load pretrained model..')
        pretrained_dict = torch.load(model_path['resnet18'])
        import network.utils as utils
        model.load_state_dict(utils.load_model_dict(model, pretrained_dict))
    return model


def resnet50_prediction(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_prediction(Bottleneck, [3, 4, 6, 3], UpProj_Block, **kwargs)
    if pretrained:
        print('==> Load pretrained model..')
        pretrained_dict = torch.load(model_path['resnet50'])
        import network.utils as utils
        model.load_state_dict(utils.load_model_dict(model, pretrained_dict))
    return model


def resnet50_completion(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_completion(Bottleneck, [3, 4, 6, 3], UpProj_Block, **kwargs)
    if pretrained:
        print('==> Load pretrained model..')
        pretrained_dict = torch.load(model_path['resnet50'])
        import network.utils as utils
        model.load_state_dict(utils.load_model_dict(model, pretrained_dict))
    return model


if __name__ == '__main__':
    img = torch.randn(4, 3, 228, 304).cuda()
    model = resnet50_prediction(pretrained=False).cuda()

    with torch.no_grad():
        pred = model(img)

    print(pred[0].shape)
    print(pred[0])

    print('affinity:', pred[1])
