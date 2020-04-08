import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import underwater, coco
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        # 根据类别选择voc或者coco数据集配置
        self.cfg = underwater
        # 计算所有feature map上的anchor
        self.priorbox = PriorBox(self.cfg)
        # self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.priors = self.priorbox.forward() # Nx4
        # 输入图像大小
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)  # 20表示通道维特征范数
        self.extras = nn.ModuleList(extras)

        # anchor回归器
        self.loc = nn.ModuleList(head[0])
        # anchor分类器
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        # 将conv4_3的特征图进行L2范数归一化
        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        # VGG16最后一个卷积conv_7特征(ReLU之后)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        # Note：在前向传播过程中，卷积层使用了ReLU激活函数
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # sources保存需要检测目标的feature_maps

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # l(x):Nx(4xnum_anchors)xHxW
            # c(x):Nxnum_classxHxW
            loc.append(l(x).permute(0, 2, 3, 1).contiguous()) # NxHxWx(4*num_anchors)
            conf.append(c(x).permute(0, 2, 3, 1).contiguous()) # NxHxWxnum_class

        # 返回一个二维数组(NxN_hat)
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    """
    cfg:网络模型配置
    i:输入通道数
    """
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            # ceil_mode在计算输出时向上取整，默认是向下取整
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    # 注意pool5 stride=1(下采样16倍输出)
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # 使用空洞卷积增大感受野(保证输出尺寸不变)
    # 细节:当使用kernel_size=3的空洞卷积需要保证输出分辨率不变的时候，padding=dilation
    # f = ceil(14+2*6-(3+2*5)) + 1 = 14
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    # 相比于原fc7缩减16倍参数量
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    """
    cfg:模型配置
    i:输入通道数
    """
    # 细节：在源码中额外添加的特征层并没有添加激活函数(线性缩放得到不同尺度的feature map)
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                # 3x3卷积，stride=2，输出分辨率减小2倍
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                # 1x1卷积通道降维
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    """
    vgg: VGG16模型
    extra_layers: 额外添加的卷积层
    cfg: 每个feature map上每个空间位置对应的anchor的数量
    在分类器后接上检测器层
    # 定位器和分类器都是3x3卷积
    """
    loc_layers = []
    conf_layers = []
    # 21表示从conv4_3卷积后的feature map进行检测
    # -2表示从conv7卷积特征之后进行检测
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        # 定位层回归anchor坐标
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        # 分类层
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]

    # trick: enumerate(iterator, idx) idx表示索引序号从idx开始编号
    for k, v in enumerate(extra_layers[1::2], 2):
        # k的值从2开始递增
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


# VGG16基础网络结构配置
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}

# 在VGG16 backbone基础上额外添加的结构配置
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    """
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    """
    # base_: VGG16网络
    # extras_: 额外的特征提取层
    # head_: 检测器头
    base_, extras_, head_ = multibox(vgg(base[str(300)], 3),
                                     add_extras(extras[str(300)], 1024),
                                     mbox[str(300)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
