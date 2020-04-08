from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    预先计算每个feature map上的anchor
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # 300
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        # SSD每个feature map上对应一个固定的anchor尺度
        self.num_priors = len(cfg['aspect_ratios'])
        # or表示集合操作(或)
        self.variance = cfg['variance'] or [0.1]
        # feature maps的大小[38, 19, 10, 5, 3, 1]
        self.feature_maps = cfg['feature_maps']
        # anchor对应的最小尺度(相对于原图300x300分辨率)
        self.min_sizes = cfg['min_sizes']
        # anchor对应的最大尺度
        self.max_sizes = cfg['max_sizes']
        # steps表示当前feature map相对于输入图像大小的下采样步数
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        """
        返回“一张图像上”所有feature map上的anchor(二维数组)
        细节: anchor的形式是:[center_x, center_y, w, h](且相对于原图归一化)
        """
        mean = []
        for k, f in enumerate(self.feature_maps):
            # itertools.product()用于计算2个list的笛卡尔积
            # repeat=2 表示range(f)与自身的笛卡尔积(常用来表示坐标位置)
            # i表示行坐标, j表示列坐标
            # f表示每一层Feature map的大小
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                # 计算feature map上每个元素对应anchor的中心点坐标(相对于原图进行归一化)
                cx = (j + 0.5) / f_k # W
                cy = (i + 0.5) / f_k # H

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                # 对于宽高比为1:1时，增加一个anchor: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)] # k:1
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)] # 1:k
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
