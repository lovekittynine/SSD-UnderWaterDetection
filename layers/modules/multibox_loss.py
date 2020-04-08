# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets: 分配标签
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        """
        num_classes：数据集类别数(包含背景)
        overlap_thresh：default 0.5, 为每个anchor分配GT时候的IOU阈值
        prior_for_matching: True
        bkg_label: 背景类别0
        neg_mining：True, 是否挖掘困难负样本
        neg_pos：3， 负样本比例
        neg_overlap：0.5
        encode_target： False
        """
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        # 背景类别
        self.background_label = bkg_label
        # 是否将GT编码成[c_x, c_y, w, h]
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # loc_data 批量anchor位置输出 shape：Nxnum_anchorsx4 [delta_x, delta_y, delta_w, delta_h]
        # conf_data 批量anchor置信度得分输出 shape：Nxnum_anchorsxnum_classes 
        # priors 每张图片先验anchor shape: num_anchorsx4 [c_x, c_y, w, h] 
        loc_data, conf_data, priors = predictions
        # batch size大小
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        # 预测的anchor的数量
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        # 分配GT坐标(target)--回归时的偏移量
        loc_t = torch.Tensor(num, num_priors, 4)
        # 保存每个anchor的类别索引
        conf_t = torch.LongTensor(num, num_priors)
        """
        针对每张图像图片进行分配GT
        """
        for idx in range(num):
            # GT box 真实的坐标值[c_x, c_y, w, h]
            # num_objsx4
            # 针对每一张图片开始循环
            truths = targets[idx][:, :-1].data
            # GT box 真实的label
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        """
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        """

        # pos : bsxnum_priors
        # pos 表示正类的掩膜
        pos = conf_t > 0
        # num_pos bsx1 表示每个图片中正类anchor的数量
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # pos_idx 对于位置坐标输出的索引
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # box offset预测值
        loc_p = loc_data[pos_idx].view(-1, 4)
        # box offset真实值
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 计算定位损失
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        # batch_conf包含所有anchor的类别预测得分
        batch_conf = conf_data.view(-1, self.num_classes)
        
        # (bs*num_priors)x1列向量
        # loss_c存储所有anchor的分类损失值
        # 交叉熵loss的等价形式
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        # print('loss_c', loss_c.shape)

        # Hard Negative Mining
        # 将正样本损失置0
        loss_c[pos.view(-1, 1)] = 0
        # loss_c[pos] = 0  # filter out pos boxes for now

        # bsxnum_priors
        # reshape: 以针对每张图片都进行挖掘
        loss_c = loss_c.view(num, -1)
        
        _, loss_idx = loss_c.sort(1, descending=True)
        # 这里挖掘困难负样本不理解？？？？
        _, idx_rank = loss_idx.sort(1)
        # num_pos = pos.long().sum(1, keepdim=True)
        # 针对每张图片中正样本数量不同，有不同数量的负样本数
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        # 负样本索引 bsxnum_priors
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        # 包含所有正负样本的置信度得分
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
