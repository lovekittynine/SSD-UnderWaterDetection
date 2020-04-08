import torch
import torch.nn as nn
from torch.autograd import Function
from ..box_utils import decode, nms
from data import voc as cfg


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """
    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        """
        num_classes:类别个数
        bkg_label：背景类别编号(default 0)
        top_k：选取前top_k个得分最高的anchor
        conf_thresh：目标类别得分阈值
        nms_thresh：非极大值抑制阈值
        """
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        # 0.45
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']


    def __call__(self, loc_data, conf_data, prior_data):
        return self.forward(loc_data, conf_data, prior_data)
        
    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        # 对每一张测试图片achor的数量
        # 由于输入尺度固定，所以anchor数量一致
        num_priors = prior_data.size(0)
        # 输出每个类别最多都有的top_k个预测box
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        # conf_preds shape: Nxnum_classxnum_priors
        # 每个anchor的类别置信度
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            # num_priorx4 [x1,y1,x2,y2]
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            # num_class X num_priors
            conf_scores = conf_preds[i].clone()

            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                # ids保留的是有效预测box的索引， count是有效box的数量
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                # output 5维分别代表类别得分值和box回归值
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        # 一共num_class*top_k个anchors(每张图像)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output

