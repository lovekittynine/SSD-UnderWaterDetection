from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import UnderWater_ROOT, UnderWater_CLASSES as labelmap
from PIL import Image
from data import UnderWaterAnnotationTransform, UnderWaterDetection, BaseTransform, UnderWater_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import numpy as np
import pandas as pd
import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd300_60000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--data_root', default=UnderWater_ROOT, help='Location of dataset root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def test_net(save_folder, net, cuda, testset, transform, thresh, visualizable=True):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'pred1.csv'
    if visualizable:
        visualize_folder = os.path.join(save_folder, 'visualize')
        if not os.path.exists(visualize_folder):
            os.makedirs(visualize_folder)

    # 预测结果
    predicts = []
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        # img is opecv Style
        img_id, img = testset.pull_image(i)
        # print(img.shape)
        img_cv = img[...]
        # img_id, annotation = testset.pull_anno(i)
        # convert to CXHXW
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = x.unsqueeze(0)

        if cuda:
            x = x.cuda()
        with torch.no_grad():
            y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        # i表示数据集类别循环
        for i in range(detections.size(1)):
            # j表示每个类别下所有的anchor
            j = 0
            # 取出类别预测置信度
            while detections[0, i, j, 0] >= 0.6:
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                # 添加预测结果
                results = [label_name, img_id+'.xml', score.item(), pt[0], pt[1], pt[2], pt[3]]
                predicts.append(results)
                pred_num += 1
                if visualizable:
                    img_cv = visualize(img_cv, coords, i)
                j += 1

        if visualizable:
            cv2.imwrite(os.path.join(visualize_folder, img_id+'.jpg'), img_cv)
    # save predicts
    df = pd.DataFrame(predicts, index=None, columns=['name','image_id','confidence','xmin','ymin','xmax','ymax'])
    df.to_csv(filename, index=None)
    print('Predict Finished!!!')

def test():
    # load net
    num_classes = len(UnderWater_CLASSES) + 1 # +1 background
    net = build_ssd('test', 512, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = UnderWaterDetection(args.data_root, image_sets='test-A')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)


def visualize(img, coords, idx):
    """
    可视化预测结果
    img：预测输入图
    coords: 预测每个box坐标[xmin, ymin, xmax, ymax]
    idx: 对于输入图像img，box的类别索引
    """
    cv2.rectangle(img,
                  (int(coords[0]), int(coords[1])),
                  (int(coords[2]), int(coords[3])),
                  COLORS[idx % 3], 2)
    cv2.putText(img, labelmap[idx - 1], (int(coords[0]), int(coords[1])),
                FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
    return img

if __name__ == '__main__':
    test()
