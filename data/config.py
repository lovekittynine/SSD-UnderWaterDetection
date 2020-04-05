# config.py
import os.path

# gets home dir cross platform
HOME = os.path.abspath("./")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],  # feature map大小
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],  # feature map大小相对于原图输入的下采样比例
    'min_sizes': [30, 60, 111, 162, 213, 264],  # 每一层feature map上对应的anchor的大小
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],  # 每层feature map上anchor对应的宽高比
    'variance': [0.1, 0.2],  # variance???是干什么的额？bounding box regression 权重
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

# Under Water dectection config
# 包含背景类
underwater = {
    'num_classes': 5,
    'lr_steps': (10000, 30000, 50000),
    'max_iter': 50000,
    'feature_maps': [128, 64, 32, 16, 14, 12],  # feature map大小
    'min_dim': 512,
    'steps': [8, 16, 32, 64, 100, 300],  # feature map大小相对于原图输入的下采样比例
    'min_sizes': [15, 30, 68, 105, 143, 180],  # 每一层feature map上对应的anchor的大小
    'max_sizes': [30, 68, 105, 143, 180, 217],
    'aspect_ratios': [[1.414], [2, 1.414], [2, 1.414], [2, 1.414], [1.414], [1.414]],  # 每层feature map上anchor对应的宽高比
    'variance': [0.1, 0.2],  # variance???是干什么的额？bounding box regression 权重
    'clip': True,
    'name': 'UnderWater',
}
