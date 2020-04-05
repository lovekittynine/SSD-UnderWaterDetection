import xml.etree.ElementTree as ET
import os
import shutil
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tqdm import tqdm

def data_analysis(image_size=1024):
    """
    分析数据集中box宽高比，高度和宽度的分布
    """
    # xml_path
    xml_path = '../dataset/train/valid_box'
    xml_files = os.listdir(xml_path)
    # image folder
    img_folder = '../dataset/train/image'
    # 记录rations
    ratios = []
    # 记录缩放后的宽度和高度
    scales = []  
    # parse xml file
    for f in tqdm(xml_files, desc='Preprocessing....'):
        target = ET.parse(os.path.join(xml_path, f)).getroot()
        img_id = f[:-4]
        img_path = os.path.join(img_folder, img_id+'.jpg')
        # read image
        img = cv2.imread(img_path)
        H, W, _ = img.shape
        # 图像缩放到1024x1024
        factor_H = image_size/H
        factor_W = image_size/W
        
        for obj in target.iter('object'):
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                if i%2==0:
                    cur_pt *= factor_W
                else:
                    cur_pt *= factor_H
                bndbox.append(cur_pt)
            # 计算缩放的ratio, H, W
            new_H, new_W = bndbox[3]-bndbox[1], bndbox[2]-bndbox[0]
            try:
                ratio = new_W/new_H
                ratios.append(ratio)
                scales.append([new_H, new_W])
            except Exception:
                pass
    
    ratios = np.array(ratios) 
    scales = np.array(scales)  
    print('max_ratio:', np.max(ratios), 'min_ratio:', np.min(ratios), 'mean_ratio:', np.mean(ratios))
    print('max_height:', np.max(scales[:, 0]), 'min_height:', np.min(scales[:, 0]), 'mean_height:', np.mean(scales[:, 0]))
    print('max_width:', np.max(scales[:, 1]), 'min_width:', np.min(scales[:, 1]), 'mean_width:', np.mean(scales[:, 1]))
    # 绘图分布
    fig = plt.figure()
    
    plt.subplot(1,3,1)
    plt.hist(ratios, bins=100)
    plt.title('宽高比分布')
    plt.xlabel('ratio')
    
    plt.subplot(1,3,2)
    plt.hist(np.round(scales[:, 0]), bins=100)
    plt.title('Height分布')
    plt.xlabel('Height')
    
    plt.subplot(1,3,3)
    plt.hist(np.round(scales[:, 1]), bins=100)
    plt.title('Width分布')
    plt.xlabel('Width')
    plt.savefig('./统计.jpg', dpi=300)
    plt.show()

if __name__ == '__main__':
    data_analysis()
