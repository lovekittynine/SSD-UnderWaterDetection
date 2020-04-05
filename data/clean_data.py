import xml.etree.ElementTree as ET
import os
import shutil

def clean_dataset():
    """
    将没有有效的box的训练图片丢弃
    """
    # xml_path
    xml_path = '../dataset/train/box'
    xml_files = os.listdir(xml_path)
    # 有效xml path
    valid_xml_path = '../dataset/train/valid_box'
    if not os.path.exists(valid_xml_path):
        os.makedirs(valid_xml_path)
    
    # parse xml file
    for f in xml_files:
        target = ET.parse(os.path.join(xml_path, f)).getroot()
        valid_boxes = 0
        for obj in target.iter('object'):
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            # 过滤掉错误标记的box
            if bndbox[0]>=bndbox[2] or bndbox[1]>=bndbox[3]:
                continue
            valid_boxes += 1
        if valid_boxes > 0:
            print('copy....')
            shutil.copy(os.path.join(xml_path, f), os.path.join(valid_xml_path, f))


if __name__ == '__main__':
    clean_dataset()
