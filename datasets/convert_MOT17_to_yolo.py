'''
创建以下四个目录，用于存放图片和标签
images/train
images/val
labels/train
labels/val
'''
import os
import shutil
import numpy as np
import configparser
if not os.path.exists('MOT17/images'):
    os.makedirs('MOT17/images/train')
    os.makedirs('MOT17/images/val')
if not os.path.exists('MOT17/labels'):
    os.makedirs('MOT17/labels/train')
    os.makedirs('MOT17/labels/val')


def convert(imgWidth, imgHeight, left, top, width, height):
    x = (left + width / 2.0) / imgWidth
    y = (top + height / 2.0) / imgHeight
    w = width / imgWidth
    h = height / imgHeight
    return ('%.6f'%x, '%.6f'%y, '%.6f'%w, '%.6f'%h) # 保留6位小数


count = 0
for mot_dir in os.listdir('D:/毕业设计/实验/dataset/MOT17/test'):  # mot_dir是例如MOT17-02-FRCNN这种
    det_path = os.path.join('D:/毕业设计/实验/dataset/MOT17/test', mot_dir, 'det/det.txt')  # det.txt路径
    dets = np.loadtxt(det_path, delimiter=',')  # 读取det.txt文件
    ini_path = os.path.join('D:/毕业设计/实验/dataset/MOT17/test', mot_dir, 'seqinfo.ini')  # seqinfo.ini路径
    conf = configparser.ConfigParser()
    conf.read(ini_path)  # 读取seqinfo.ini文件
    seqLength = int(conf['Sequence']['seqLength'])  # MOT17-02-FRCNN序列的长度
    imgWidth = int(conf['Sequence']['imWidth'])  # 图片宽度
    imgHeight = int(conf['Sequence']['imHeight'])  # 图片长度
    for det in dets:
        frame_id, _, left, top, width, height = int(det[0]), det[1], det[2], det[3], det[4], det[5]
        box = convert(imgWidth, imgHeight, left, top, width, height)
        if '-' in ''.join(box) or float(box[0]) > 1.0 or float(box[1]) > 1.0 or float(box[2]) > 1.0 or float(
                box[3]) > 1.0:
            print(imgWidth, imgHeight, left, top, width, height)
            print(box)
            count += 1
            continue
        image_name = mot_dir + '-' + '%06d' % frame_id + '.jpg'  # MOT17-02-FRCNN-000001.jpg
        label_name = mot_dir + '-' + '%06d' % frame_id + '.txt'  # MOT17-02-FRCNN-000001.txt
        oldimgpath = os.path.join('D:/毕业设计/实验/dataset/MOT17/test', mot_dir, 'img1',
                                  '%06d' % frame_id + '.jpg')  # train/MOT17-02-FRCNN/img1/000001.jpg
        if frame_id <= seqLength:  # 前一半划分给训练集
            newimgpath = os.path.join('MOT17/images', 'test', image_name)  # images/train/MOT17-02-FRCNN-000001.jpg
            labelpath = os.path.join('MOT17/labels', 'test', label_name)  # labels/train/MOT17-02-FRCNN-000001.txt
        else:  # 后一半划分给验证集
            newimgpath = os.path.join('MOT17/images', 'val', image_name)  # images/val/MOT17-02-FRCNN-000001.jpg
            labelpath = os.path.join('MOT17/labels', 'val', label_name)  # labels/val/MOT17-02-FRCNN-000001.txt
        if not os.path.exists(newimgpath):  # 如果图片没复制过去，就复制，
            shutil.copyfile(oldimgpath, newimgpath)  # 把旧图片复制到新的地方
        with open(labelpath, 'a') as f:  # 写label文件
            f.write(f'0 {box[0]} {box[1]} {box[2]} {box[3]}\n')
print('count:', count)
