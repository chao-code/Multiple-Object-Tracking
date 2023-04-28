"""
将 train set、val set和 test set路径分别存储的到.txt文件
"""

import os


def generate(dir, label):
    files = os.listdir(dir)
    files.sort()
    print
    '****************'
    print
    'input :', dir
    print
    'start...'
    if label == 0:
        listText = open('./MOT17/test_list.txt', 'a')
        for file in files:
            fileType = os.path.split(file)
            if fileType[1] == '.txt':
                continue
            name = 'datasets/MOT17/images/test/'+file + '\n'
            listText.write(name)
        listText.close()
    elif label == 1:
        listText = open('./MOT17/train_list.txt', 'a')
        for file in files:
            fileType = os.path.split(file)
            if fileType[1] == '.txt':
                continue
            name = 'datasets/MOT17/images/train/' + file + '\n'
            listText.write(name)
        listText.close()
    else:
        listText = open('./MOT17/val_list.txt', 'a')
        for file in files:
            fileType = os.path.split(file)
            if fileType[1] == '.txt':
                continue
            name = 'datasets/MOT17/images/val/' + file + '\n'
            listText.write(name)
        listText.close()
    print
    'down!'
    print
    '****************'

if __name__ == '__main__':
    outer_path = './MOT17/images/'  # 这里是你的图片的目录
    i = 0
    folderlist = os.listdir(outer_path)  # 列举文件夹
    for folder in folderlist:
        generate(os.path.join(outer_path, folder), i)
        i += 1
