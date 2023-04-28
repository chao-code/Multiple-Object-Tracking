import os
import cv2
import torch
import numpy as np


class TrackerLoader(torch.utils.data.Dataset):
    def __init__(self, DATASET_ROOT, path, img_size, seq=None) -> None:
        """
        从不同格式的数据集中加载图像数据。
        Args:
            path: 图像的文件路径
            img_size: 目标图像大小,表示宽度和高度，默认为 1280 | 元组、int
            seq: 数据集的序列名，用于过滤数据，默认为 None
        """
        super().__init__()
        self.DATASET_ROOT = DATASET_ROOT
        self.img_files = []

        # 如果format为’yolo’，则假设路径是一个文件，并使用open函数打开该文件，并逐行读取内容。
        # 每一行表示一个图像文件的绝对路径。如果该行中包含了seq参数指定的序列名，则将该行添加到img_files列表中
        assert os.path.isfile(path), f'your path is {path}, path must be your path file'
        with open(path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                elems = line.split('/')
                if elems[-2] in seq:  #
                    self.img_files.append(os.path.join(self.DATASET_ROOT, line))  # add abs path

        # 使用assert语句检查img_files列表是否为空
        assert self.img_files is not None

        # 根据img_size参数判断目标图像大小是一个整数还是一个列表/元组，并分别赋值给width和height属性
        if type(img_size) == int:
            self.width, self.height = img_size, img_size
        elif type(img_size) == list or type(img_size) == tuple:
            self.width, self.height = img_size[0], img_size[1]

    def __getitem__(self, index):
        """
        return: img after resize and origin image, class(torch.Tensor)
        """
        img_path = self.img_files[index]  # 当前图像文件的路径
        # 使用cv2.imread函数读取图像文件，得到一个(H,W,C)形状的数组，存储在img变量中
        img = cv2.imread(img_path)  # (H,W,C)
        # 使用assert语句检查img变量是否为空，如果为空，则抛出异常，并显示失败加载的图像路径
        assert img is not None, f'Fail to load image{img_path}'

        return img

    def __len__(self):
        return len(self.img_files)