import numpy as np
import cv2
import os

path = '/path/to/your/video/file.mp4'
video_file_name = os.path.basename(path)
video_file_name = os.path.splitext(video_file_name)[0]
print(video_file_name)