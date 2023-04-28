import torch
import math
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious


a1 = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
b1 = [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
a = np.ascontiguousarray(a1, dtype=float)
b = np.ascontiguousarray(b1, dtype=float)

c_a = (a[:, 0] + a[:, 2]) / 2
c_b = (b[:, 0] + b[:, 2]) / 2
print('c_a:', c_a[2])
print('c_b:', c_b[0])