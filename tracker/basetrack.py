"""
basetrack.py: 定义基础跟踪器的类。
BaseTracker类包含了初始化、更新、重置等方法，以及一些属性如 tracklets、results等。
"""
import numpy as np
from collections import OrderedDict


# tracker的种类,由 state 属性标记
class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0  # 创建BaseTrack对象的数量
    track_id = 0
    is_activated = False  # 跟踪对象是否被激活
    state = TrackState.New
    history = OrderedDict()  # 每个对象的历史轨迹，使用有序字典类型
    features = []  # 每个对象的特征向量列表
    curr_feature = None  # 每个对象的当前特征向量
    score = 0  # 每个对象的置信度分数
    start_feature = None  # 每个对象出现在视频中的第一帧编号
    frame_id = 0  # 每个对象出现在视频中的最后一帧编号
    time_since_update = 0  # 记录每个对象距离上次更新过了多少帧

    # multi-camera
    location = (np.inf, np.inf)

    # 使用@property装饰器定义一个名为end_frame的属性方法。
    # 返回frame_id作为结束帧编号。
    @property
    def end_frame(self):
        return self.frame_id

    # 使用@staticmethod装饰器定义一个名为next_id的静态方法。
    # 在创建新对象时调用该方法生成唯一标识，并将_count加一。
    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    # 根据输入参数激活新建或重匹配的目标，并更新其属性值。
    def activate(self, *args):
        raise NotImplementedError

    # 根据运动模型预测目标在下一帧中的位置和状态，并更新其属性值。
    def predict(self):
        raise NotImplementedError

    # 根据检测结果和特征匹配更新目标在当前帧中的位置和状态，并更新其属性值。
    def update(self, *args, **kwargs):
        raise NotImplementedError

    # 在目标丢失时调用该方法将其状态改为Lost，并更新其属性值。
    def mark_lost(self):
        self.state = TrackState.Lost

    # 在目标移除时调用该方法将其状态改为Removed，并更新其属性值。
    def mark_removed(self):
        self.state = TrackState.Removed