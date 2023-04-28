import time

class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0  # 调用次数
        self.start_time = 0.
        self.diff = 0.  # 单次执行的时间差
        self.average_time = 0.
        self.duration = 0.  # 持续时间

    """计时开始"""
    def tic(self):
        self.start_time = time.time()

    """计时结束"""
    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            self.duration = self.average_time
        else:
            self.duration = self.diff
        return self.duration

    """清零所有属性"""
    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        self.duration = 0.
