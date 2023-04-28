'''
kalman filter
'''

import numpy as np
import scipy.linalg

"""
卡方分布的0.95分位数与自由度的关系 (N = 1,2,...,9)
N:构成卡方分布的正态随机变量的个数
马氏距离可以通过阈值判断异常数据，其平方服从卡方分布。若以0.95置信水平检测异常，可以用卡方分布的0.95分位数作为阈值。
"""
chi2inv95 ={
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}


class KalmanFilter(object):
    """
    状态空间: [x, y, a, h, vx, vy, va, vh]
        (x, y)-检测框中心点坐标;  a-宽高比;  h-高度; vx vy va vh:各自速度
    检测框的运动遵循恒速模型，观测变量是检测框的位置:(x, y, a, h)
    """
    def __init__(self):
        ndim, dt = 4, 1.  # ndim:状态向量维度， dt:状态转移矩阵中的时间间隔

        # self._motion_mat(状态转移矩阵F):描述系统状态如何随时间变化  x' = Fx
        # 1 0 0 0 dt 0 0 0 | 0 1 0 0 0 dt 0 0 | 0 0 1 0 0 0 dt 0 | 0 0 0 1 0 0 0 dt
        # 0 0 0 0 1 0 0 0  | 0 0 0 0 0 1 0 0  | 0 0 0 0 0 0 1 0  | 0 0 0 0 0 0 0 1
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # self._update_mat(观测矩阵):描述观测值如何与状态向量相关
        self._update_mat = np.eye(ndim, 2 * ndim)
        # self._std_weight_position:位置不确定性的权重
        self._std_weight_position = 1. / 20
        # self._std_weight_velocity:速度不确定性的权重
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        根据第一个测量值 measurement计算滤波器的初始状态估计和初始协方差矩阵
        Args:
            measurement:(ndarray)-测量值, 检测框坐标(x,y,a,h)
        Returns:
            (ndarray, ndarray)
            返回新轨迹的均值向量(8维) + 协方差矩阵(8×8维)
            未观察到的速度被初始化为 0
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],  # 根据目标的高度来调整协方差矩阵中位置和速度的方差
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        # 协方差矩阵
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        预测方程：
            Xk = Fk · Xk-1 + Bk · Uk
            Pk = Fk · Pk-1 Fk.T + Qk
        Args:
            mean: (ndarray)-上一个时刻中物体状态的8维均值向量
            covariance: (ndarray)-物体状态在上一时刻的8×8维协方差矩阵
        Returns:
            (ndarray, ndarray)
            预测状态的均值向量 + 协方差矩阵
            未观察到的速度被初始化为 0
        """
        # 位置标准差
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        # 速度标准差
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        # 初始化噪声矩阵Q
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # 更新状态向量mean, 即x' = Ax
        mean = np.dot(mean, self._motion_mat.T)
        # 下一时刻的协方差 = 当前协方差矩阵 + 过程噪声协方差矩阵(状态转移时的不确定性)
        # P' = APA(^T) + Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        """
        将状态分布投射到测量空间:
            u = Hk · Xk
            p = Hk · Pk · Hk.T
        Args:
            mean: (ndarray)-状态均值向量(8维)
            covariance: (ndarray)-状态协方差矩阵(8×8维)
        Returns:
            (ndarray, ndarray)
            给定状态估计的投影均值 + 协方差矩阵
        """
        # 位置标准差
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        # 初始化噪声矩阵R-检测器的噪声矩阵(4×4维)
        innovation_cov = np.diag(np.square(std))
        # 将均值向量映射到测量空间，即Hx'
        mean = np.dot(self._update_mat, mean)
        # 将协方差矩阵映射到测量空间，即HP'H^T
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        # HP'H^T + R
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        预测方程(矢量)：
            Xk = Fk · Xk-1 + Bk · Uk
            Pk = Fk · Pk-1 Fk.T + Qk
        Args:
            mean: (ndarray)-前一个时刻中物体状态的N×8维均值矩阵
            covariance: (ndarray)-对象状态在上一时刻的N×8×8维协方差矩阵
        Returns:
            (ndarray, ndarray)
            预测状态的均值向量 + 协方差矩阵
            未观察到的速度被初始化为 0
        """
        # 位置标准差
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        # 速度标准差
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        # 初始化噪声矩阵Q
        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        # 将均值向量映射到测量空间，即Hx'
        mean = np.dot(mean, self._motion_mat.T)
        # 将协方差矩阵映射到测量空间，即HP'H^T
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        # 下一时刻的协方差 = 当前协方差矩阵 + 噪声协方差
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        运行卡尔曼滤波校正步骤，根据观测值来修正物体状态的预测：
            K‘ = Pk Hk.T (Hk Pk Hk.T + Rk)^(-1)
            Xk' = Xk + K'(Zk - Hk Xk)
            Pk' = Pk - K' Hk Pk
            Pk' = Pk - K' S K'.T
            S = Hk Pk Hk.T + Rk
        Args:
            mean: (ndarray)-预测的状态的均值向量(8维)
            covariance: (ndarray)-状态的协方差矩阵(8×8维)  covariance = Hk Pk Hk.T + Rk
            measurement: (ndarray)-测量向量(x, y, a, h)
        Returns:
            (ndarray, ndarray)-返回测量校正后的状态分布
        """
        # 将mean和covariance映射到测量空间，得到Hx'和S
        projected_mean, projected_cov = self.project(mean, covariance)
        # 使用scipy.linalg.cho_factor()计算projected_cov的chol_factor分解
        # lower=True:计算下三角Cholesky因子; check_finite=False:不检查projected_cov是否只有有限数值
        # chol_factor分解将对称正定矩阵分解为一个下三角矩阵和其转置矩阵的乘积 A=LL^T
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        # 计算卡尔曼增益
        # scipy.linalg.cho_solve():返回一个数值x,表示Ax=b的解
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # 观测值与预测值之间的差异，即z - Hx'
        innovation = measurement - projected_mean
        # 新的状态向量 + 协方差矩阵，即x = x' + Ky
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        # P = (I - KH)P'
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """
        计算状态分布和测量之间的门限距离,如果'only_position'为 False,则卡方分布自由度为 4，否则为 2。
        Args:
            mean: (ndarray)-状态分布的均值向量(8维)
            covariance: (ndarray)-状态分布的协方差矩阵(8×8维)
            measurements: (ndarray)-包含N个测量值的矩阵(N×4维),每个测量值格式(x, y, a, h)
            only_position:若为True,距离计算将只针对包围盒的中心位置进行。
        Returns:
            返回一个长度为 N的数组，其中第 i个元素包含了(均值向量, 协方差矩阵)和测量值[i]之间的马氏距离。

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False,
                overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError('invalid distance metric')