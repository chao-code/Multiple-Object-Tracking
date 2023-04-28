import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
import kalman_filter
import math
from cython_bbox import bbox_overlaps as bbox_ious


def merge_matches(m1, m2, shape):
    """
    矩阵匹配：比较两个稀疏矩阵的相似度，找出两个矩阵中非零元素的位置是否相同或接近，以及非零元素的值是否相等或相似。
    Args:
        m1: 待匹配矩阵 1
        m2: 待匹配矩阵 2
        shape: 三元组(O,P,Q)——O(m1行数); P(m1列数,m2行数); Q(m2列数)

    Returns:
        match: list-包含两个矩阵中非零元素对应的行列索引
        unmatched_0: tuple-第一帧中未匹配目标行索引
        unmatched_Q: tuple-第二帧中未匹配目标行索引
    """
    O, P, Q = shape
    # m1,m2是稀疏矩阵表达形式
    m1 = np.asarray(m1)  # [O,P]
    m2 = np.asarray(m2)  # [P,Q]

    # 创建稀疏矩阵
    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    # 创建掩码
    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))  # O未匹配行索引
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))  # Q^T未匹配行索引

    return match, unmatched_O, unmatched_Q


def _indices_to_matched(cost_matrix, indices, thresh):
    """
    从代价矩阵中找出匹配与未匹配的索引
    Args:
        cost_matrix:二维数组，表示两目标之间的代价
        indices:二维数组，代价矩阵的行、列索引
        thresh:匹配的阈值

    Returns:
        matches:满足阈值的索引对
        unmatched_a:元组，第一组中未匹配的行索引
        unmatched_b:元组，第二组中未匹配的行索引
    """
    # 利用zip和tuple将indices转化为元组，并用来索引cost_matrix，得到matched_cost数组
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


# TODO
def linear_assignment(cost_matrix, thresh):
    """
    Jonker-Volgenant算法，从代价矩阵中找出匹配与未匹配的索引
    Args:
        cost_matrix, thresh:
    Returns:
        matches(numpy.ndarray), unmatched_a, unmatched_b
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:  # 大于0表示该行有匹配的值，并将其添加到matches列表中
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]  # 取行索引
    unmatched_b = np.where(y < 0)[0]  # 取行索引
    matches = np.asarray(matches)  # 不会复制输入对象
    return matches, unmatched_a, unmatched_b


# TODO
def bbox_iou(bbox_a, bbox_b):
    """
    计算两组方框之间的IOU值。
    Args:
        bbox_a: numpy.ndarray | 二维数组 | shape:(N, 4), N代表 bbox_a包含的方框个数
        bbox_b: numpy.ndarray | 二维数组 | shape:(K, 4), N代表 bbox_b包含的方框个数
    Returns:
        iou: numpy.ndarray | 二维数组 | shape为(N, K)
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    iou = area_i / (area_a[:, None] + area_b - area_i)
    return iou


# TODO
def ious(atlbrs, btlbrs):
    """
    计算多个检测框的IoU
    Args:
        atlbrs: list, element:tuple——检测框左上角和右下角坐标 (0, 0, 2, 2)
        btlbrs: list, element:tuple——检测框左上角和右下角坐标 (1, 1, 3, 3)
    Returns: ious:matrix
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),  # list->np.array 内存连续存储
        np.ascontiguousarray(btlbrs, dtype=float)   # list->np.array 内存连续存储
    )
    return ious


# TODO
def iou_distance(atracks, btracks):
    """
    计算两组轨迹之间的IoU距离(代价)，即 1-IoU。表示两组轨迹之间的相似度。
    Args:
        atracks: list of tracks. element:Track object or np.ndarray——(left top x, left top y, right bottom x, right bottom y)
        btracks: list of tracks. element:Track object or np.ndarray——(left top x, left top y, right bottom x, right bottom y)

    Returns: IoU distance matrix
    """
    # 判断 atracks 和 btracks 是不是列表，以及列表中的元素是不是np.ndarray类型。
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:  # 若 atracks 和 btracks 不是列表，则使用track.tlbr提取检测框坐标
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    计算两组轨迹之间的IoU距离(代价)，即 1-IoU。表示两组轨迹之间的相似度。
    考虑了运动影响。
    Args:
        atracks: list of tracks. element:Track object or np.ndarray——(left top x, left top y, right bottom x, right bottom y)
        btracks: list of tracks. element:Track object or np.ndarray——(left top x, left top y, right bottom x, right bottom y)

    Returns: IoU distance matrix
    """
    # 判断 atracks 和 btracks 是不是列表，以及列表中的元素是不是np.ndarray类型。
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:  # 若 atracks 和 btracks 不是列表，则使用track.tlwh_to_tlbr(track.pred_bbox)提取检测框坐标
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    计算两组轨迹之间的嵌入距离，即两组轨迹对应的外观特征向量之间的距离。
    外观特征向量：描述对象外观的高维向量，可以用来比较对象之间的相似度；
    距离：余弦距离、欧氏距离、马氏距离
    Args:
        tracks: list | elements:STrack 表示卡尔曼滤波后的跟踪对象，外观特征向量：smooth_feat
        detections: list | elements:BaseTrack 表示检测框检测到的跟踪对象，外观特征向量：curr_feat
        metric: str 表示距离度量方式，默认为 cosine
    Returns:
        cost_matrix:嵌入距离矩阵
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    # for i, track in enumerate(tracks):
        # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    # cdisk函数返回track_features和det_features的‘metric’距离
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # 归一化特征
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    根据卡尔曼滤波器的状态分布，将代价矩阵中不匹配的代价矩阵元素设为无穷大。
    这样可以减少错误匹配的可能性。
    Args:
        kf: 卡尔曼滤波器对象
        cost_matrix: 代价矩阵(N×M维度),N是跟踪对象数量, M是检测对象数量
        tracks: list | 跟踪对象
        detections: list | 检测对象
        only_position: bool | 表示是否只考虑跟踪对象的位置信息进行门限判断
    Returns:
        cost_matrix: 代价矩阵
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4  # 用于门限判决的维度
    gating_threshold = kalman_filter.chi2inv95[gating_dim]  # 卡尔曼滤波结果作为阈值
    measurements = np.asarray([det.to_xyah() for det in detections])  # 检测目标[x1, y1, x2, y2] -> [x, y, a, h]
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """
    根据卡尔曼滤波器的状态分布和门限距离，对代价矩阵进行加权融合。
    这样可以综合考虑跟踪对象和检测对象之间的外观相似度和运动一致性。
    Args:
        kf: 卡尔曼滤波器对象
        cost_matrix: 代价矩阵(N×M维度),N是跟踪对象数量, M是检测对象数量
        tracks: list | 跟踪对象
        detections: list | 检测对象
        only_position: bool | 表示是否只考虑跟踪对象的位置信息进行门限判断
        lambda_: float | 表示代价矩阵和门限距离的加权系数
    Returns:
        cost_matrix: 代价矩阵
    """
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4  # 用于门限判决的维度
    gating_threshold = kalman_filter.chi2inv95[gating_dim]  # 卡尔曼滤波结果作为阈值
    measurements = np.asarray([det.to_xyah() for det in detections])  # 检测目标[x1, y1, x2, y2] -> [x, y, a, h]
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance  # 将代价矩阵和门限距离进行加权融合。
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """
    根据 reid相似度和 iou相似度，对代价矩阵进行加权融合。
    这样可以综合考虑跟踪对象和检测对象之间的外观相似度和空间重叠程度。
    Args:
        cost_matrix: 代价矩阵(N×M维度),N是跟踪对象数量, M是检测对象数量
        tracks: list | 跟踪对象
        detections: list | 检测对象
    Returns:
        fuse_cost: numpy.ndarray
    """
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix  # reid相似度
    iou_dist = iou_distance(tracks, detections)  # 计算跟踪对象和目标对象的交并比距离
    iou_sim = 1 - iou_dist  # 将交并比距离转化为iou相似度
    fuse_sim = reid_sim * (1 + iou_sim) / 2  # reid相似度和iou相似度加权融合
    det_scores = np.array([det.score for det in detections])  # 获取检测对象的置信分数(det_scores)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)  # 将其扩展为与代价矩阵同样的形状。
    # 将检测对象的置信分数也考虑进相似度的计算中，这样可以增加对检测结果的信任度。
    # 但是这个代码可能会导致一些问题，比如当置信分数很高,而reid相似度很低时，会造成错误的匹配。
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim  # 将融合后的相似度转化为代价值
    return fuse_cost

# TODO
def fuse_score(cost_matrix, detections):
    """
    根据 iou相似度和检测对象的置信分数，对代价矩阵进行加权融合。
    这样可以综合考虑跟踪对象和检测对象之间的空间重叠程度和检测结果的可信度。
    Args:
        cost_matrix: 代价矩阵(N×M维度),N是跟踪对象数量, M是检测对象数量
        detections: list | 检测对象
    Returns:
        fuse_cost: numpy.ndarray
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix  # reid相似度
    det_scores = np.array([det.score for det in detections])  # 获取检测对象的置信分数(det_scores)
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)  # 将其扩展为与代价矩阵同样的形状。
    fuse_sim = iou_sim * det_scores  # iou相似度和检测对象的置信分数加权融合
    fuse_cost = 1 - fuse_sim  # 将融合后的相似度转化为代价值
    return fuse_cost