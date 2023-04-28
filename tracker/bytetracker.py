import numpy as np
from kalman_filter import KalmanFilter
import matching
from basetrack import BaseTrack, TrackState


"""表示单个跟踪对象的类，每个轨迹都有自己的属性，如id、边界框、检测框、状态等"""
class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score):
        """新的检测目标初始化为 tracker，state=0，is_activated=False"""
        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=float)  # 保存目标框属性
        self.kalman_filter = None
        self.mean, self.covariance = None, None  # 保存卡尔曼滤波对于这个轨迹的mean和convariance
        self.is_activated = False  # tracker的状态：激活、休眠

        self.score = score  # 轨迹分数，采用当前帧的目标框分数作为轨迹分数
        self.tracklet_len = 0  # 轨迹追踪的帧数，初始为0，后面每追踪成功（调用update方法），则+1

    def predict(self):
        mean_state = self.mean.copy()  # mean:目标位置、速度
        if self.state != TrackState.Tracked:  # 若目标不是跟踪状态
            mean_state[7] = 0  # vy2置为0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """
        激活这条轨迹，如果是当前帧是视频流的第一帧，那么设置self.is_activated=True，否则这个属性依旧是False；
        设置state属性为TrackState.Tracked
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()  # self.track_id: self.is_activated=True轨迹的id，全局唯一标志
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        重新激活这个轨迹（之前处于丢失状态）
        重新计算卡尔曼滤波的 mean, covariance；
        重新计算 self.tracklet_len；
        更新 self.frame_id,self.score；
        """
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True  # 轨迹已被激活
        self.frame_id = frame_id
        if new_id:  # 根据new_id参数判断是否需要给目标分配一个新的轨迹编号
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        更新已匹配轨迹信息，设置 is_activate=True;
        主要更新 self.frame_id,  self.score,  tracklet_len,  卡尔曼滤波 mean, conv;
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, args, frame_rate=30):
        self.tracked_stracks = []  # 维护当前追踪中的轨迹列表 | type: list[STrack]
        self.lost_stracks = []  # 维护到前一帧为止的追踪中丢失了检测框的轨迹列表 | type: list[STrack]
        self.removed_stracks = []  # 维护删除的轨迹列表 | type: list[STrack]

        self.frame_id = 0  # 当前视频流的帧id，默认从1开始，每次调用update方法，+1
        self.args = args  # 包含了从命令行解析得到的参数

        self.det_thresh = args.track_thresh + 0.1  # 检测阈值，高于这个阈值且无法在当前轨迹中找到匹配的检测框可以生成一条新的轨迹。
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, output_results):
        self.frame_id += 1

        """保存当前帧处理结束之后的 tracker"""
        activated_starcks = []  # 保存当前帧匹配到持续追踪的轨迹
        refind_stracks = []  # 保存当前帧匹配到之前目标丢失的轨迹
        lost_stracks = []  # 保存当前帧没有匹配到目标的轨迹
        removed_stracks = []  # 保存当前帧删除的轨迹

        """step1: 将objects转换为[x1，y1，x2，y2，score]的格式，并构建strack"""
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            # yolov5输出的结果，需要将第五列（置信度）和第六列（类别得分）相乘得到scores，并提取前四列作为bboxes。
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1 y1 x2 y2

        """根据scroe和track_thresh将strack分为detetions(dets)和detections_low(dets_second)"""
        remain_inds = scores > self.args.track_thresh  # 0.5 高分检测框，score 是 cls_conf * obj_conf
        inds_low = scores > 0.1  # 低分检测框最小值
        inds_high = scores < self.args.track_thresh  # 低分检测框最大值

        inds_second = np.logical_and(inds_low, inds_high)  # 筛选分数处于0.1<分数<阈值的(AND运算)

        # 高分检测框以及置信度（cls_conf * obj_conf）
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]

        # 低分检测框以及置信度（cls_conf*obj_conf）
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            # 将检测框初始化为 tracker
            # bbox和score；均值、方差为None、is_activated=False、tracklet_len=0、state=0
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        """遍历tracked_stracks（所有的轨迹），如果track还activated，加入tracked_stracks（继续匹配该帧），否则加入unconfirmed"""
        unconfirmed = []  # 未激活轨迹
        tracked_stracks = []  # 已匹配，type: list[STrack]
        # is_activated表示除了第一帧外中途只出现一次的目标轨迹（新轨迹，没有匹配过或从未匹配到其他轨迹）
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)  # 新track
            else:
                tracked_stracks.append(track)  # 优先关联激活状态的（正常的）trackers

        ''' Step 2: First association, with high score detection boxes'''
        # 将 track_stracks 和 lost_stracks 合并得到track_pool
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # 将 strack_pool 送入 muti_predict 进行预测, 更新均值和方差
        STrack.multi_predict(strack_pool)

        # 计算 strack_pool（当前帧的预测框和之前未匹配到轨迹的bbox）和 detections(高分) 的 iou_distance(代价矩阵)
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            # 融合IoU代价矩阵和检测框置信度，确保dists更可信 dists = 1 - (IoU * score)
            dists = matching.fuse_score(dists, detections)

        # 用match_thresh = 0.8(越大说明iou越小)过滤较小的iou，利用匈牙利算法进行匹配，得到matches, u_track, u_detection
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # 根据匹配到的检测框，更新参数
        # 遍历matches，如果state为Tracked，调用update方法，并加入到activated_stracks，否则调用re_activate，并加入refind_stracks
        # matches = [itracked, idet] itracked指的是轨迹的索引，idet 指的是当前目标框的索引，意思是第几个轨迹匹配第几个目标框
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # 更新轨迹的bbox为当前匹配到的bbox
                # 更新 tracklet_len，frame_id，坐标，置信度，卡尔曼的均值和方差，state=1，is_activated=True，track_id 不变
                track.update(detections[idet], self.frame_id)
                # activated_starcks是目前能持续追踪到的轨迹
                activated_starcks.append(track)  # 放入激活列表
            else:
                # 若不是正常tracker（这里只有丢失的），丢失的tracker根据det更新参数
                # tracklet_len=0，frame_id，坐标，置信度，卡尔曼的均值和方差，state=1，is_activated=True，track_id 不变
                track.re_activate(det, self.frame_id, new_id=False)
                # refind_stracks是重新追踪到的轨迹
                refind_stracks.append(track)  # 放入重现列表

        ''' Step 3: Second association, with low score detection boxes'''
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        # 找出第一次匹配中没匹配到的轨迹（激活状态）
        # u_track: 高分检测框没有匹配到的trackers，这里有一个tracker的种类过滤
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # 计算 r_tracked_stracks 和 detections_second 的 iou_distance(代价矩阵)
        # 以下和第一次关联相同
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # 用match_thresh = 0.5过滤较小的iou，利用匈牙利算法进行匹配，得到matches, u_track, u_detection
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        # 遍历matches，如果state为Tracked，调用update方法，并加入到activated_stracks，否则调用re_activate，并加入refind_stracks
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:  # 这个 else 分支应该多余了
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 遍历u_track（第二次匹配也没匹配到的轨迹），将state不是Lost的轨迹，调用mark_losk方法，并加入lost_stracks，等待下一帧匹配
        # lost_stracks加入上一帧还在持续追踪但是这一帧两次匹配不到的轨迹
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                # 若不是丢失种类的tracker，改变种类，放入 lost_stracks
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # 尝试将第一次关联没有匹配到的高分检测框与中途第一次出现的轨迹（未激活轨迹）匹配
        # 当前帧的目标框会优先和长期存在的轨迹（包括持续追踪的和断追的轨迹）匹配，再和只出现过一次的目标框匹配
        detections = [detections[i] for i in u_detection]
        # 计算unconfirmed和detections的iou_distance(代价矩阵)
        # unconfirmed是不活跃的轨迹（过了30帧）
        # 与休眠状态的tracker匹配
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        # 用match_thresh = 0.7过滤较小的iou，利用匈牙利算法进行匹配，得到matches, u_track, u_detection
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # 遍历matches，如果state为Tracked，调用update方法，并加入到activated_stracks，否则调用re_activate，并加入refind_stracks
        for itracked, idet in matches:
            # 匹配到休眠的 tracker，将其参数和状态更新，放入到激活列表中
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        # 遍历u_unconfirmed，调用mark_removd方法，并加入removed_stracks
        for it in u_unconfirmed:
            # 中途出现一次的轨迹和当前目标框匹配失败，删除该轨迹（认为是检测器误判）
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # 遍历u_detection（前两步都没匹配成功的目标框），对于score大于high_thresh，调用activate方法，并加入activated_stracks
        # 此时还没匹配的u_detection将赋予新的id
        # 对没有匹配到的检测框判断，<0.6 的过滤掉，>0.6 更新 tracker_id，均值、方差，状态，is_activate=False
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            # 只有第一帧新建的轨迹会被标记为is_activate=True，其他帧不会
            track.activate(self.kalman_filter, self.frame_id)
            # 把新的轨迹加入到当前活跃轨迹中
            activated_starcks.append(track)

        """ Step 5: Update state"""
        # 遍历lost_stracks，对于丢失超过max_time_lost(30)的轨迹，调用mark_removed方法，并加入removed_stracks
        for track in self.lost_stracks:
            # 有的丢失 tracker 在第一次关联中已经匹配到了，因此 end_frame 已经更新
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)  # 放入删除列表

        # print('Ramained match {} s'.format(t4-t3))

        # 根据当前帧的匹配结果，过滤前一帧(正常种类)tracker，包括(上一帧)新的、正常的，tracked_stracks 中若在当前帧匹配到了，state 必会设为1（正常种类）
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 上一步（过滤的）tracked_stracks 和 当前的激活列表（主要是当前帧产生的新的 tracker）
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        # 上一步 tracked_stracks 和 当前的重新列表（主要是丢失的又重新匹配的）
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # 遍历lost_stracks,去除tracked_stracks和removed_stracks中存在的轨迹
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        # 合并self.lost_stracks 和 当前帧的新的丢失的 tracker
        self.lost_stracks.extend(lost_stracks)
        # 合并以前帧和当前帧删除类型的 traker
        self.removed_stracks.extend(removed_stracks)
        # 调用remove_duplicate_stracks函数，计算tracked_stracks，lost_stracks的iou_distance，对于iou_distance<0.15的认为是同一个轨迹，
        # 就是去重，在 tracked_stracks 和 lost_stracks 中，将iou大的、时间久的tracker保留
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # 输出，当前帧激活状态的tracker，新的、丢失的、删除的都不会输出
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb