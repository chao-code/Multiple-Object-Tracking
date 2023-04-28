from bytetracker import BYTETracker

class ByteTrack():
    def __init__(self, opts):
        self.opts = opts
        self.tracker = BYTETracker(opts, frame_rate=self.opts.fps)


    def track(self, boxes):
        if boxes is not None:
            online_targets = self.tracker.update(boxes)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_cls = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.opts.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self.opts.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    online_cls.append(0)
            return online_tlwhs, online_ids, online_cls