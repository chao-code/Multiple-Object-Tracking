import os
import numpy as np
import motmetrics as mm
mm.lap.default_solver = 'lap'

GT_PREFIX = './datasets/MOT17/images/gt'
RESULT_PREFIX = './results/GT_results'

class SeqEvaluator:
    def __init__(self, seq_name, gt_name, data_type='mot', ignore_cls_idx=set()) -> None:
        """
        seq_name: name of the sequence
        gt_name:  name of the gt sequence
        data_type: data format: "mot"
        ignore_cls_idx: the class of object ihnored | set
        """
        self.seq_name = seq_name
        self.data_type = data_type
        self.ignore_cls_idx = ignore_cls_idx

        if self.data_type == 'mot':
            self.valid_cls_idx = {i for i in range(1, 11)} - self.ignore_cls_idx
        else:
            raise NotImplementedError

        self.gt_frame_dict = self.read_result(gt_name, is_gt=True)
        self.gt_ignore_frame_dict = self.read_result(gt_name, is_ignore=True)
        self.acc = mm.MOTAccumulator(auto_id=True)

    def read_result(self, seq_name, is_gt=False, is_ignore=False) -> dict:
        """
        result to dict
        """
        result_dict = dict()
        if is_gt or is_ignore:
            seq_name = os.path.join(GT_PREFIX, seq_name)
        else:
            seq_name = os.path.join(RESULT_PREFIX, seq_name)
        with open(seq_name, 'r') as f:
            for line in f.readlines():
                line = line.replace(' ', ',')
                linelist = line.spilt(',')
                fid = int(linelist[0])
                result_dict.setdefault(fid, list())

                if is_gt:
                    label = int(float(linelist[7]))
                    mark = int(float(linelist[6]))
                    if mark == 0 or label not in self.valid_cls_idx:
                        continue
                    score = 1
                elif is_ignore:
                    label = int(float(linelist[7]))
                    if self.data_type == 'mot':
                        vis_ratio = float(linelist[8])
                    else:
                        raise NotImplementedError
                    if label not in self.ignore_cls_idx and vis_ratio >= 0:
                        continue
                    score = 1
                else:
                    score = -1

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(float(linelist[1]))

                result_dict[fid].append((tlwh, target_id, score))

            f.close()
        return result_dict

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids) -> None:
        """
        evaluate the metrics of a frame
        frame_id: the frame ordinal number of the current frame | int
        trk_tlwhs: top-left-width-height | tuple
        trk_ids: target ID | int
        """
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tiwhs, gt_ids = self.unzip_objs(gt_objs)[:2]

        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = self.unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(lambda a: np.asarray(a, dtype=int), [match_is, match_js])
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # IoU matching
        # TODO: more concise method
        iou_distance = mm.distances.iou_matrix(gt_tiwhs, trk_tlwhs, max_iou=0.5)

        self.acc.update(gt_ids, trk_ids, iou_distance)

    def eval_seq(self) -> mm.MOTAccumulator:
        self.acc = mm.MOTAccumulator(auto_id=True)
        result_frame_dict = self.read_result(self.seq_name, is_gt=False)
        frames = sorted(list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys())))

        for frame_id in frames: # 评估每一帧
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = self.unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids)

        return self.acc

    def unzip_objs(self, objs):
        if len(objs) > 0:
            tlwhs, ids, scores = zip(*objs)
        else:
            tlwhs, ids, scores = [], [], []
        tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

        return tlwhs, ids, scores


def evaluate(result_files, gt_files, data_type, result_folder=''):
    """
    result_files: format: frame_id, track_id, x, y, w, h, conf | list[str]
    gt_files: list[str[
    data_type: data format: "mot" | str
    result_folder: if result files is under a folder, then add to result prefix
    """
    assert len(result_files) == len(gt_files)

    accs = []

    for idx, result_f in enumerate(result_files):
        gt_f = gt_files[idx]

        evaluator = SeqEvaluator(seq_name=os.path.join(result_folder, result_f), gt_name=gt_f, data_type=data_type)
        accs.append(evaluator.eval_seq())

    # 得到总指标
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs,
        metrics=metrics,
        names=result_files,
        generate_overall=True
    )
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)









