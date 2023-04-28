import os
import torch
import cv2
import numpy as np
import tqdm
import argparse
import yaml
from time import gmtime, strftime
from timer import Timer
from PIL import Image
from bytetrack import ByteTrack
from bytetracker import BYTETracker
from visualize import plot_tracking
import dataloader


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append(os.getcwd())

from YOLOv7 import yolov7
from models.experimental import attempt_load
from utils.torch_utils import select_device, time_synchronized, TracedModel
print('Note: running yolov7 detector')


def set_basic_params(cfgs):
    global CATEGORY_DICT, DATASET_ROOT, CERTAIN_SEQS, IGNORE_SEQS
    CATEGORY_DICT = cfgs['CATEGORY_DICT']  # 类别
    DATASET_ROOT = cfgs['DATASET_ROOT']  # datasets root
    CERTAIN_SEQS = cfgs['CERTAIN_SEQS']  # 选择的序列
    IGNORE_SEQS = cfgs['IGNORE_SEQS']  # 忽略的序列


timer = Timer()
seq_fps = []  # 存储每一个seq运行的时间

def demo_image(opts, cfgs):
    set_basic_params(cfgs)

    TRACKER_DICT = {
        'bytetrack': ByteTrack,
    }

    if opts.save_videos:
        opts.save_images = True

    """step 1: Initialize the detector and tracker"""
    # with open(opts.config) as f:
    #     cfg = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = yolov7(opts, device)
    tracker = TRACKER_DICT[opts.tracker](opts)

    """step 2: load dataset and track"""
    # firstly, create seq list
    seqs = []
    DATA_ROOT = os.path.join(DATASET_ROOT, f'{opts.dataset}',f'{opts.image_path}')  # TODO
    if opts.data_format == 'yolo':
        with open(DATA_ROOT, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                elems = line.split('/')

                if elems[-2] not in seqs:
                    seqs.append(elems[-2])
    else:
        raise NotImplementedError
    seqs = sorted(seqs)
    seqs = [seq for seq in seqs if seq not in IGNORE_SEQS]


    if not None in CERTAIN_SEQS: seqs = CERTAIN_SEQS
    print(f'Seqs will be evalueated, total{len(seqs)}:')
    print(seqs)

    # secondly, for each seq, instantiate dataloader class and track
    # every time assign a different folder to store results
    folder_name = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    folder_name = folder_name[5:-3].replace('-', '_')
    folder_name = folder_name.replace(' ', '_')
    folder_name = folder_name.replace(':', '_')
    folder_name = opts.tracker + '_' + folder_name


    for seq in seqs:
        print(f'---------------tracking seq {seq}---------------\n')
        if opts.data_format == 'yolo':
            path = DATA_ROOT
        else:
            raise NotImplementedError("Data format must be yolo")
        loader = dataloader.TrackerLoader(DATASET_ROOT, path, opts.img_size[0], seq)
        data_loader = torch.utils.data.DataLoader(loader, batch_size=1)

        results = []  # store current seq results
        frame_id = 0

        if(frame_id == 0):
            tracker = TRACKER_DICT[opts.tracker](opts)

        pbar = tqdm.tqdm(desc=f"{seq}", ncols=80)  # 创建进度条对象
        for i, img in enumerate(data_loader):
            img = np.array(img).squeeze(0)
            pbar.update()
            timer.tic()  # start timing from this img

            # detect
            with torch.no_grad():
                out = detector.detect(img)
                # print(out)
                # out = torch.from_numpy(out)
            torch.cuda.empty_cache()

            # NOTE: yolo v7 origin out format: [xc, yc, w, h, conf, cls0_conf, cls1_conf, ..., clsn_conf]
            # if opts.det_output_format == 'yolo':
            #     cls_conf, cls_idx = torch.max(out[:, 5:], dim=1)
            #     # out[:, 4] *= cls_conf  # fuse object and cls conf
            #     out[:, 5] = cls_idx

            # results of track
            if not len(out):
                out = [[1e-6, 1e-6, 1e-6, 1e-6, 1e-6]]
                out = np.asarray(out)
            bbox = out[:, :5]
            # print(type(bbox))
            # print(bbox.shape)
            # print(bbox)
            tlwhs, ids, cls = tracker.track(bbox)

            # save results (.txt)
            results.append((frame_id + 1, ids, tlwhs, cls))
            timer.toc()  # end timing until this img

            # save results (images)
            if opts.save_images:
                image = plot_tracking(img, tlwhs, ids, frame_id=frame_id + 1, fps=1. / timer.diff)
                # plot_img(img, frame_id, [tlwhs, ids, 0], save_dir=os.path.join('results/images/', seq))

            save_dir = os.path.join('results/images/', seq)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=image)

            frame_id += 1

        seq_fps.append(i / timer.total_time)
        timer.clear()
        pbar.close()

        # thirdly, save results
        # every time assign a different name
        save_results(folder_name, seq, results)

        # finally, save videos
        if opts.save_images and opts.save_videos:
            save_videos(seq_names=seq)

    """step 3: evaluate results"""
    print(f'fps: {seq_fps}')
    print(f'average fps: {np.mean(seq_fps)}')


def demo_video(opts, cfgs):
    set_basic_params(cfgs)

    TRACKER_DICT = {
        'bytetrack': ByteTrack,
    }

    """初始化检测器和跟踪器"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector = yolov7(opts, device)
    tracker = TRACKER_DICT[opts.tracker](opts)

    """提取视频文件名"""
    video_file_name = os.path.basename(opts.video_path)
    video_file_name_split = os.path.splitext(video_file_name)[0]
    cap = cv2.VideoCapture(opts.video_path)
    # 保存视频结果
    if opts.save_videos:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(f'./results/videos/{video_file_name}', fourcc, fps, (width, height))

    """读取数据"""
    vid = cv2.VideoCapture(opts.video_path)
    flag, image = vid.read()

    results = []  # store current seq results

    """开始跟踪"""
    print(f'---------------tracking {video_file_name}---------------\n')
    # pbar = tqdm.tqdm(desc=f"{video_file_name}", ncols=80)  # 创建进度条对象
    frame_id = 0
    count = 0
    while flag:
        timer.tic()

        # detect
        with torch.no_grad():
            bbox = detector.detect(image)
            bbox = bbox[:, :5]
            # print(type(bbox))
            # print(bbox.shape)
        torch.cuda.empty_cache()

        # track
        tlwhs, ids, cls = tracker.track(bbox)

        # save results
        results.append((frame_id + 1, ids, tlwhs, cls))
        timer.toc()  # end timing until this img

        # plot
        img_plot = plot_tracking(
            image, tlwhs, ids, frame_id=frame_id + 1, fps=1. / timer.diff
        )

        # save video
        if opts.save_videos:
            out.write(img_plot)

        # load next image
        flag, image = vid.read()
        frame_id += 1
        count += 1
        if count == 50:
            print(f"frame_id: {frame_id}")
            count = 0

    print(f"frame number: {frame_id}")
    if opts.save_videos:
        out.release()
    cap.release()

    # save_results(video_file_name_split, video_file_name_split, results)
    # average_time = frame_id + 1 / timer.total_time
    # timer.clear()
    # pbar.close()
    # print('average time:', average_time)


def save_results(folder_name, seq_name, results, data_type='default'):
    assert len(results)
    if not data_type == 'default':
        raise NotImplementedError  # TODO

    if not os.path.exists(f'./results/GT_results/{folder_name}'):
        os.makedirs(f'./results/GT_results/{folder_name}')

    with open(os.path.join(f'./results/GT_results/{folder_name}/{seq_name}' + '.txt'), 'w') as f:
        for frame_id, targets_ids, tlwhs, clses in results:
            if data_type == 'default':
                for id, tlwh, cls in zip(targets_ids, tlwhs, clses):
                    f.write(f'{frame_id},{id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},-1,-1,-1,-1\n')

    f.close()

    return folder_name


def save_videos(seq_names):
    if not isinstance(seq_names, list):
        seq_names = [seq_names]

    for seq in seq_names:
        images_path = os.path.join('results/images/', seq)
        images_name = sorted(os.listdir(images_path))

        if not os.path.exists(f'results/videos'):
            os.makedirs(f'results/videos')

        to_video_path = os.path.join('results/videos/', seq + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        img0 = Image.open(os.path.join(images_path, images_name[0]))
        vw = cv2.VideoWriter(to_video_path, fourcc, 15, img0.size)  # TODO
        print('Save videos Done!!')
        for img in images_name:
            if img.endswith('.jpg'):
                frame = cv2.imread(os.path.join(images_path, img))
                vw.write(frame)


def plot_img(img, frame_id, results, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_ = np.ascontiguousarray(np.copy(img))
    tlwhs, ids = results[0], results[1]
    for tlwh, id in zip(tlwhs, ids):
        # convert tlwh to tlbr
        tlbr = tuple([int(tlwh[0]), int(tlwh[1]), int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])])
        # draw a rect
        cv2.rectangle(img_, tlbr[:2], tlbr[2:], get_color(id), thickness=3, )
        # note the id and cls
        text = f'{CATEGORY_DICT.get(0, "Unknown")}-{id}'
        cv2.putText(img_, text, (tlbr[0], tlbr[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                        color=(255, 164, 0), thickness=2)

    cv2.imwrite(filename=os.path.join(save_dir, f'{frame_id:05d}.jpg'), img=img_)


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def main(opts, cfgs):
    if opts.image:
        demo_image(opts, cfgs)
    if opts.video:
        demo_video(opts, cfgs)
    if (not opts.image) and (not opts.video):
        raise NotImplementedError

if __name__ =='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='MOT17_train', help='certain datasets root')  # TODO
    parser.add_argument('--image_path', type=str, default='train_list.txt', help='image path')
    parser.add_argument('--weights', type=str, default='./weights/best9.pt', help='model path')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='image sizes')
    parser.add_argument('--conf_thresh', type=float, default=0.1, help='filter tracks')
    parser.add_argument('--iou_thresh', type=float, default=0.55, help='IOU thresh to filter tracks')
    parser.add_argument('--agnostic_nms', type=bool, default=False, help='agnostic nms')
    parser.add_argument('--augment', type=bool, default=False, help='image augment')
    parser.add_argument('--trace_model', type=bool, default=False, help='traced model of YOLOv7')
    parser.add_argument('--classes', type=int, default=0, help='classes')
    parser.add_argument('--image', action='store_true', help='image')
    parser.add_argument('--video', action='store_true', help='video')
    parser.add_argument('--video_path', type=str, default='./videos/palace.mp4', help='video path')

    parser.add_argument('--track_thresh', type=float, default=0.5, help='thresh for track')
    parser.add_argument('--track_buffer', type=int, default=40, help='tracking buffer')
    parser.add_argument('--match_thresh', type=float, default=0.8, help='match thresh')
    parser.add_argument('--aspect_ratio_thresh', type=float, default=1.6, help='aspect ratio thresh')
    parser.add_argument('--min_box_area', type=int, default=10, help='min box area')
    parser.add_argument('--fps', type=int, default=30, help='fps')  # TODO
    parser.add_argument('--mot20', type=bool, default=False, help='mot20')  # TODO

    parser.add_argument('--nms_thresh', type=float, default=0.7, help='thresh for NMS')
    parser.add_argument('--save_images', action='store_true', help='save tracking results (image)')
    parser.add_argument('--save_videos', action='store_true', help='save tracking results (video)')
    parser.add_argument('--data_format', type=str, default='yolo', help='data format')
    parser.add_argument('--det_output_format', type=str, default='yolo', help='data format of output of detector')
    parser.add_argument('--tracker', type=str, default='bytetrack', help='sort,deepsort...')  # TODO

    opts = parser.parse_args()

    # NOTE: read path of datasets, sequences and TrackEval configs
    with open(f'./tracker/config_files/{opts.dataset}.yaml', 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    # with open(f'./tracker/config_files/config.yaml', 'r') as f:
    #     cfgs = yaml.load(f, Loader=yaml.FullLoader)
    main(opts, cfgs)
