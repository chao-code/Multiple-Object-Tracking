import torch
import numpy as np

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import time_synchronized, TracedModel
from utils.datasets import letterbox

class yolov7():
    def __init__(self, opts=None, device=''):
        self.opts = opts
        self.img_size = opts.img_size[0]
        self.augment = opts.augment
        self.conf_thresh = opts.conf_thresh
        self.iou_thresh = opts.iou_thresh
        self.classes = opts.classes
        self.agnostic_nms = opts.agnostic_nms
        self.trace_model = opts.trace_model
        self.device = device
        self.model = attempt_load(opts.weights, map_location=self.device)  # load FP32 model
        # self.model = self.ckpt['ema' if self.ckpt.get('ema') else 'model'].float().fuse().eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.img_size, s=self.stride)  # check img_size
        if self.trace_model:
            print(self.img_size)
            self.model = TracedModel(self.model, self.device, self.img_size)
        else:
            self.model.to(self.device)

        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()  # to FP16
        # Run once before inference
        if self.device.type != 'cpu':
            _ = self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                    next(self.model.parameters())))

    def process(self, image):
        """
        Convert the input image to a size that matches the model input
        """
        # Padded resize
        img = letterbox(image, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        # numpy to torch
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # normalization
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # add batch size as dimension[0]
        return img

    def detect(self, image):
        img = self.process(image)
        # Inference
        # t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh,
                                   classes=self.classes, agnostic=self.agnostic_nms)

        # t2 = time_synchronized()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                boxes = det.cpu().numpy()
            else:
                boxes = []

        return boxes

