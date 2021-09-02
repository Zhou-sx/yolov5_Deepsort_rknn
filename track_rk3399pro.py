from typing import List, Any

import cv2
import time
import random
import numpy as np
from rknn.api import RKNN

import sys
sys.path.insert(0, './yolov5')

# from yolov5.utils.google_utils import attempt_download#这个用不到
# from yolov5.models.experimental import attempt_load#这个用不到
from yolov5.utils.datasets import LoadImages, LoadStreams#LoadImages是加载现成的视频 LoadStreams是加载摄像头
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh#这些功能性函数可以直接写在代码里
from yolov5.utils.torch_utils import select_device, time_synchronized#select_device用不到
# from yolov5.utils.plots import plot_one_box#也属于功能性函数
from deep_sort_pytorch.utils.parser import get_config#这个有大用
"""下面这些绝对保留"""
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
# import torch
# import torch.backends.cudnn as cudnn

"""
yolov5-deepsort for rknn
"""

# RKNN Detector 目标检测
MASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
ANCHORS = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
CLASSES = ['person', 'vehicle']


class RKNNDetector:
    """
        RKNN Detector：
        <1> 要求 .rknn文件 inference输出格式为 三个特征层 不用拼接
        <2> img [H,W,C]
        <3> 后处理 _predict 得到的结果是 [Xmin, Ymin, Xmax, Ymax, confidence, class]
    """
    def __init__(self, model, wh, masks, anchors, names):
        self.wh = wh
        self._masks = masks
        self._anchors = anchors
        self.names = names
        if isinstance(model, str):
            model = load_rknn_model(model)
        self._rknn = model
        self.draw_box = False

    def _predict(self, img_src, _img, gain, conf_thres=0.4, iou_thres=0.45):
        src_h, src_w = img_src.shape[:2]
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        pred_onx = self._rknn.inference(inputs=[_img])
        print("inference time:\t", time.time() - t0)
        boxes, classes, scores = [], [], []
        for t in range(3):
            input0_data = sigmoid(pred_onx[t][0])
            input0_data = np.transpose(input0_data, (1, 2, 0, 3))
            grid_h, grid_w, channel_n, predict_n = input0_data.shape
            anchors: List[Any] = [self._anchors[i] for i in self._masks[t]]
            box_confidence = input0_data[..., 4]
            box_confidence = np.expand_dims(box_confidence, axis=-1)
            box_class_probs = input0_data[..., 5:]
            box_xy = input0_data[..., :2]
            box_wh = input0_data[..., 2:4]
            col = np.tile(np.arange(0, grid_w), grid_h).reshape(-1, grid_w)
            row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_w)
            col = col.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            row = row.reshape((grid_h, grid_w, 1, 1)).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            box_xy = box_xy * 2 - 0.5 + grid
            box_wh = (box_wh * 2) ** 2 * anchors
            box_xy /= (grid_w, grid_h)  # 计算原尺寸的中心
            box_wh /= self.wh  # 计算原尺寸的宽高
            box_xy -= (box_wh / 2.)  # 计算原尺寸的中心
            box = np.concatenate((box_xy, box_wh), axis=-1)
            res = filter_boxes(box, box_confidence, box_class_probs, conf_thres)
            boxes.append(res[0])
            classes.append(res[1])
            scores.append(res[2])
        boxes, classes, scores = np.concatenate(boxes), np.concatenate(classes), np.concatenate(scores)
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s, iou_thres)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
        if len(nboxes) < 1:
            return [], []
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        box_list = []
        for (x, y, w, h), score, cl in zip(boxes, scores, classes):
            x *= gain[0]
            y *= gain[1]
            w *= gain[0]
            h *= gain[1]
            x1 = max(0, np.floor(x))
            y1 = max(0, np.floor(y))
            x2 = min(src_w, np.floor(x + w + 0.5))
            y2 = min(src_h, np.floor(y + h + 0.5))
            box_list.append(np.array([[x1, y1, x2, y2, score, cl]]))
            if self.draw_box:
                plot_one_box((int(x1), int(y1), int(x2), int(y2)), img_src, label=self.names[cl], line_thickness = 1)
        return box_list

    def predict_resize(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理使用resize
        return: labels,boxes
        """
        _img = cv2.resize(img_src, self.wh)
        gain = img_src.shape[:2][::-1]
        return self._predict(img_src, _img, gain, conf_thres, iou_thres, )

    def predict(self, img_src, conf_thres=0.4, iou_thres=0.45):
        """
        预测一张图片，预处理保持宽高比
        return: labels,boxes
        """
        _img, gain = letterbox(img_src, self.wh)
        return self._predict(img_src, _img, gain, conf_thres, iou_thres)

    def close(self):
        self._rknn.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def get_max_scale(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1)
    return scale


def get_new_size(img, scale):
    return tuple(map(int, np.array(img.shape[:2][::-1]) * scale))


class AutoScale:
    def __init__(self, img, max_w, max_h):
        self._src_img = img
        self.scale = get_max_scale(img, max_w, max_h)
        self._new_size = get_new_size(img, self.scale)
        self.__new_img = None

    @property
    def size(self):
        return self._new_size

    @property
    def new_img(self):
        if self.__new_img is None:
            self.__new_img = cv2.resize(self._src_img, self._new_size)
        return self.__new_img


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def filter_boxes(boxes, box_confidences, box_class_probs, conf_thres):
    box_scores = box_confidences * box_class_probs  # 条件概率， 在该cell存在物体的概率的基础上是某个类别的概率
    box_classes = np.argmax(box_scores, axis=-1)  # 找出概率最大的类别索引
    box_class_scores = np.max(box_scores, axis=-1)  # 最大类别对应的概率值
    pos = np.where(box_class_scores >= conf_thres)  # 找出概率大于阈值的item
    # pos = box_class_scores >= OBJ_THRESH  # 找出概率大于阈值的item
    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]
    return boxes, classes, scores


def nms_boxes(boxes, scores, iou_thres):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    # 如果想要显示图片 打开
    # cv2.imshow('display', img)
    # if cv2.waitKey(10000) & 0xFF == ord('q'): exit()


def letterbox(img, new_wh=(416, 416), color=(114, 114, 114)):
    a = AutoScale(img, *new_wh)
    new_img = a.new_img
    h, w = new_img.shape[:2]
    new_img = cv2.copyMakeBorder(new_img, 0, new_wh[1] - h, 0, new_wh[0] - w, cv2.BORDER_CONSTANT, value=color)
    return new_img, (new_wh[0] / a.scale, new_wh[1] / a.scale)


def load_rknn_model(PATH):
    rknn = RKNN()
    print('--> Loading model')
    ret = rknn.load_rknn(PATH)
    if ret != 0:
        print('load rknn model failed')
        exit(ret)
    print('done')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn


def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    根据 id 添加固定颜色的简单函数

    有用！！！
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def detect(opt):
    out, source, dt_model, show_vid, save_vid, save_txt, imgsz = \
        opt.output, opt.source, opt.rknn_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.img_size

    # Load model
    '''加载模型这一块将完全被rknn替代'''
    dt_model = load_rknn_model(dt_model)
    stride = 32
    imgsz = check_img_size(imgsz, s=stride)
    names = ['person', 'vehicle']
    detector = RKNNDetector(dt_model, (imgsz, imgsz), MASKS, ANCHORS, CLASSES)

    # case: ckpt.t7
    # Load deepsort extractor model
    reid_model = './deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.rknn'
    extractor = load_rknn_model(reid_model)

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    # case: ckpt.t7
    # deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
    #                     max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    #                     nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    #                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
    #                     use_cuda=True)
    # case: .rknn
    deepsort = DeepSort(extractor,  # 传入RKNN对象
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    if os.path.exists(out):#判断输出路径是否存在
        pass
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Set Dataloader
    vid_path, vid_writer = None, None

    '''说白了就是读视频'''
    #加载现有视频用到的是这个
    dataset = LoadImages(source, img_size=imgsz)

    '''yysy，我觉得你这个开始计时的时间不太对'''
    t0 = time.time()

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        '''这里开始逐帧处理视频'''
        img = img.transpose((1,2,0)).astype('float32')

        # Detection Inference
        '''推理一张图片，这个用rknn替代'''
        t1 = time_synchronized()
        pred = detector.predict(img)
        t2 = time_synchronized()

        # Process detections
        '''后处理'''
        for i, det in enumerate(pred):  # detections per image
            '''处理每个目标'''
            p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[0:2]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[0:2], det[:, :4], im0.shape, Is_rknn=True).round()  # changed in this fun
                # 测试一下
                plot_one_box(det[0][:4], im0) # Test for edits

                # Print results
                # 由Torch换成Numpy
                for c in np.unique(det[:, -1]):
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                '''最关键的地方用deepsort了'''
                outputs = deepsort.update(xywhs, confs, clss, im0)

                # draw boxes for visualization
                '''画框，可用可不用'''
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        color = compute_color_for_id(id)
                        plot_one_box(bboxes, im0, label=label, color=color, line_thickness=2)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx, id, bbox_left,
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if show_vid:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_txt or save_vid:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rknn_model', type=str, default='yolov5/weights/yolov5s.rknn', help='model.rknn path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='video path')#要追踪的视频的路径
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    detect(args)
