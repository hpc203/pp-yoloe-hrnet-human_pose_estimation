import os
import copy
import time
import argparse
import cv2
import numpy as np
import onnxruntime
import math
from loguru import logger
from keypoint_postprocess import HRNetPostProcess
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

class PP_YOLOE():
    def __init__(self, prob_threshold=0.8):
        self.class_names = ['person']
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession('mot_ppyoloe_l_36e_pipeline.onnx', so)
        self.input_size = (
            self.session.get_inputs()[0].shape[3], self.session.get_inputs()[0].shape[2])  ###width, height
        self.confThreshold = prob_threshold
        self.scale_factor = np.array([1., 1.], dtype=np.float32)

    def preprocess(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.transpose(img, [2, 0, 1])
        return img

    def detect(self, srcimg):
        img = self.preprocess(srcimg)
        inputs = {'image': img[None, :, :, :], 'scale_factor': self.scale_factor[None, :]}
        ort_inputs = {i.name: inputs[i.name] for i in self.session.get_inputs() if i.name in inputs}
        output = self.session.run(None, ort_inputs)
        bbox, bbox_num = output
        keep_idx = (bbox[:, 1] > self.confThreshold) & (bbox[:, 0] > -1)
        bbox = bbox[keep_idx, :]
        ratioh = srcimg.shape[0] / self.input_size[1]
        ratiow = srcimg.shape[1] / self.input_size[0]
        dets = []
        for (clsid, score, xmin, ymin, xmax, ymax) in bbox:
            xmin = xmin * ratiow
            ymin = ymin * ratioh
            xmax = xmax * ratiow
            ymax = ymax * ratioh
            dets.append([xmin, ymin, xmax, ymax, score, int(clsid)])
        return np.array(dets, dtype=np.float32)

class KeyPointDetector():
    def __init__(self):
        self.class_names = ["keypoint"]
        so = onnxruntime.SessionOptions()
        so.log_severity_level = 3
        self.session = onnxruntime.InferenceSession('dark_hrnet_w32_256x192.onnx', so)
        self.inpWidth = self.session.get_inputs()[0].shape[3]
        self.inpHeight = self.session.get_inputs()[0].shape[2]
        self.imshape = np.array([[self.inpWidth, self.inpHeight]], dtype=np.float32)
        self.visual_thresh = 0.5
        self.use_dark = False
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0],
                       [0, 255, 0], \
                       [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
                       [85, 0, 255], \
                       [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    def resize_image(self, srcimg, keep_ratio=True):
        padh, padw, newh, neww = 0, 0, self.inpWidth, self.inpHeight
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_LINEAR)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT, value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_LINEAR)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_LINEAR)
        return img, newh, neww, padh, padw

    def predict(self, img, box_info):
        img, newh, neww, padh, padw = self.resize_image(img)
        img = np.transpose(img.astype(np.float32), [2, 0, 1])
        result = self.session.run(None, {'image': img[None, :, :, :]})

        results = {}
        center = np.round(self.imshape / 2.)
        scale = self.imshape / 200.
        keypoint_postprocess = HRNetPostProcess(use_dark=self.use_dark)
        kpts, scores = keypoint_postprocess(result[0], center, scale)
        for i in range(kpts.shape[1]):
            kpts[0, i, 0] = box_info['xmin'] + (kpts[0, i, 0] - padw) * (
                    box_info['xmax'] - box_info['xmin']) / neww
            kpts[0, i, 1] = box_info['ymin'] + (kpts[0, i, 1] - padh) * (
                    box_info['ymax'] - box_info['ymin']) / newh
        results['keypoint'] = kpts
        results['score'] = scores
        return results

    def visualize_pose(self, canvas, results):
        skeletons = np.array(results['keypoint'], dtype=np.float32)
        kpt_nums = 17
        if len(skeletons) > 0:
            kpt_nums = skeletons.shape[1]
        if kpt_nums == 17:  # plot coco keypoint
            EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
                     (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
                     (13, 15), (14, 16), (11, 12)]
        else:  # plot mpii keypoint
            EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (3, 6), (6, 7), (7, 8),
                     (8, 9), (10, 11), (11, 12), (13, 14), (14, 15), (8, 12),
                     (8, 13)]
        NUM_EDGES = len(EDGES)
        for i in range(kpt_nums):
            for j in range(len(skeletons)):
                if skeletons[j][i, 2] < self.visual_thresh:
                    continue

                color = self.colors[i]
                cv2.circle(canvas, tuple(skeletons[j][i, 0:2].astype('int32')), 10, color, thickness=-1)

        stickwidth = 2
        for i in range(NUM_EDGES):
            for j in range(len(skeletons)):
                edge = EDGES[i]
                if skeletons[j][edge[0], 2] < self.visual_thresh or skeletons[j][edge[1], 2] < self.visual_thresh:
                    continue

                cur_canvas = canvas.copy()
                X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
                Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                           (int(length / 2), stickwidth),
                                           int(angle), 0, 360, 1)
                color = self.colors[i]
                cv2.fillConvexPoly(cur_canvas, polygon, color)
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
        return canvas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='imgs/person.jpg', help="video path")
    parser.add_argument('--confThreshold', default=0.7, type=float, help='class confidence')
    args = parser.parse_args()

    net = PP_YOLOE(prob_threshold=args.confThreshold)
    kpt_predictor = KeyPointDetector()

    srcimg = cv2.imread(args.imgpath)
    dets = net.detect(srcimg)
    for i in range(dets.shape[0]):
        xmin, ymin, xmax, ymax = int(dets[i, 0]), int(dets[i, 1]), int(dets[i, 2]), int(dets[i, 3])
        results = kpt_predictor.predict(srcimg[ymin:ymax, xmin:xmax],
                                        {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
        cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
        srcimg = kpt_predictor.visualize_pose(srcimg, results)

    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()