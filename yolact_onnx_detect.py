import cv2
import os
import numpy as np
import copy
import math
import onnx
import onnxruntime as rt
from collections import OrderedDict
from itertools import product
import argparse

COLORS = ((244,  67,  54),
          (233,  30,  99),
          (156,  39, 176),
          (103,  58, 183),
          ( 63,  81, 181),
          ( 33, 150, 243),
          (  3, 169, 244),
          (  0, 188, 212),
          (  0, 150, 136),
          ( 76, 175,  80),
          (139, 195,  74),
          (205, 220,  57),
          (255, 235,  59),
          (255, 193,   7),
          (255, 152,   0),
          (255,  87,  34),
          (121,  85,  72),
          (158, 158, 158),
          ( 96, 125, 139))

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def sigmoid_proc(x):
    return 1.0 / (1.0 + np.exp(-x))

def point_form(boxes):
    return np.concatenate((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), axis=1)  # xmax, ymax

def decode(loc, priors, use_yolo_regressors:bool=False):
    if use_yolo_regressors:
        # Decoded boxes in center-size notation
        boxes = np.concatenate((
            loc[:, :2] + priors[:, :2],
            priors[:, 2:] * np.exp(loc[:, 2:])
        ), axis=1)

        boxes = point_form(boxes)
    else:
        variances = [0.1, 0.2]
        
        boxes = np.concatenate((
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
    
    return boxes

def intersect(box_a, box_b):
    n = box_a.shape[0]
    A = box_a.shape[1]
    B = box_b.shape[1]
    max_xy = np.minimum(np.broadcast_to(np.expand_dims(box_a[:, :, 2:], axis=2), (n, A, B, 2)), 
                    np.broadcast_to(np.expand_dims(box_b[:, :, 2:], axis=1), (n, A, B, 2)))
    min_xy = np.maximum(np.broadcast_to(np.expand_dims(box_a[:, :, :2], axis=2), (n, A, B, 2)),
                    np.broadcast_to(np.expand_dims(box_b[:, :, :2], axis=1), (n, A, B, 2)))
    return np.prod(np.clip(max_xy - min_xy, 0, np.max(max_xy - min_xy)), axis=3)

def jaccard(box_a, box_b):
    if len(box_a.shape) == 2:
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
    
    inter = intersect(box_a, box_b)
    area_a = np.broadcast_to(np.expand_dims((box_a[:, :, 2]-box_a[:, :, 0]) *
              (box_a[:, :, 3]-box_a[:, :, 1]), axis=2), inter.shape)  # [A,B]
    area_b = np.broadcast_to(np.expand_dims((box_b[:, :, 2]-box_b[:, :, 0]) *
              (box_b[:, :, 3]-box_b[:, :, 1]), axis=1), inter.shape)  # [A,B]
    union = area_a + area_b - inter

    out = inter / union
    return out

class YolactDetect(object):
    def __init__(self, num_cls, net_w, net_h, top_k, conf_thr, nms_thr):
        self.num_class = num_cls
        self.top_k = 200
        self.conf_thresh = conf_thr
        self.nms_thresh = nms_thr
        self.det_num = top_k
        self.net_size = [3, net_h, net_w]
        self.scales = [24, 48, 96, 192, 384]
        self.strides = [8, 16, 32, 64, 128]
        self.aspect_ratios = [1., 0.5, 2.]

    def fast_nms(self, boxes, masks, scores_src):
        scores = np.sort(scores_src, axis=1)[:,::-1]
        idx = np.argsort(-scores_src, axis=1)
        idx = idx[:, :self.top_k]
        scores = scores[:, :self.top_k]

        num_classes, num_dets = idx.shape
        
        boxes = np.reshape(boxes[np.reshape(idx, (-1)), :], (num_classes, num_dets, 4))
        masks = np.reshape(masks[np.reshape(idx, (-1)), :], (num_classes, num_dets, -1))
        
        iou = jaccard(boxes, boxes)
        iou = np.triu(iou, k=1)
        iou_max = np.max(iou, axis=1)
        keep = (iou_max <= self.nms_thresh)
        classes = np.broadcast_to(np.arange(num_classes)[:, None], keep.shape)
        classes = classes[keep]
        
        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        
        scorescp = copy.deepcopy(scores)
        scores = np.sort(scorescp, axis=0)[::-1]
        idx = np.argsort(-scorescp, axis=0)
        idx = idx[:self.det_num]
        scores = scores[:self.det_num]
        
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
        
        return boxes, masks, classes, scores

    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data):
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores = np.max(cur_scores, axis=0)
        
        keep = (conf_scores > self.conf_thresh)
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]
        
        if scores.shape[1] == 0:
            return None
        
        boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores)
        
        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}
    
    def make_prior(self, conv_h, conv_w, id):
        prior_data = []
        for j, i in product(range(conv_h), range(conv_w)):
            x = (i + 0.5) / conv_w
            y = (j + 0.5) / conv_h
            for ar in self.aspect_ratios:
                ar = math.sqrt(ar)
                w = self.scales[id] * ar / self.net_size[2]
                h = w
                prior_data.append([x, y, w, h])
        prior_data = np.array(prior_data, dtype=np.float32)
        
        return prior_data 

def sanitize_coordinates(_x1, _x2, img_size, padding=0, cast=True):
    _x1 = _x1 * img_size
    _x2 = _x2 * img_size
    if cast:
        _x1 = _x1.astype(np.int64)
        _x2 = _x2.astype(np.int64)
    x1 = np.minimum(_x1, _x2)
    x2 = np.maximum(_x1, _x2)
    x1 = np.clip(x1-padding, 0, np.max(x1-padding))
    x2 = np.clip(x2+padding, np.min(x2+padding), img_size)

    return x1, x2

def crop(masks, boxes):
    h, w, n = masks.shape
    x1, x2 = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, 1, cast=False)
    y1, y2 = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, 1, cast=False)
    rows = np.broadcast_to(np.reshape(np.arange(w, dtype=np.float32), (1, -1, 1)), (h, w, n))
    cols = np.broadcast_to(np.reshape(np.arange(h, dtype=np.float32), (-1, 1, 1)), (h, w, n))
    masks_left = rows >= np.reshape(x1, (1, 1, -1))
    masks_right = rows < np.reshape(x2, (1, 1, -1))
    masks_up = cols >= np.reshape(y1, (1, 1, -1))
    masks_down = cols < np.reshape(y2, (1, 1, -1))
    
    crop_mask = masks_left * masks_right * masks_up * masks_down
    
    return masks * crop_mask

def postprocess(det_output, w, h, crop_masks=True, score_threshold=.5):
    dets = det_output[0]
    dets = dets['detection']
    if dets is None:
        return None, None, None, None
    if score_threshold > 0:
        keep = dets['score'] > score_threshold
        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][keep]
        if dets['score'].shape[0] == 0:
            return None, None, None, None
    classes = dets['class']
    boxes = dets['box']
    scores = dets['score']
    masks = dets['mask']
    proto_data = dets['proto']
    masks = proto_data @ masks.T
    masks = sigmoid_proc(masks)
    if crop_masks:
        masks = crop(masks, boxes)
    #masks = np.transpose(masks, (2, 0, 1))
    srcshape = masks.shape
    masks = cv2.resize(masks, (w, h), interpolation=cv2.INTER_LINEAR)
    dstshape = masks.shape
    if(len(srcshape) != len(dstshape)):
        masks = np.expand_dims(masks, axis=-1)
    masks = np.transpose(masks, (2, 0, 1))
    # masks.gt_(0.5)
    masks = np.where(masks > 0.5, 1., 0.)
    
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, cast=False)
    boxes = boxes.astype(np.int64)
        
    return classes, scores, boxes, masks

def get_color(j, undo_transform=True):
    color_idx = (j * 5) % len(COLORS)
    color = COLORS[color_idx]
    if not undo_transform:
        color = (color[2], color[1], color[0])
    else:
        color = np.array([color[2], color[1], color[0]], dtype=np.float32) / 255.
    return color    

def prep_display(dets_out, img, top_k, score_thr, mask_alpha=0.45):
    img_ = img / 255.0
    h, w, _ = img.shape
    t = postprocess(dets_out, w, h, score_threshold=score_thr)
    
    idx = np.argsort(-t[1])[:top_k]
    masks = t[3][idx]
    classes, scores, boxes = [x[idx] for x in t[:3]]
    
    num_dets_to_consider = min(top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < score_thr:
            num_dets_to_consider = j
            break
    
    if num_dets_to_consider > 0:
        masks = masks[:num_dets_to_consider, :, :, None]
        colors = [get_color(j).reshape(1, 1, 1, 3) for j in range(num_dets_to_consider)]
        colors = np.concatenate(colors, axis=0)
        masks_color = np.repeat(masks, 3, axis=3) * colors * mask_alpha
        inv_alph_masks = masks * (-mask_alpha) + 1
        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = np.cumprod(inv_alph_masks[:(num_dets_to_consider - 1)], axis=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += np.sum(masks_color_cumul, axis=0)
        img_ = img_ * np.prod(inv_alph_masks, axis=0) + masks_color_summand
    
    img_array = img_ * 255
    img_array = img_array.astype(np.uint8)
    for j in reversed(range(num_dets_to_consider)):
        x1, y1, x2, y2 = boxes[j, :]
        color = get_color(j, undo_transform=False)
        score = scores[j]
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 1)
        
        _class = COCO_CLASSES[classes[j]]
        text_str = '%s: %.2f' % (_class, score)
        font_face = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.6
        font_thickness = 1
        text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
        
        text_pt = (x1, y1 - 3)
        text_color = [255, 255, 255]
        cv2.rectangle(img_array, (x1, y1), (x1 + text_w, y1 - text_h - 4), color, -1)
        cv2.putText(img_array, text_str, text_pt, font_face, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return img_array


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=str, default="", help="input images path")
    parser.add_argument("--image", type=str, default="", help="input image file")
    parser.add_argument("--onnx", type=str, default="", help="onnx model file")
    parser.add_argument("--dst", type=str, default="", help="detection results save path")
    parser.add_argument("--net_w", type=str, default="576", help="networks input width")
    parser.add_argument("--net_h", type=str, default="576", help="networks input height")
    parser.add_argument("--conf_thr", type=str, default="0.5", help="detect confidence thresh")
    parser.add_argument("--nms_thr", type=str, default="0.4", help="detection fast nms thresh")
    parser.add_argument("--top_k", type=str, default="15", help="max detection object number")
    args_param = parser.parse_args()
    return args_param   

def main(args):
    # your onnx path
    onnx_path=args.onnx
    img_path=args.images
    save_path=args.dst
    if not img_path:
        img_path=args.image
    if not img_path:
        print("ERROR: images path or image file must be seted at least one!")
        assert(0)
    if not onnx_path:
        print("ERROR: onnx dir need set!")
        assert(0)
        
    yolact = YolactDetect(81, int(args.net_w), int(args.net_h), int(args.top_k), float(args.conf_thr), float(args.nms_thr))

    img_files = []
    if os.path.isdir(img_path):
        imglists = os.listdir(img_path)
        for imgname in imglists:
            img_file = os.path.join(img_path, imgname)
            img_files.append(img_file)
    else:
        img_files.append(img_path)
    for imgdir in img_files:
        image = cv2.imread(imgdir)
        img = cv2.resize(image, (yolact.net_size[2], yolact.net_size[1]), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]
        img = np.ascontiguousarray(img).astype(np.float32)
        img = img.transpose((2, 0, 1))
        img /= 255.0

        onnx_model=onnx.load(onnx_path)
        ort_session = rt.InferenceSession(onnx_model.SerializeToString())
        ort_inputs={}
        ort_inputs[ort_session.get_inputs()[0].name]=np.expand_dims(img, axis=0)

        outputs = [x.name for x in ort_session.get_outputs()]
        ort_outs = ort_session.run(outputs, ort_inputs)
        out_dict=OrderedDict(zip(outputs, ort_outs))
        
        conf_data = out_dict['818']
        loc_data = out_dict['814']
        mask_data = out_dict['816']
        proto_data = out_dict['618']
        prior_data = None
        yolact.num_class = conf_data.shape[2]

        for id, stride in enumerate(yolact.strides):
            conv_h, conv_w = round(yolact.net_size[2] / float(stride) + 0.005), round(yolact.net_size[1] / float(stride) + 0.005)
            dataprior = yolact.make_prior(conv_h, conv_w, id)
            if id == 0:
                prior_data = copy.deepcopy(dataprior)
            else:
                prior_data = np.concatenate([prior_data, dataprior], axis=0)

        batch_size = loc_data.shape[0]
        num_priors = prior_data.shape[0]

        conf_preds = np.transpose(np.reshape(conf_data, (batch_size, num_priors, yolact.num_class)), (0, 2, 1))
        out = []
        for batch_idx in range(batch_size):
            decoded_boxes = decode(loc_data[batch_idx], prior_data)
            result = yolact.detect(batch_idx, conf_preds, decoded_boxes, mask_data)
            result['proto'] = proto_data[batch_idx]
            
            out.append({'detection': result})

        img_numpy = prep_display(out, image, yolact.det_num, yolact.conf_thresh)
        savename = "output_" + imgdir.split("/")[-1]
        save_dir = os.path.join(save_path, savename)
        cv2.imwrite(save_dir, img_numpy)

if __name__ == "__main__":
    args = parse_args()
    main(args)