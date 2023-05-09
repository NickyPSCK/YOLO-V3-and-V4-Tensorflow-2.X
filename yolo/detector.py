import random

import cv2
import colorsys

import numpy as np
import tensorflow as tf

from yolo.util import YOLOBase
from yolo.estimator import YOLOEstimator


class YOLOObjectDetector(YOLOBase):

    def __init__(self,
                 model: YOLOEstimator,
                 classes: list = None):

        self.model = model
        self.classes = classes
        self.input_size = self.model.model.layers[0]._init_input_shape[0]
        self.no_of_class = self.model.model.layers[-1]._saved_model_inputs_spec[2].shape[-1]

        if self.model.inference_using_pretrained:
            self.classes = self.model.pretrained_classes

        if self.classes is None:
            self.classes = list(range(self.no_of_class))
        else:
            if self.no_of_class != len(self.classes):
                raise ValueError('Number of classes mismatch with model output')

        _ = self.detect_image(np.zeros((100, 100, 3)))

    @staticmethod
    def draw_bbox(image,
                  bboxes,
                  classes,
                  show_label=False,
                  show_confidence=False,
                  Text_colors=(255, 255, 0),
                  rectangle_colors=None,
                  tracking=False):
        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        # print("hsv_tuples", hsv_tuples)
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = bbox[4]
            class_ind = int(bbox[5])

            if rectangle_colors is not None:
                bbox_color = rectangle_colors
            else:
                bbox_color = colors[class_ind]

            bbox_thick = int(0.6 * (image_h + image_w) / 1000)

            if bbox_thick < 1:
                bbox_thick = 1

            fontScale = 0.75 * bbox_thick
            (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

            # put object rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

            if show_label:
                # get text label
                score_str = " {:.2f}".format(score) if show_confidence else ""

                if tracking:
                    score_str = " " + str(score)

                try:
                    label = "{}".format(classes[class_ind]) + score_str
                except KeyError:
                    print("You received KeyError, this might be that you are trying to use yolo original weights")
                    print("while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")

                # get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                      fontScale, thickness=bbox_thick)
                # put filled text rectangle
                cv2.rectangle(image,
                              (x1, y1),
                              (x1 + text_width, y1 - text_height - baseline),
                              bbox_color,
                              thickness=cv2.FILLED)

                # put text above rectangle
                cv2.putText(image,
                            label,
                            (x1, y1 - 4),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale,
                            Text_colors,
                            bbox_thick,
                            lineType=cv2.LINE_AA)

        return image

    @staticmethod
    def bboxes_iou(boxes1,
                   boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

        return ious

    @staticmethod
    def nms(bboxes,
            iou_threshold,
            sigma=0.3,
            method='nms'):
        """
        :param bboxes: (xmin, ymin, xmax, ymax, score, class)

        Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
            https://github.com/bharatsingh430/soft-nms
        """
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]
            # Process 1: Determine whether the number of bounding boxes is greater than 0
            while len(cls_bboxes) > 0:
                # Process 2: Select the bounding box with the highest score according to socre order A
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                # Process 3: Calculate this bounding box A and
                # Remain all iou of the bounding box and remove those bounding boxes whose iou value is higher than the threshold
                iou = YOLOObjectDetector.bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                assert method in ['nms', 'soft-nms']

                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0

                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    @staticmethod
    def postprocess_boxes(pred_bbox,
                          original_image,
                          input_size,
                          score_threshold):
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # 1. (x, y, w, h) --> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                    pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
        # 2. (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
        org_h, org_w = original_image.shape[:2]
        resize_ratio = min(input_size / org_w, input_size / org_h)

        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # 3. clip some boxes those are out of range
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # 4. discard some invalid boxes
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # 5. discard boxes with low scores
        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)
        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    def detect_image(self,
                     image_rgb: str,
                     score_threshold: float = 0.3,
                     iou_threshold: float = 0.45,
                     show_label: bool = False,
                     show_confidence: bool = False,
                     rectangle_colors: tuple = None):

        image_data = self.image_preprocess(np.copy(image_rgb),
                                           [self.input_size, self.input_size])

        image_data = image_data[np.newaxis, ...].astype(np.float32)

        pred_bbox = self.model.model_for_prediction(image_data,
                                                    training=False)

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = self.postprocess_boxes(pred_bbox,
                                        image_rgb,
                                        self.input_size,
                                        score_threshold)
        bboxes = self.nms(bboxes,
                          iou_threshold,
                          method='nms')

        result_image = self.draw_bbox(image_rgb,
                                      bboxes,
                                      classes=self.classes,
                                      show_label=show_label,
                                      show_confidence=show_confidence,
                                      rectangle_colors=rectangle_colors)

        return result_image, bboxes
