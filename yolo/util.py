import cv2
import numpy as np
import tensorflow as tf


class YOLOBase:
    @staticmethod
    def check_model_version(version: str):
        version = version.lower()
        version_pos_val = ('v3', 'v3_tiny', 'v4', 'v4_tiny')
        if version not in version_pos_val:
            raise ValueError(f'version must be in the list: {version_pos_val}')
        return version

    @staticmethod
    def strides_and_anchors(version: str):

        version = YOLOBase.check_model_version(version)

        if version == 'v3':
            strides = [8, 16, 32]
            anchors = [[[10, 13], [16, 30], [33, 23]],
                       [[30, 61], [62, 45], [59, 119]],
                       [[116, 90], [156, 198], [373, 326]]]

        elif version == 'v4':
            strides = [8, 16, 32]
            anchors = [[[12, 16], [19, 36], [40, 28]],
                       [[36, 75], [76, 55], [72, 146]],
                       [[142, 110], [192, 243], [459, 401]]]

        elif version == 'v3_tiny':
            strides = [16, 32]
            anchors = [[[10, 14], [23, 27], [37, 58]],
                       [[81, 82], [135, 169], [344, 319]]]
            # anchors = [[[23, 27],  [37, 58],   [81,  82]], # this line can be uncommented for default coco weights

        elif version == 'v4_tiny':
            strides = [16, 32]
            anchors = [[[10, 14], [23, 27], [37, 58]],
                       [[81, 82], [135, 169], [344, 319]]]
            # anchors = [[[23, 27],  [37, 58],   [81,  82]], # this line can be uncommented for default coco weights

        strides = np.array(strides)
        anchors = (np.array(anchors).T / strides).T

        return strides, anchors

    @staticmethod
    def read_class_names(class_file_name):
        names = list()
        with open(class_file_name, 'r') as data:
            for name in data:
                names.append(name.strip('\n'))
        return names

    @staticmethod
    def bbox_iou(boxes1, boxes2):
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return 1.0 * inter_area / union_area

    @staticmethod
    def bbox_giou(boxes1, boxes2):
        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        boxes1 = tf.concat([tf.minimum(boxes1[..., :2], boxes1[..., 2:]),
                            tf.maximum(boxes1[..., :2], boxes1[..., 2:])], axis=-1)
        boxes2 = tf.concat([tf.minimum(boxes2[..., :2], boxes2[..., 2:]),
                            tf.maximum(boxes2[..., :2], boxes2[..., 2:])], axis=-1)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        # Calculate the iou value between the two bounding boxes
        iou = inter_area / union_area

        # Calculate the coordinates of the upper left corner and the lower right corner of the smallest closed convex surface
        enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # Calculate the area of the smallest closed convex surface C
        enclose_area = enclose[..., 0] * enclose[..., 1]

        # Calculate the GIoU value according to the GioU formula
        giou = iou - 1.0 * (enclose_area - union_area) / enclose_area

        return giou

    @staticmethod
    def bbox_ciou(boxes1, boxes2):  # testing (should be better than giou)
        boxes1_coor = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2_coor = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left = tf.maximum(boxes1_coor[..., 0], boxes2_coor[..., 0])
        up = tf.maximum(boxes1_coor[..., 1], boxes2_coor[..., 1])
        right = tf.maximum(boxes1_coor[..., 2], boxes2_coor[..., 2])
        down = tf.maximum(boxes1_coor[..., 3], boxes2_coor[..., 3])

        c = (right - left) * (right - left) + (up - down) * (up - down)
        iou = YOLOBase.bbox_iou(boxes1, boxes2)

        u = (boxes1[..., 0] - boxes2[..., 0]) * (boxes1[..., 0] - boxes2[..., 0]) + (boxes1[..., 1] - boxes2[..., 1]) * (boxes1[..., 1] - boxes2[..., 1])
        d = u / c

        ar_gt = boxes2[..., 2] / boxes2[..., 3]
        ar_pred = boxes1[..., 2] / boxes1[..., 3]

        ar_loss = 4 / (np.pi * np.pi) * (tf.atan(ar_gt) - tf.atan(ar_pred)) * (tf.atan(ar_gt) - tf.atan(ar_pred))
        alpha = ar_loss / (1 - iou + ar_loss + 0.000001)
        ciou_term = d + alpha * ar_loss

        return iou - ciou_term

    @staticmethod
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    @staticmethod
    def image_preprocess(image, target_size, gt_boxes=None):
        ih, iw = target_size
        h, w, _ = image.shape

        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        image_paded = image_paded / 255.

        if gt_boxes is None:
            return image_paded
        else:
            gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
            gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
            return image_paded, gt_boxes
