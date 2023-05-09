import os
import random

import cv2
import numpy as np
import tensorflow as tf

from yolo.util import YOLOBase


class YOLODatasetGenerator(YOLOBase):

    def __init__(self,
                 anotation_path: str,
                 version: str = 'v4',
                 input_sizes: int = 416,
                 batch_size: int = 4,
                 data_aug: bool = False,
                 load_images_to_ram: bool = True,
                 anchor_per_scale: int = 3,
                 max_bbox_per_scale: int = 100):

        self.annot_path = anotation_path

        self.input_sizes = input_sizes
        self.batch_size = batch_size
        self.data_aug = data_aug
        self.load_images_to_ram = load_images_to_ram

        self.anchor_per_scale = anchor_per_scale
        self.max_bbox_per_scale = max_bbox_per_scale

        self.strides, self.anchors = self.strides_and_anchors(version=version)

        if 'tiny' in version:
            self.train_yolo_tiny = True
        else:
            self.train_yolo_tiny = False

        self.annotations, self.classes = self.load_annotations()

        self.no_of_classes = len(self.classes)
        self.no_of_samples = len(self.annotations)

        self.num_batchs = int(np.ceil(self.no_of_samples / self.batch_size))
        self.batch_count = 0

    def load_annotations(self):
        classes = set()
        final_annotations = list()
        with open(self.annot_path, 'r') as f:
            txt = f.read().splitlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]

        np.random.shuffle(annotations)

        for annotation in annotations:
            # fully parse annotations
            line = annotation.split()
            image_path = ''
            index = 1
            for i, one_line in enumerate(line):
                if not one_line.replace(',', '').isnumeric():
                    if image_path != '':
                        image_path += ' '
                    image_path += one_line
                else:
                    index = i
                    break
            if not os.path.exists(image_path):
                raise KeyError(f'{image_path} does not exist ... ')

            if self.load_images_to_ram:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ADD
            else:
                image = None

            # bboxes = line[index:]
            bboxes = list(map(int, line[index:][0].split(',')))
            final_annotations.append([image_path, bboxes, image])
            classes.update(bboxes[4::5])

        classes = sorted(classes)

        return final_annotations, classes

    def delete_bad_annotation(self,
                              bad_annotation):
        print(f'Deleting {bad_annotation} annotation line')
        # bad_image_path = bad_annotation[0]
        bad_image_name = bad_annotation[0].split('/')[-1]  # can be used to delete bad image
        # bad_xml_path = bad_annotation[0][:-3]+'xml'  # can be used to delete bad xml file

        # remove bad annotation line from annotation file
        with open(self.annot_path, "r+") as f:
            d = f.readlines()
            f.seek(0)
            for i in d:
                if bad_image_name not in i:
                    f.write(i)
            f.truncate()

    def random_horizontal_flip(self,
                               image,
                               bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self,
                    image,
                    bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0),
                                       np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0),
                                       np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self,
                         annotation,
                         mAP: bool = False):
        if self.load_images_to_ram:
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # ADD

        # bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])
        bboxes = np.array([annotation[1]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        if mAP:
            return image, bboxes

        image, bboxes = self.image_preprocess(np.copy(image),
                                              [self.input_sizes, self.input_sizes],
                                              np.copy(bboxes))
        return image, bboxes

    def preprocess_true_boxes(self,
                              bboxes):
        output_levels = len(self.strides)

        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.no_of_classes)) for i in range(output_levels)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(output_levels)]
        bbox_count = np.zeros((output_levels,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            onehot = np.zeros(self.no_of_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.no_of_classes, 1.0 / self.no_of_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            for i in range(output_levels):  # range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        if self.train_yolo_tiny:
            label_mbbox, label_lbbox = label
            mbboxes, lbboxes = bboxes_xywh
            return label_mbbox, label_lbbox, mbboxes, lbboxes

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice([self.input_sizes])
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size,
                                    self.train_input_size,
                                    self.train_input_size,
                                    3), dtype=np.float32)

            if self.train_yolo_tiny:
                batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[0],
                                              self.train_output_sizes[0],
                                              self.anchor_per_scale, 5 + self.no_of_classes), dtype=np.float32)
                batch_label_lbbox = np.zeros((self.batch_size,
                                              self.train_output_sizes[1],
                                              self.train_output_sizes[1],
                                              self.anchor_per_scale, 5 + self.no_of_classes), dtype=np.float32)
            else:
                batch_label_sbbox = np.zeros((self.batch_size,
                                              self.train_output_sizes[0],
                                              self.train_output_sizes[0],
                                              self.anchor_per_scale, 5 + self.no_of_classes), dtype=np.float32)
                batch_label_mbbox = np.zeros((self.batch_size,
                                              self.train_output_sizes[1],
                                              self.train_output_sizes[1],
                                              self.anchor_per_scale, 5 + self.no_of_classes), dtype=np.float32)
                batch_label_lbbox = np.zeros((self.batch_size,
                                              self.train_output_sizes[2],
                                              self.train_output_sizes[2],
                                              self.anchor_per_scale, 5 + self.no_of_classes), dtype=np.float32)

                batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)

            exceptions = False
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num

                    if index >= self.no_of_samples:
                        index -= self.no_of_samples

                    annotation = self.annotations[index]
                    image, bboxes = self.parse_annotation(annotation)
                    try:
                        if self.train_yolo_tiny:
                            label_mbbox, label_lbbox, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                        else:
                            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                    except IndexError:
                        exceptions = True
                        self.delete_bad_annotation(annotation)
                        print("IndexError, something wrong with", annotation[0], "removed this line from annotation file")

                    batch_image[num, :, :, :] = image
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes

                    if not self.train_yolo_tiny:
                        batch_label_sbbox[num, :, :, :, :] = label_sbbox
                        batch_sbboxes[num, :, :] = sbboxes

                    num += 1

                if exceptions:
                    print('\n')
                    raise Exception("There were problems with dataset, I fixed them, now restart the training process.")

                self.batch_count += 1

                if not self.train_yolo_tiny:
                    batch_smaller_target = batch_label_sbbox, batch_sbboxes

                batch_medium_target = batch_label_mbbox, batch_mbboxes
                batch_larger_target = batch_label_lbbox, batch_lbboxes

                if self.train_yolo_tiny:
                    return batch_image, (batch_medium_target, batch_larger_target)
                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def __len__(self):
        return self.num_batchs
