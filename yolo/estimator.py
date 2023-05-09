import os
import shutil
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

from yolo.util import YOLOBase
from yolo.data import YOLODatasetGenerator
from yolo.model import YOLOModel


class YOLOEstimator(YOLOBase):
    def __init__(self,
                 version: str = 'v4',
                 input_size: int = 416,
                 channels: int = 3,
                 inference_using_pretrained: bool = False,
                 init_pretrained_weight: bool = True,
                 disable_gpu: bool = False,
                 pretrained_weight_dir: str = None):

        self.check_hardware()

        if disable_gpu:
            self.disable_gpu()

        version = self.check_model_version(version)

        # Input Attributes
        self.version = version
        self.input_size = input_size
        self.channels = channels
        self.inference_using_pretrained = inference_using_pretrained
        self.init_pretrained_weight = init_pretrained_weight
        self.pretrained_weight_dir = pretrained_weight_dir

        # Attributes to be calculated
        self.no_of_classes = None
        self.model = None
        self.model_for_prediction = None
        self.pretrained_classes = None

        # Constants
        self.yolo_iou_threshold = 0.5
        self.train_lr_init = 1e-4
        self.train_lr_end = 1e-6
        self.strides, self.anchors = self.strides_and_anchors(version=self.version)
        self.optimizer = tf.keras.optimizers.Adam()

        # IO
        self.train_model_name = 'Model'
        self.train_log_dir = 'train_log'
        self.train_checkpoints_dir = 'checkpoints'

        if os.path.exists(self.train_log_dir):
            shutil.rmtree(self.train_log_dir)

    @staticmethod
    def check_hardware():

        tensorflow_version = tf.__version__
        keras_versiob = tf.keras.__version__
        devices = device_lib.list_local_devices()
        gpus = tf.config.list_physical_devices('GPU')
        print('\n')
        print(f'Tensorflow version: {tensorflow_version}')
        print(f'Keras version: {keras_versiob}')
        print(f'Devices: {devices}')
        print(f'GPUs: {gpus}')
        print('\n')

        if len(gpus) > 0:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError:
                pass

    @staticmethod
    def disable_gpu():
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except Exception as inst:
            print(inst)
            pass

    @classmethod
    def load(cls,
             model_dir: str,
             disable_gpu: bool = False):
        save_weights_path = os.path.join(model_dir, 'model_weights')
        save_config_path = os.path.join(model_dir, 'model_config.ycf')
        with open(save_config_path, 'rb') as f:
            config_dict = pickle.load(f)

        cls = cls(version=config_dict['version'],
                  input_size=config_dict['input_size'],
                  channels=config_dict['channels'],
                  inference_using_pretrained=False,
                  init_pretrained_weight=False,
                  disable_gpu=disable_gpu)

        cls.no_of_classes = config_dict['no_of_classes']

        cls.model = YOLOModel().create_model(no_of_classes=config_dict['no_of_classes'],
                                             version=config_dict['version'],
                                             input_size=config_dict['input_size'],
                                             channels=config_dict['channels'],
                                             training=True)

        cls.load_weights(save_weights_path)
        cls.create_model_for_prediction()
        return cls

    @staticmethod
    def copy_model_weight(source_model, destination_model):
        for i, l in enumerate(source_model.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    destination_model.layers[i].set_weights(layer_weights)
                except ValueError:
                    # print("skipping", destination_model.layers[i].name)
                    pass
        return destination_model

    def load_weights(self,
                     weight_path):
        try:
            self.model.load_weights(weight_path)
            self.create_model_for_prediction()

        except ValueError:
            print("Shapes are incompatible, transfering Darknet weights")

    def save(self,
             model_dir):
        os.makedirs(model_dir, exist_ok=True, mode=777)
        save_weights_path = os.path.join(model_dir, 'model_weights')
        save_config_path = os.path.join(model_dir, 'model_config.ycf')

        self.model.save_weights(save_weights_path)

        with open(save_config_path, 'wb') as f:
            config_dict = {'version': self.version,
                           'input_size': self.input_size,
                           'channels': self.channels,
                           'no_of_classes': self.no_of_classes}
            pickle.dump(config_dict, f)

    def load_darknet_pretrained_model(self):

        tf.keras.backend.clear_session()  # used to reset layer names

        if self.pretrained_weight_dir is not None:
            folder_dir = self.pretrained_weight_dir
        else:
            folder_dir = os.getcwd() + '/pretrained_weight/'

        pre_trained_pretrained_classes_path = folder_dir + 'coco.names'

        if self.version == 'v3':
            pre_trained_yolo_darknet_weight_path = folder_dir + 'yolov3.weights'
            range1 = 75
            range2 = [58, 66, 74]
        elif self.version == 'v4':
            pre_trained_yolo_darknet_weight_path = folder_dir + 'yolov4.weights'
            range1 = 110
            range2 = [93, 101, 109]
        elif self.version == 'v3_tiny':
            pre_trained_yolo_darknet_weight_path = folder_dir + 'yolov3-tiny.weights'
            range1 = 13
            range2 = [9, 12]
        elif self.version == 'v4_tiny':
            pre_trained_yolo_darknet_weight_path = folder_dir + 'yolov4-tiny.weights'
            range1 = 21
            range2 = [17, 20]

        self.pretrained_classes = self.read_class_names(pre_trained_pretrained_classes_path)
        self.pretrained_no_of_classes = len(self.pretrained_classes)

        model = YOLOModel().create_model(no_of_classes=self.pretrained_no_of_classes,
                                         version=self.version,
                                         input_size=self.input_size,
                                         channels=self.channels,
                                         training=False)

        # --------------------------------------------------------------------------------------------
        # load Darknet original weights to TensorFlow model
        # --------------------------------------------------------------------------------------------
        with open(pre_trained_yolo_darknet_weight_path, 'rb') as wf:
            major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

            j = 0
            for i in range(range1):

                if i > 0:
                    conv_layer_name = f'conv2d_{i}'
                else:
                    conv_layer_name = 'conv2d'

                if j > 0:
                    bn_layer_name = f'batch_normalization_{j}'
                else:
                    bn_layer_name = 'batch_normalization'

                conv_layer = model.get_layer(conv_layer_name)
                filters = conv_layer.filters
                k_size = conv_layer.kernel_size[0]
                in_dim = conv_layer.input_shape[-1]

                if i not in range2:
                    # darknet weights: [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)

                    # tf weights: [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                    bn_layer = model.get_layer(bn_layer_name)
                    j += 1
                else:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, k_size, k_size)
                conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))

                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

                if i not in range2:
                    conv_layer.set_weights([conv_weights])
                    bn_layer.set_weights(bn_weights)
                else:
                    conv_layer.set_weights([conv_weights, conv_bias])

            assert len(wf.read()) == 0, 'failed to read all data'

        return model

    def _compute_loss(self,
                      pred,
                      conv,
                      label,
                      bboxes,
                      i=0):
        conv_shape = tf.shape(conv)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]
        input_size = self.strides[i] * output_size
        conv = tf.reshape(conv, (batch_size,
                                 output_size,
                                 output_size,
                                 3,
                                 5 + self.no_of_classes))

        conv_raw_conf = conv[:, :, :, :, 4:5]
        conv_raw_prob = conv[:, :, :, :, 5:]

        pred_xywh = pred[:, :, :, :, 0:4]
        pred_conf = pred[:, :, :, :, 4:5]

        label_xywh = label[:, :, :, :, 0:4]
        respond_bbox = label[:, :, :, :, 4:5]
        label_prob = label[:, :, :, :, 5:]

        giou = tf.expand_dims(self.bbox_giou(pred_xywh, label_xywh), axis=-1)
        input_size = tf.cast(input_size, tf.float32)

        bbox_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = respond_bbox * bbox_loss_scale * (1 - giou)

        iou = self.bbox_iou(pred_xywh[:, :, :, :, np.newaxis, :],
                            bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])
        # Find the value of IoU with the real box The largest prediction box
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)

        # If the largest iou is less than the threshold,
        # it is considered that the prediction box contains no objects, then the background box
        respond_bgd = (1.0 - respond_bbox) * tf.cast(max_iou < self.yolo_iou_threshold, tf.float32)

        conf_focal = tf.pow(respond_bbox - pred_conf, 2)

        # Calculate the loss of confidence
        # we hope that if the grid contains objects,
        # then the network output prediction box has a confidence of 1 and 0 when there is no object.

        conf_loss = conf_focal * (respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                         logits=conv_raw_conf)
                                  + respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=respond_bbox,
                                                                                          logits=conv_raw_conf))

        prob_loss = respond_bbox * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob,
                                                                           logits=conv_raw_prob)

        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

        return giou_loss, conf_loss, prob_loss

    def compute_loss(self,
                     target,
                     pred_result):
        if 'tiny' in self.version:
            grid = 2
        else:
            grid = 3

        giou_loss = 0
        conf_loss = 0
        prob_loss = 0

        for i in range(grid):
            conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
            loss_items = self._compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        return giou_loss, conf_loss, prob_loss

    def validate_loss(self,
                      image_data,
                      target,
                      training):
        pred_result = self.model(image_data, training=training)
        giou_loss, conf_loss, prob_loss = self.compute_loss(target, pred_result)
        total_loss = giou_loss + conf_loss + prob_loss
        return giou_loss, conf_loss, prob_loss, total_loss

    def adjust_weights(self,
                       image_data,
                       target):
        with tf.GradientTape() as tape:
            giou_loss, conf_loss, prob_loss, total_loss = self.validate_loss(image_data, target, training=True)
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            self.global_steps.assign_add(1)

            if self.global_steps < self.warmup_steps:
                lr = self.global_steps / self.warmup_steps * self.train_lr_init
            else:
                lr = self.train_lr_end + 0.5 * (self.train_lr_init - self.train_lr_end) * (
                    (1 + tf.cos((self.global_steps - self.warmup_steps) / (self.total_steps - self.warmup_steps) * np.pi)))

            self.optimizer.lr.assign(lr.numpy())

            # writing summary data
            with self._training_log_writer.as_default():
                tf.summary.scalar("lr", self.optimizer.lr, step=self.global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=self.global_steps)
                tf.summary.scalar("loss/giou_loss", giou_loss, step=self.global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=self.global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=self.global_steps)

            self._training_log_writer.flush()

        return self.global_steps.numpy(), self.optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    def validate_step(self,
                      val_data: YOLODatasetGenerator,
                      epoch: int):
        count = 0
        giou_val = 0
        conf_val = 0
        prob_val = 0
        total_val = 0

        for image_data, target in val_data:
            results = self.validate_loss(image_data, target, training=False)
            count += 1
            giou_val += results[0].numpy()
            conf_val += results[1].numpy()
            prob_val += results[2].numpy()
            total_val += results[3].numpy()

        with self._validation_log_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val / count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val / count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val / count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val / count, step=epoch)

        self._validation_log_writer.flush()

        return count, giou_val, conf_val, prob_val, total_val

    def create_model_for_prediction(self):
        self.model_for_prediction = YOLOModel().create_model(no_of_classes=self.no_of_classes,
                                                             version=self.version,
                                                             input_size=self.input_size,
                                                             channels=self.channels,
                                                             training=False)
        self.model_for_prediction = self.copy_model_weight(source_model=self.model,
                                                           destination_model=self.model_for_prediction)

    def fit(self,
            train_data: YOLODatasetGenerator = None,
            val_data: YOLODatasetGenerator = None,
            epocs: int = 5,
            warmup_epocs: int = 2,
            save_checkpoint: bool = True,
            save_best_only: bool = True):

        if self.inference_using_pretrained:
            self.model = self.load_darknet_pretrained_model()
            self.model_for_prediction = self.load_darknet_pretrained_model()
            self.no_of_classes = self.pretrained_no_of_classes
        else:
            self.no_of_classes = train_data.no_of_classes
            if self.model is None:
                if self.init_pretrained_weight:
                    darknet_model = self.load_darknet_pretrained_model()

                self.model = YOLOModel().create_model(no_of_classes=self.no_of_classes,
                                                      version=self.version,
                                                      input_size=self.input_size,
                                                      channels=self.channels,
                                                      training=True)

                if self.init_pretrained_weight:
                    self.model = self.copy_model_weight(source_model=darknet_model,
                                                        destination_model=self.model)
                    del darknet_model

            steps_per_epoch = len(train_data)
            self.global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
            self.warmup_steps = warmup_epocs * steps_per_epoch
            self.total_steps = epocs * steps_per_epoch

            self._training_log_writer = tf.summary.create_file_writer(self.train_log_dir)
            self._validation_log_writer = tf.summary.create_file_writer(self.train_log_dir)

            best_val_loss = 1000  # should be large at start

            for epoch in range(epocs):

                cur_step = 1
                for image_data, target in train_data:
                    global_steps, lr, giou_loss, conf_loss, prob_loss, total_loss = self.adjust_weights(image_data, target)
                    # cur_step = global_steps % steps_per_epoch
                    print(f'''epoch:{epoch:2.0f} step:{cur_step:5.0f}/{steps_per_epoch}, lr:{lr:.6f}, \
                    giou_loss:{giou_loss:7.2f}, conf_loss:{conf_loss:7.2f}, prob_loss:{prob_loss:7.2f}, total_loss:{total_loss:7.2f}''')
                    cur_step += 1

                if val_data is not None:
                    count, giou_val, conf_val, prob_val, total_val = self.validate_step(val_data=val_data, epoch=epoch)
                    print(f'''Validation epoch:{epoch:2.0f}, giou_val_loss:{giou_val/count:7.2f}, conf_val_loss:{conf_val/count:7.2f}, \
                    prob_val_loss:{prob_val/count:7.2f}, total_val_loss:{total_val/count:7.2f}''')

                    new_best_val_loss = total_val / count
                    if save_best_only and (new_best_val_loss < best_val_loss):
                        model_name = self.train_model_name + '_best'
                        save_best_path = os.path.join(self.train_checkpoints_dir, model_name)
                        self.model.save_weights(save_best_path)
                        best_val_loss = new_best_val_loss

                if save_checkpoint:
                    if val_data is not None:
                        model_name = self.train_model_name + f'_val_loss_{new_best_val_loss:7.2f}'
                    else:
                        model_name = self.train_model_name + f'_val_loss_{epoch}'
                    save_checkpoint_path = os.path.join(self.train_checkpoints_dir, model_name)
                    self.model.save_weights(save_checkpoint_path)

            self.create_model_for_prediction()

    def predict(self,
                x,
                **kwarg):
        return self.model_for_prediction.predict(x, **kwarg)
