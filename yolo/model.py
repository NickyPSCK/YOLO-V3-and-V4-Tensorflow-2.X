import tensorflow as tf
from yolo.util import YOLOBase


class BatchNormalization(tf.keras.layers.BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class YOLOModel(YOLOBase):
    def __init__(self):
        pass

    def _layer_convolutional(self, input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
        if downsample:
            input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = tf.keras.layers.Conv2D(filters=filters_shape[-1],
                                      kernel_size=filters_shape[0],
                                      strides=strides,
                                      padding=padding,
                                      use_bias=(not bn),
                                      kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                      bias_initializer=tf.constant_initializer(0.))(input_layer)
        if bn:
            conv = BatchNormalization()(conv)
        if activate:
            if activate_type == "leaky":
                conv = tf.keras.layers.LeakyReLU(alpha=0.1)(conv)
            elif activate_type == "mish":
                conv = self.mish(conv)

        return conv

    def _layer_residual_block(self, input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
        short_cut = input_layer
        conv = self._layer_convolutional(input_layer=input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
        conv = self._layer_convolutional(input_layer=conv, filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type)
        residual_output = short_cut + conv
        return residual_output

    def _layer_upsample(self, input_layer):
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

    def _layer_route_group(self, input_layer, groups, group_id):
        convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
        return convs[group_id]

    def _layer_darknet53(self, input_data):
        input_data = self._layer_convolutional(input_data, (3, 3, 3, 32))
        input_data = self._layer_convolutional(input_data, (3, 3, 32, 64), downsample=True)

        for i in range(1):
            input_data = self._layer_residual_block(input_data, 64, 32, 64)

        input_data = self._layer_convolutional(input_data, (3, 3, 64, 128), downsample=True)

        for i in range(2):
            input_data = self._layer_residual_block(input_data, 128, 64, 128)

        input_data = self._layer_convolutional(input_data, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_data = self._layer_residual_block(input_data, 256, 128, 256)

        route_1 = input_data
        input_data = self._layer_convolutional(input_data, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_data = self._layer_residual_block(input_data, 512, 256, 512)

        route_2 = input_data
        input_data = self._layer_convolutional(input_data, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_data = self._layer_residual_block(input_data, 1024, 512, 1024)

        return route_1, route_2, input_data

    def _layer_cspdarknet53(self, input_data):
        input_data = self._layer_convolutional(input_data, (3, 3, 3, 32), activate_type="mish")
        input_data = self._layer_convolutional(input_data, (3, 3, 32, 64), downsample=True, activate_type="mish")

        route = input_data
        route = self._layer_convolutional(route, (1, 1, 64, 64), activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
        for i in range(1):
            input_data = self._layer_residual_block(input_data, 64, 32, 64, activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 64, 64), activate_type="mish")

        input_data = tf.concat([input_data, route], axis=-1)
        input_data = self._layer_convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
        input_data = self._layer_convolutional(input_data, (3, 3, 64, 128), downsample=True, activate_type="mish")
        route = input_data
        route = self._layer_convolutional(route, (1, 1, 128, 64), activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 128, 64), activate_type="mish")
        for i in range(2):
            input_data = self._layer_residual_block(input_data, 64, 64, 64, activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 64, 64), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self._layer_convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
        input_data = self._layer_convolutional(input_data, (3, 3, 128, 256), downsample=True, activate_type="mish")
        route = input_data
        route = self._layer_convolutional(route, (1, 1, 256, 128), activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 256, 128), activate_type="mish")
        for i in range(8):
            input_data = self._layer_residual_block(input_data, 128, 128, 128, activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 128, 128), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self._layer_convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
        route_1 = input_data
        input_data = self._layer_convolutional(input_data, (3, 3, 256, 512), downsample=True, activate_type="mish")
        route = input_data
        route = self._layer_convolutional(route, (1, 1, 512, 256), activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 512, 256), activate_type="mish")
        for i in range(8):
            input_data = self._layer_residual_block(input_data, 256, 256, 256, activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 256, 256), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self._layer_convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
        route_2 = input_data
        input_data = self._layer_convolutional(input_data, (3, 3, 512, 1024), downsample=True, activate_type="mish")
        route = input_data
        route = self._layer_convolutional(route, (1, 1, 1024, 512), activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 1024, 512), activate_type="mish")
        for i in range(4):
            input_data = self._layer_residual_block(input_data, 512, 512, 512, activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 512, 512), activate_type="mish")
        input_data = tf.concat([input_data, route], axis=-1)

        input_data = self._layer_convolutional(input_data, (1, 1, 1024, 1024), activate_type="mish")
        input_data = self._layer_convolutional(input_data, (1, 1, 1024, 512))
        input_data = self._layer_convolutional(input_data, (3, 3, 512, 1024))
        input_data = self._layer_convolutional(input_data, (1, 1, 1024, 512))

        max_pooling_1 = tf.keras.layers.MaxPool2D(pool_size=13, padding='SAME', strides=1)(input_data)
        max_pooling_2 = tf.keras.layers.MaxPool2D(pool_size=9, padding='SAME', strides=1)(input_data)
        max_pooling_3 = tf.keras.layers.MaxPool2D(pool_size=5, padding='SAME', strides=1)(input_data)
        input_data = tf.concat([max_pooling_1, max_pooling_2, max_pooling_3, input_data], axis=-1)

        input_data = self._layer_convolutional(input_data, (1, 1, 2048, 512))
        input_data = self._layer_convolutional(input_data, (3, 3, 512, 1024))
        input_data = self._layer_convolutional(input_data, (1, 1, 1024, 512))

        return route_1, route_2, input_data

    def _layer_darknet19_tiny(self, input_data):
        input_data = self._layer_convolutional(input_data, (3, 3, 3, 16))
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = self._layer_convolutional(input_data, (3, 3, 16, 32))
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = self._layer_convolutional(input_data, (3, 3, 32, 64))
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = self._layer_convolutional(input_data, (3, 3, 64, 128))
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = self._layer_convolutional(input_data, (3, 3, 128, 256))
        route_1 = input_data
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)
        input_data = self._layer_convolutional(input_data, (3, 3, 256, 512))
        input_data = tf.keras.layers.MaxPool2D(2, 1, 'same')(input_data)
        input_data = self._layer_convolutional(input_data, (3, 3, 512, 1024))

        return route_1, input_data

    def _layer_cspdarknet53_tiny(self, input_data):
        input_data = self._layer_convolutional(input_data, (3, 3, 3, 32), downsample=True)
        input_data = self._layer_convolutional(input_data, (3, 3, 32, 64), downsample=True)
        input_data = self._layer_convolutional(input_data, (3, 3, 64, 64))

        route = input_data
        input_data = self._layer_route_group(input_data, 2, 1)
        input_data = self._layer_convolutional(input_data, (3, 3, 32, 32))
        route_1 = input_data
        input_data = self._layer_convolutional(input_data, (3, 3, 32, 32))
        input_data = tf.concat([input_data, route_1], axis=-1)
        input_data = self._layer_convolutional(input_data, (1, 1, 32, 64))
        input_data = tf.concat([route, input_data], axis=-1)
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

        input_data = self._layer_convolutional(input_data, (3, 3, 64, 128))
        route = input_data
        input_data = self._layer_route_group(input_data, 2, 1)
        input_data = self._layer_convolutional(input_data, (3, 3, 64, 64))
        route_1 = input_data
        input_data = self._layer_convolutional(input_data, (3, 3, 64, 64))
        input_data = tf.concat([input_data, route_1], axis=-1)
        input_data = self._layer_convolutional(input_data, (1, 1, 64, 128))
        input_data = tf.concat([route, input_data], axis=-1)
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

        input_data = self._layer_convolutional(input_data, (3, 3, 128, 256))
        route = input_data
        input_data = self._layer_route_group(input_data, 2, 1)
        input_data = self._layer_convolutional(input_data, (3, 3, 128, 128))
        route_1 = input_data
        input_data = self._layer_convolutional(input_data, (3, 3, 128, 128))
        input_data = tf.concat([input_data, route_1], axis=-1)
        input_data = self._layer_convolutional(input_data, (1, 1, 128, 256))
        route_1 = input_data
        input_data = tf.concat([route, input_data], axis=-1)
        input_data = tf.keras.layers.MaxPool2D(2, 2, 'same')(input_data)

        input_data = self._layer_convolutional(input_data, (3, 3, 512, 512))

        return route_1, input_data

    def _layer_yolo_v3(self, input_layer, no_of_classes):
        # After the input layer enters the Darknet-53 network, we get three branches
        route_1, route_2, conv = self._layer_darknet53(input_layer)
        # See the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
        conv = self._layer_convolutional(conv, (1, 1, 1024, 512))
        conv = self._layer_convolutional(conv, (3, 3, 512, 1024))
        conv = self._layer_convolutional(conv, (1, 1, 1024, 512))
        conv = self._layer_convolutional(conv, (3, 3, 512, 1024))
        conv = self._layer_convolutional(conv, (1, 1, 1024, 512))
        conv_lobj_branch = self._layer_convolutional(conv, (3, 3, 512, 1024))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255]
        conv_lbbox = self._layer_convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (no_of_classes + 5)), activate=False, bn=False)

        conv = self._layer_convolutional(conv, (1, 1, 512, 256))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter
        conv = self._layer_upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)
        conv = self._layer_convolutional(conv, (1, 1, 768, 256))
        conv = self._layer_convolutional(conv, (3, 3, 256, 512))
        conv = self._layer_convolutional(conv, (1, 1, 512, 256))
        conv = self._layer_convolutional(conv, (3, 3, 256, 512))
        conv = self._layer_convolutional(conv, (1, 1, 512, 256))
        conv_mobj_branch = self._layer_convolutional(conv, (3, 3, 256, 512))

        # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
        conv_mbbox = self._layer_convolutional(conv_mobj_branch, (1, 1, 512, 3 * (no_of_classes + 5)), activate=False, bn=False)

        conv = self._layer_convolutional(conv, (1, 1, 256, 128))
        conv = self._layer_upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv = self._layer_convolutional(conv, (1, 1, 384, 128))
        conv = self._layer_convolutional(conv, (3, 3, 128, 256))
        conv = self._layer_convolutional(conv, (1, 1, 256, 128))
        conv = self._layer_convolutional(conv, (3, 3, 128, 256))
        conv = self._layer_convolutional(conv, (1, 1, 256, 128))
        conv_sobj_branch = self._layer_convolutional(conv, (3, 3, 128, 256))

        # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
        conv_sbbox = self._layer_convolutional(conv_sobj_branch, (1, 1, 256, 3 * (no_of_classes + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def _layer_yolo_v4(self, input_layer, no_of_classes):
        route_1, route_2, conv = self._layer_cspdarknet53(input_layer)

        route = conv
        conv = self._layer_convolutional(conv, (1, 1, 512, 256))
        conv = self._layer_upsample(conv)
        route_2 = self._layer_convolutional(route_2, (1, 1, 512, 256))
        conv = tf.concat([route_2, conv], axis=-1)

        conv = self._layer_convolutional(conv, (1, 1, 512, 256))
        conv = self._layer_convolutional(conv, (3, 3, 256, 512))
        conv = self._layer_convolutional(conv, (1, 1, 512, 256))
        conv = self._layer_convolutional(conv, (3, 3, 256, 512))
        conv = self._layer_convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = self._layer_convolutional(conv, (1, 1, 256, 128))
        conv = self._layer_upsample(conv)
        route_1 = self._layer_convolutional(route_1, (1, 1, 256, 128))
        conv = tf.concat([route_1, conv], axis=-1)

        conv = self._layer_convolutional(conv, (1, 1, 256, 128))
        conv = self._layer_convolutional(conv, (3, 3, 128, 256))
        conv = self._layer_convolutional(conv, (1, 1, 256, 128))
        conv = self._layer_convolutional(conv, (3, 3, 128, 256))
        conv = self._layer_convolutional(conv, (1, 1, 256, 128))

        route_1 = conv
        conv = self._layer_convolutional(conv, (3, 3, 128, 256))
        conv_sbbox = self._layer_convolutional(conv, (1, 1, 256, 3 * (no_of_classes + 5)), activate=False, bn=False)

        conv = self._layer_convolutional(route_1, (3, 3, 128, 256), downsample=True)
        conv = tf.concat([conv, route_2], axis=-1)

        conv = self._layer_convolutional(conv, (1, 1, 512, 256))
        conv = self._layer_convolutional(conv, (3, 3, 256, 512))
        conv = self._layer_convolutional(conv, (1, 1, 512, 256))
        conv = self._layer_convolutional(conv, (3, 3, 256, 512))
        conv = self._layer_convolutional(conv, (1, 1, 512, 256))

        route_2 = conv
        conv = self._layer_convolutional(conv, (3, 3, 256, 512))
        conv_mbbox = self._layer_convolutional(conv, (1, 1, 512, 3 * (no_of_classes + 5)), activate=False, bn=False)

        conv = self._layer_convolutional(route_2, (3, 3, 256, 512), downsample=True)
        conv = tf.concat([conv, route], axis=-1)

        conv = self._layer_convolutional(conv, (1, 1, 1024, 512))
        conv = self._layer_convolutional(conv, (3, 3, 512, 1024))
        conv = self._layer_convolutional(conv, (1, 1, 1024, 512))
        conv = self._layer_convolutional(conv, (3, 3, 512, 1024))
        conv = self._layer_convolutional(conv, (1, 1, 1024, 512))

        conv = self._layer_convolutional(conv, (3, 3, 512, 1024))
        conv_lbbox = self._layer_convolutional(conv, (1, 1, 1024, 3 * (no_of_classes + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def _layer_yolo_v3_tiny(self, input_layer, no_of_classes):
        # After the input layer enters the Darknet-53 network, we get three branches
        route_1, conv = self._layer_darknet19_tiny(input_layer)

        conv = self._layer_convolutional(conv, (1, 1, 1024, 256))
        conv_lobj_branch = self._layer_convolutional(conv, (3, 3, 256, 512))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 26, 26, 255]
        conv_lbbox = self._layer_convolutional(conv_lobj_branch, (1, 1, 512, 3 * (no_of_classes + 5)), activate=False, bn=False)

        conv = self._layer_convolutional(conv, (1, 1, 256, 128))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter
        conv = self._layer_upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv_mobj_branch = self._layer_convolutional(conv, (3, 3, 128, 256))
        # conv_mbbox is used to predict medium size objects, shape = [None, 13, 13, 255]
        conv_mbbox = self._layer_convolutional(conv_mobj_branch, (1, 1, 256, 3 * (no_of_classes + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]

    def _layer_yolo_v4_tiny(self, input_layer, no_of_classes):
        route_1, conv = self._layer_cspdarknet53_tiny(input_layer)

        conv = self._layer_convolutional(conv, (1, 1, 512, 256))

        conv_lobj_branch = self._layer_convolutional(conv, (3, 3, 256, 512))
        conv_lbbox = self._layer_convolutional(conv_lobj_branch, (1, 1, 512, 3 * (no_of_classes + 5)), activate=False, bn=False)

        conv = self._layer_convolutional(conv, (1, 1, 256, 128))
        conv = self._layer_upsample(conv)
        conv = tf.concat([conv, route_1], axis=-1)

        conv_mobj_branch = self._layer_convolutional(conv, (3, 3, 128, 256))
        conv_mbbox = self._layer_convolutional(conv_mobj_branch, (1, 1, 256, 3 * (no_of_classes + 5)), activate=False, bn=False)

        return [conv_mbbox, conv_lbbox]

    def decode(self, conv_output, no_of_classes, strides, anchors, i=0):
        '''
        where i = 0, 1 or 2 to correspond to the three grid scales
        '''
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + no_of_classes))

        # conv_raw_dxdy = conv_output[:, :, :, :, 0:2] # offset of center position
        # conv_raw_dwdh = conv_output[:, :, :, :, 2:4] # Prediction box length and width offset
        # conv_raw_conf = conv_output[:, :, :, :, 4:5] # confidence of the prediction box
        # conv_raw_prob = conv_output[:, :, :, :, 5: ] # category probability of the prediction box
        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(conv_output, (2, 2, 1, no_of_classes), axis=-1)

        # next need Draw the grid. Where output_size is equal to 13, 26 or 52
        # y = tf.range(output_size, dtype=tf.int32)
        # y = tf.expand_dims(y, -1)
        # y = tf.tile(y, [1, output_size])
        # x = tf.range(output_size,dtype=tf.int32)
        # x = tf.expand_dims(x, 0)
        # x = tf.tile(x, [output_size, 1])
        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        # xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        # y_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * strides[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i]) * strides[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def create_model(self,
                     no_of_classes: int,
                     version: str = 'v4',
                     input_size: int = 416,
                     channels: int = 3,
                     training: bool = False):

        version = self.check_model_version(version)
        strides, anchors = self.strides_and_anchors(version=version)

        input_layer = tf.keras.layers.Input([input_size, input_size, channels])

        if version == 'v3':
            conv_tensors = self._layer_yolo_v3(input_layer, no_of_classes)
        elif version == 'v4':
            conv_tensors = self._layer_yolo_v4(input_layer, no_of_classes)
        elif version == 'v3_tiny':
            conv_tensors = self._layer_yolo_v3_tiny(input_layer, no_of_classes)
        elif version == 'v4_tiny':
            conv_tensors = self._layer_yolo_v4_tiny(input_layer, no_of_classes)

        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            pred_tensor = self.decode(conv_tensor, no_of_classes, strides, anchors, i)
            if training:
                output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)

        yolo = tf.keras.Model(input_layer, output_tensors)

        return yolo
