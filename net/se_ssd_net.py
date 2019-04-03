# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
_USE_FUSED_BN = True


# vgg_16/conv2/conv2_1/biases
# vgg_16/conv4/conv4_3/biases
# vgg_16/conv1/conv1_1/biases
# vgg_16/fc6/weights
# vgg_16/conv3/conv3_2/biases
# vgg_16/conv5/conv5_3/biases
# vgg_16/conv3/conv3_1/weights
# vgg_16/conv4/conv4_2/weights
# vgg_16/conv1/conv1_1/weights
# vgg_16/conv5/conv5_3/weights
# vgg_16/conv4/conv4_1/weights
# vgg_16/conv3/conv3_3/weights
# vgg_16/conv5/conv5_2/biases
# vgg_16/conv3/conv3_2/weights
# vgg_16/conv4/conv4_2/biases
# vgg_16/conv5/conv5_2/weights
# vgg_16/conv3/conv3_1/biases
# vgg_16/conv2/conv2_2/weights
# vgg_16/fc7/weights
# vgg_16/conv5/conv5_1/biases
# vgg_16/conv1/conv1_2/biases
# vgg_16/conv2/conv2_2/biases
# vgg_16/conv4/conv4_1/biases
# vgg_16/fc7/biases
# vgg_16/fc6/biases
# vgg_16/conv4/conv4_3/weights
# vgg_16/conv2/conv2_1/weights
# vgg_16/conv5/conv5_1/weights
# vgg_16/conv3/conv3_3/biases
# vgg_16/conv1/conv1_2/weights

class ReLuLayer(tf.layers.Layer):
    def __init__(self, name, **kwargs):
        super(ReLuLayer, self).__init__(name=name, **kwargs)
        self._name = name

    def build(self, input_shape):
        self._relu = lambda x: tf.nn.relu(x, name=self._name)
        self.built = True

    def call(self, inputs):
        return self._relu(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


def forward_module(m, inputs, training=False):
    if isinstance(m, tf.layers.BatchNormalization) or isinstance(m, tf.layers.Dropout):
        return m.apply(inputs, training=training)
    return m.apply(inputs)


class SE_SSD300_NET(object):
    def __init__(self, data_format='channels_first', backbone_batch_normal=False, additional_batch_normal=False):
        super(SE_SSD300_NET, self).__init__()
        self._data_format = data_format
        self._bn_axis = -1 if data_format == 'channels_last' else 1
        # initializer = tf.glorot_uniform_initializer  glorot_normal_initializer
        self._conv_initializer = tf.glorot_uniform_initializer
        self._conv_bn_initializer = tf.glorot_uniform_initializer  # lambda : tf.truncated_normal_initializer(mean=0.0, stddev=0.005)
        # VGG layers
        self._conv1_block = self.conv_block(2, 64, 3, (1, 1),
                                            'conv1') if not backbone_batch_normal else self.conv_bn_block(2, 64, 3,
                                                                                                          (1, 1),
                                                                                                          'conv1_bn')
        self._pool1 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool1')
        self._conv2_block = self.conv_block(2, 128, 3, (1, 1),
                                            'conv2') if not backbone_batch_normal else self.conv_bn_block(2, 128, 3,
                                                                                                          (1, 1),
                                                                                                          'conv2_bn')
        self._pool2 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool2')
        self._conv3_block = self.conv_block(3, 256, 3, (1, 1),
                                            'conv3') if not backbone_batch_normal else self.conv_bn_block(3, 256, 3,
                                                                                                          (1, 1),
                                                                                                          'conv3_bn')
        self._pool3 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool3')
        self._conv4_block = self.conv_block(3, 512, 3, (1, 1),
                                            'conv4') if not backbone_batch_normal else self.conv_bn_block(3, 512, 3,
                                                                                                          (1, 1),
                                                                                                          'conv4_bn')
        self._pool4 = tf.layers.MaxPooling2D(2, 2, padding='same', data_format=self._data_format, name='pool4')
        self._conv5_block = self.conv_block(3, 512, 3, (1, 1),
                                            'conv5') if not backbone_batch_normal else self.conv_bn_block(3, 512, 3,
                                                                                                          (1, 1),
                                                                                                          'conv5_bn')
        self._pool5 = tf.layers.MaxPooling2D(3, 1, padding='same', data_format=self._data_format, name='pool5')
        self._conv6 = [tf.layers.Conv2D(filters=1024, kernel_size=3, strides=1, padding='same', dilation_rate=6,
                                        data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='fc6', _scope='fc6',
                                        _reuse=None)] if not additional_batch_normal else self.conv_bn_block(1, 1024, 3,
                                                                                                             (1, 1),
                                                                                                             'fc6_bn')
        self._conv7 = [tf.layers.Conv2D(filters=1024, kernel_size=1, strides=1, padding='same',
                                        data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                                        kernel_initializer=self._conv_initializer(),
                                        bias_initializer=tf.zeros_initializer(),
                                        name='fc7', _scope='fc7',
                                        _reuse=None)] if not additional_batch_normal else self.conv_bn_block(1, 1024, 1,
                                                                                                             (1, 1),
                                                                                                             'fc7_bn')
        # SSD layers
        with tf.variable_scope('additional_layers') as scope:
            self._conv8_block = self.ssd_conv_block(256, 2,
                                                    'conv8') if not additional_batch_normal else self.ssd_conv_bn_block(
                256, 2, 'conv8_bn')
            self._conv9_block = self.ssd_conv_block(128, 2,
                                                    'conv9') if not additional_batch_normal else self.ssd_conv_bn_block(
                128, 2, 'conv9_bn')
            self._conv10_block = self.ssd_conv_block(128, 1, 'conv10',
                                                     padding='valid') if not additional_batch_normal else self.ssd_conv_bn_block(
                128, 1, 'conv10_bn', padding='valid')
            self._conv11_block = self.ssd_conv_block(128, 1, 'conv11',
                                                     padding='valid') if not additional_batch_normal else self.ssd_conv_bn_block(
                128, 1, 'conv11_bn', padding='valid')

    def l2_normalize(self, x, name):
        with tf.name_scope(name, "l2_normalize", [x]) as name:
            axis = -1 if self._data_format == 'channels_last' else 1
            square_sum = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
            x_inv_norm = tf.rsqrt(tf.maximum(square_sum, 1e-10))
            return tf.multiply(x, x_inv_norm, name=name)

    def forward(self, inputs, training=False):
        # inputs should in BGR
        feature_layers = []
        # forward vgg layers
        for conv in self._conv1_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool1.apply(inputs)

        inputs = self.squeeze_excitation_layer(inputs,layer_name='se_conv1')

        for conv in self._conv2_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool2.apply(inputs)

        inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv2')

        for conv in self._conv3_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool3.apply(inputs)

        inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv3')

        for conv in self._conv4_block:
            inputs = forward_module(conv, inputs, training=training)

        # conv4_3
        with tf.variable_scope('conv4_3_scale') as scope:
            weight_scale = tf.Variable([20.] * 512, trainable=training, name='weights')
            if self._data_format == 'channels_last':
                weight_scale = tf.reshape(weight_scale, [1, 1, 1, -1], name='reshape')
            else:
                weight_scale = tf.reshape(weight_scale, [1, -1, 1, 1], name='reshape')

            feature_layers.append(tf.multiply(weight_scale, self.l2_normalize(inputs, name='norm'), name='rescale'))

            inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv4')

        inputs = self._pool4.apply(inputs)

        for conv in self._conv5_block:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self._pool5.apply(inputs)

        inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv5')

        # forward fc layers
        for conv in self._conv6:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv6')
        for conv in self._conv7:
            inputs = forward_module(conv, inputs, training=training)
        inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv7')
        # fc7
        feature_layers.append(inputs)

        # forward ssd layers
        for layer in self._conv8_block:
            inputs = forward_module(layer, inputs, training=training)
        inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv8')

        # conv8
        feature_layers.append(inputs)

        for layer in self._conv9_block:
            inputs = forward_module(layer, inputs, training=training)
        inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv9')
        # conv9
        feature_layers.append(inputs)

        for layer in self._conv10_block:
            inputs = forward_module(layer, inputs, training=training)
        inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv10')

        # conv10
        feature_layers.append(inputs)

        for layer in self._conv11_block:
            inputs = forward_module(layer, inputs, training=training)
        inputs = self.squeeze_excitation_layer(inputs, layer_name='se_conv11')

        # conv11
        feature_layers.append(inputs)

        return feature_layers

    def global_average_pooling(self,inputs):
        if self._data_format == 'channels_first':
            inputs = tf.reduce_mean(inputs, axis=[2, 3])
        else:
            inputs = tf.reduce_mean(inputs, axis=[1, 2])
        return inputs

    def squeeze_excitation_layer(self, inputs, layer_name,ratio=4):
        with tf.name_scope(layer_name):
            if self._data_format == 'channels_first':
                out_dim = inputs.shape.as_list()[1]
            else:
                out_dim = inputs.shape.as_list()[3]
            squeeze = self.global_average_pooling(inputs)
            excitation = tf.layers.dense(squeeze, units=out_dim / ratio, name=layer_name + '_fully_connected1')
            excitation = tf.nn.relu(excitation)
            excitation = tf.layers.dense(excitation, units=out_dim, name=layer_name + '_fully_connected2')
            excitation = tf.nn.sigmoid(excitation/0.1)
            # excitation = tf.nn.sigmoid(excitation)
            # excitation = tf.nn.softmax(excitation)
            tf.summary.histogram(name="se_activation/"+layer_name,values=excitation[0])
            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])
            scale = inputs * excitation
            return scale

    def conv_block(self, num_blocks, filters, kernel_size, strides, name, reuse=None):
        with tf.variable_scope(name):
            conv_blocks = []
            for ind in range(1, num_blocks + 1):
                conv_blocks.append(
                    tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                     data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                                     kernel_initializer=self._conv_initializer(),
                                     bias_initializer=tf.zeros_initializer(),
                                     name='{}_{}'.format(name, ind), _scope='{}_{}'.format(name, ind), _reuse=None)
                )
            return conv_blocks

    def conv_bn_block(self, num_blocks, filters, kernel_size, strides, name, reuse=None):
        with tf.variable_scope(name):
            conv_blocks = []
            for ind in range(1, num_blocks + 1):
                conv_blocks.append(
                    tf.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                     data_format=self._data_format, activation=None, use_bias=False,
                                     kernel_initializer=self._conv_bn_initializer(),
                                     bias_initializer=None,
                                     name='{}_{}'.format(name, ind), _scope='{}_{}'.format(name, ind), _reuse=None)
                )
                conv_blocks.append(
                    tf.layers.BatchNormalization(axis=self._bn_axis, name='{}_{}'.format(name, ind),
                                                 _scope='{}_{}'.format(name, ind), _reuse=None)
                )
                conv_blocks.append(
                    ReLuLayer('{}_{}'.format(name, ind), _scope='{}_{}'.format(name, ind), _reuse=None)
                )
            return conv_blocks

    def ssd_conv_block(self, filters, strides, name, padding='same', reuse=None):
        with tf.variable_scope(name):
            conv_blocks = []
            conv_blocks.append(
                tf.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding=padding,
                                 data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=self._conv_initializer(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='{}_1'.format(name), _scope='{}_1'.format(name), _reuse=None)
            )
            conv_blocks.append(
                tf.layers.Conv2D(filters=filters * 2, kernel_size=3, strides=strides, padding=padding,
                                 data_format=self._data_format, activation=tf.nn.relu, use_bias=True,
                                 kernel_initializer=self._conv_initializer(),
                                 bias_initializer=tf.zeros_initializer(),
                                 name='{}_2'.format(name), _scope='{}_2'.format(name), _reuse=None)
            )
            return conv_blocks

    def ssd_conv_bn_block(self, filters, strides, name, padding='same', reuse=None):
        with tf.variable_scope(name):
            conv_bn_blocks = []
            conv_bn_blocks.append(
                tf.layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding=padding,
                                 data_format=self._data_format, activation=None, use_bias=False,
                                 kernel_initializer=self._conv_bn_initializer(),
                                 bias_initializer=None,
                                 name='{}_1'.format(name), _scope='{}_1'.format(name), _reuse=None)
            )
            conv_bn_blocks.append(
                tf.layers.BatchNormalization(axis=self._bn_axis, name='{}_bn1'.format(name),
                                             _scope='{}_bn1'.format(name), _reuse=None)
            )
            conv_bn_blocks.append(
                ReLuLayer('{}_relu1'.format(name), _scope='{}_relu1'.format(name), _reuse=None)
            )
            conv_bn_blocks.append(
                tf.layers.Conv2D(filters=filters * 2, kernel_size=3, strides=strides, padding=padding,
                                 data_format=self._data_format, activation=None, use_bias=False,
                                 kernel_initializer=self._conv_bn_initializer(),
                                 bias_initializer=None,
                                 name='{}_2'.format(name), _scope='{}_2'.format(name), _reuse=None)
            )
            conv_bn_blocks.append(
                tf.layers.BatchNormalization(axis=self._bn_axis, name='{}_bn2'.format(name),
                                             _scope='{}_bn2'.format(name), _reuse=None)
            )
            conv_bn_blocks.append(
                ReLuLayer('{}_relu2'.format(name), _scope='{}_relu2'.format(name), _reuse=None)
            )
            return conv_bn_blocks

    def multibox_head(self, feature_layers, num_classes, num_anchors_depth_per_layer, data_format='channels_first',
                      bn_detection_head=False, training=False):
        with tf.variable_scope('multibox_head'):
            cls_preds = []
            loc_preds = []
            for ind, feat in enumerate(feature_layers):
                loc_pred = tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, (3, 3), use_bias=True,
                                            name='loc_{}'.format(ind), strides=(1, 1),
                                            padding='same', data_format=data_format, activation=None,
                                            kernel_initializer=tf.glorot_uniform_initializer(),
                                            bias_initializer=tf.zeros_initializer())
                if bn_detection_head:
                    loc_pred = tf.layers.batch_normalization(loc_pred, axis=self._bn_axis,
                                                             name='loc_preds_bn{}'.format(ind), training=training)

                cls_pred = tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * num_classes, (3, 3), use_bias=True,
                                            name='cls_{}'.format(ind), strides=(1, 1),
                                            padding='same', data_format=data_format, activation=None,
                                            kernel_initializer=tf.glorot_uniform_initializer(),
                                            bias_initializer=tf.zeros_initializer())
                if bn_detection_head:
                    cls_pred = tf.layers.batch_normalization(cls_pred, axis=self._bn_axis,
                                                             name='cls_preds_bn{}'.format(ind), training=training)
                loc_preds.append(loc_pred)
                cls_preds.append(cls_pred)
            return loc_preds, cls_preds


