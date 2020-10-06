import tensorflow as tf
import numpy as np
from Octave.tf_cnn_basic import BN_AC, AC

def resblock(temp_tensor, convId):
    conv_secondID = 0
    skip_tensor = temp_tensor

    conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [1, 1, 12, 32],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [32], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
    out_tensor = tf.nn.relu(
        tf.nn.bias_add(tf.nn.conv2d(temp_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b))
    conv_secondID += 1

    conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [1, 1, 32, 8],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [8], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
    out_tensor = tf.nn.bias_add(tf.nn.conv2d(out_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
    conv_secondID += 1

    conv_w = tf.get_variable("conv_%02d_%02d_w" % (convId, conv_secondID), [3, 3, 8, 12],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_%02d_b" % (convId, conv_secondID), [12], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
    out_tensor = tf.nn.bias_add(tf.nn.conv2d(out_tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
    conv_secondID += 1

    # skip + out_tensor
    out_tensor = tf.add(skip_tensor, out_tensor)
    return out_tensor


def model(input_tensor):
    convId = 0
    conv_00_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 1, 12],
                                initializer=tf.contrib.layers.xavier_initializer())
    conv_00_b = tf.get_variable("conv_%02d_b" % (convId), [12], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_00_w))
    tensor = tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b)
    convId += 1

    # Residual Block x 3
    for i in range(3):
        tensor = resblock(tensor, convId)
        convId += 1

    conv_w = tf.get_variable("conv_%02d_w" % (convId), [3, 3, 12, 2],
                             initializer=tf.contrib.layers.xavier_initializer())
    conv_b = tf.get_variable("conv_%02d_b" % (convId), [2], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)

    tensor = tf.add(tensor, input_tensor)

    return tensor

def crlc_octave_model(input_tensor):
    alpha = 0.5
    num_in = 32
    num_mid = 64
    num_out = 256
    i = 1

    hf_conv1_x, lf_conv1_x = Residual_Unit_first(
        data=input_tensor,
        alpha=alpha,
        num_in=(num_in if i == 1 else num_out),
        num_mid=num_mid,
        num_out=num_out,
        name=('conv2_B%02d' % i),
        first_block=(i == 1),
        stride=((1, 1) if (i == 1) else (1, 1)))


def Residual_Unit_first(data, alpha, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    hf_data_m, lf_data_m = firstOctConv_BN_AC(data=data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_mid,
                                              kernel=(1, 1), pad='valid', name=('%s_conv-m1' % name))
    hf_data_m, lf_data_m = octConv_BN_AC(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid,
                                         num_filter_out=num_mid, kernel=(3, 3), pad='same',
                                         name=('%s_conv-m2' % name), stride=stride, num_group=g)
    hf_data_m, lf_data_m = octConv_BN(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid,
                                      num_filter_out=num_out, kernel=(1, 1), pad='valid', name=('%s_conv-m3' % name))

    if first_block:
        hf_data, lf_data = firstOctConv_BN(data=data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_out,
                                           kernel=(1, 1), pad='valid', name=('%s_conv-w1' % name), stride=stride)

    hf_outputs = ElementWiseSum(hf_data, hf_data_m, name=('%s_hf_sum' % name))
    lf_outputs = ElementWiseSum(lf_data, lf_data_m, name=('%s_lf_sum' % name))

    hf_outputs = AC(hf_outputs, name=('%s_hf_act' % name))
    lf_outputs = AC(lf_outputs, name=('%s_lf_act' % name))
    return hf_outputs, lf_outputs


def firstOctConv_BN_AC(data, alpha, num_filter_in, num_filter_out, kernel, pad, stride=(1, 1), name=None, w=None,
                       b=None, no_bias=True, attr=None, num_group=1):
    hf_data, lf_data = firstOctConv(data=data, settings=(0, alpha), ch_in=num_filter_in, ch_out=num_filter_out,
                                    name=name, kernel=kernel, pad=pad, stride=stride)
    out_hf = BN_AC(data=hf_data, name=('%s_hf') % name)
    out_lf = BN_AC(data=lf_data, name=('%s_lf') % name)
    return out_hf, out_lf


# 第一次八度卷积，只区分高低频，所以卷积核取1x1,不填充，滑动步长也为1x1
def firstOctConv(data, settings, ch_in, ch_out, name, kernel=(1, 1), pad='valid', stride=(1, 1)):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    lf_ch_in = ch_in - hf_ch_in
    lf_ch_out = ch_out - hf_ch_out

    hf_data = data

    if stride == (2, 2):
        hf_data = Pooling(data=hf_data, pool_type='avg', kernel=(2, 2), stride=(2, 2), name=('%s_hf_down' % name))
    hf_conv = Conv(data=hf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1, 1),
                   name=('%s_hf_conv' % name))
    hf_pool = Pooling(data=hf_data, pool_type='avg', kernel=(2, 2), stride=(2, 2), name=('%s_hf_pool' % name))
    hf_pool_conv = Conv(data=hf_pool, num_filter=lf_ch_out, kernel=kernel, pad=pad, stride=(1, 1),
                        name=('%s_hf_pool_conv' % name))

    out_h = hf_conv
    out_l = hf_pool_conv
    return out_h, out_l


def conv2d(tensor, shape, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv_w = tf.get_variable("conv_w", shape, initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / (shape[0] * shape[1] * shape[2]))))
        conv_b = tf.get_variable("conv_b", [shape[-1]], initializer=tf.constant_initializer(0))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
        convResult = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
    return convResult


def conv2dc(tensor, num_filter, kernel, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        shape = kernel
        shape[2] = num_filter

        conv_w = tf.get_variable("conv_w", shape, initializer=tf.random_normal_initializer(
            stddev=np.sqrt(2.0 / (shape[0] * shape[1] * shape[2]))))
        conv_b = tf.get_variable("conv_b", [shape[-1]], initializer=tf.constant_initializer(0))
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
        convResult = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
    return convResult


def Conv(data, num_filter, kernel, stride=(1, 1), pad='valid', name=None, no_bias=False, w=None, b=None, attr=None,
         num_group=1):
    if w is None:
        conv = tf.layers.conv2d(inputs=data, filters=num_filter, kernel_size=kernel,
                                strides=stride, padding=pad, name=('%s__conv' % name), use_bias=no_bias)
    else:
        if b is None:
            conv = tf.layers.conv2d(data=data, num_filter=num_filter, kernel_size=kernel,
                                    stride=stride, padding=pad, name=('%s__conv' % name), use_bias=no_bias,
                                    kernel_initializer=w)
        else:
            conv = tf.layers.conv2d(data=data, num_filter=num_filter, kernel_size=kernel,
                                    stride=stride, padding=pad, name=('%s__conv' % name), use_bias=True,
                                    kernel_initializer=w, bias_initializer=b)
    return conv


def Pooling(data, pool_type='avg', kernel=(2, 2), pad='valid', stride=(2, 2), name=None):
    if pool_type == 'avg':
        return tf.layers.average_pooling2d(inputs=data, pool_size=kernel, strides=stride, padding=pad, name=name)
    elif pool_type == 'max':
        return tf.layers.max_pooling2d(inputs=data, pool_size=kernel, strides=stride, padding=pad, name=name)


def UpSampling(lf_conv, scale=2, sample_type='nearest', num_args=1, name=None):
    return tf.keras.layers.UpSampling2D(size=(scale, scale), name=name)(lf_conv)


def lastOctConv(hf_data, lf_data, settings, ch_in, ch_out, name, kernel=(1, 1), pad='valid', stride=(1, 1)):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    if stride == (2, 2):
        hf_data = Pooling(data=hf_data, pool_type='avg', kernel=(2, 2), stride=(2, 2), name=('%s_hf_down' % name))
    hf_conv = Conv(data=hf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1, 1),
                   name=('%s_hf_conv' % name))

    lf_conv = Conv(data=lf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1, 1),
                   name=('%s_lf_conv' % name))
    out_h = hf_conv + lf_conv

    return out_h


def OctConv(hf_data, lf_data, settings, ch_in, ch_out, name, kernel=(1, 1), pad='valid', stride=(1, 1)):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    lf_ch_in = ch_in - hf_ch_in
    lf_ch_out = ch_out - hf_ch_out

    if stride == (2, 2):
        hf_data = Pooling(data=hf_data, pool_type='avg', kernel=(2, 2), stride=(2, 2), name=('%s_hf_down' % name))
    hf_conv = Conv(data=hf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1, 1),
                   name=('%s_hf_conv' % name))
    hf_pool = Pooling(data=hf_data, pool_type='avg', kernel=(2, 2), stride=(2, 2), name=('%s_hf_pool' % name))
    hf_pool_conv = Conv(data=hf_pool, num_filter=lf_ch_out, kernel=kernel, pad=pad, stride=(1, 1),
                        name=('%s_hf_pool_conv' % name))

    lf_conv = Conv(data=lf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1, 1),
                   name=('%s_lf_conv' % name))
    if stride == (2, 2):
        lf_upsample = lf_conv
        lf_down = Pooling(data=lf_data, pool_type='avg', kernel=(2, 2), stride=(2, 2), name=('%s_lf_down' % name))
    else:
        lf_upsample = UpSampling(lf_conv, scale=2, sample_type='nearest', num_args=1, name='%s_lf_upsample' % name)
        lf_down = lf_data
    lf_down_conv = Conv(data=lf_down, num_filter=lf_ch_out, kernel=kernel, pad=pad, stride=(1, 1),
                        name=('%s_lf_down_conv' % name))

    out_h = hf_conv + lf_upsample
    out_l = hf_pool_conv + lf_down_conv

    return out_h, out_l
