import tensorflow as tf


def BN(data, bn_momentum=0.9, name=None):
    return tf.layers.batch_normalization(data, momentum=bn_momentum, name=('%s__bn' % name))


def AC(data, name=None):
    return tf.nn.relu(data, name=('%s__relu' % name))


def BN_AC(data, name=None):
    bn = BN(data=data, name=name)
    bn_ac = AC(data=bn, name=name)
    return bn_ac


def Conv(data, num_filter, kernel, name):
    conv_w = tf.get_variable('%s__conv_w' % name, [kernel[0], kernel[1], data.shape[3], num_filter],
                             initializer=tf.contrib.layers.xavier_initializer())

    conv_b = tf.get_variable('%s__conv_b' % name, [num_filter], initializer=tf.constant_initializer(0))
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, tf.contrib.layers.l2_regularizer(1.)(conv_w))
    conv = tf.nn.bias_add(tf.nn.conv2d(data, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b,
                          name=('%s__conv' % name))
    return conv


def Conv_BN(data, num_filter, kernel, name):
    cov = Conv(data=data, num_filter=num_filter, kernel=kernel, name=name)
    cov_bn = BN(data=cov, name=('%s__bn' % name))
    return cov_bn


def Conv_BN_AC(data, num_filter, kernel, name):
    cov_bn = Conv_BN(data=data, num_filter=num_filter, kernel=kernel, name=name)
    cov_ba = AC(data=cov_bn, name=('%s__ac' % name))
    return cov_ba


def BN_Conv(data, num_filter, kernel, name=None):
    bn = BN(data=data, name=('%s__bn' % name))
    bn_cov = Conv(data=bn, num_filter=num_filter, kernel=kernel, name=name)
    return bn_cov


def AC_Conv(data, num_filter, kernel, name):
    ac = AC(data=data, name=('%s__ac' % name))
    ac_cov = Conv(data=ac, num_filter=num_filter, kernel=kernel, name=name)
    return ac_cov


def BN_AC_Conv(data, num_filter, kernel, name):
    bn = BN(data=data, name=('%s__bn' % name))
    ba_cov = AC_Conv(data=bn, num_filter=num_filter, kernel=kernel, name=name)
    return ba_cov


def Pooling(data, pool_type='avg', kernel=(2, 2), pad='valid', stride=(2, 2), name=None):
    if pool_type == 'avg':
        return tf.layers.average_pooling2d(inputs=data, pool_size=kernel, strides=stride, padding=pad, name=name)
    elif pool_type == 'max':
        return tf.layers.max_pooling2d(inputs=data, pool_size=kernel, strides=stride, padding=pad, name=name)


def ElementWiseSum(x, y, name):
    return tf.add(x=x, y=y, name=name)


def UpSampling(lf_conv, scale, name):
    return tf.keras.layers.UpSampling2D(size=(scale, scale), name=name)(lf_conv)
