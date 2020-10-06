import tensorflow as tf
from Octave.tf_cnn_basic import *


# 第一次八度卷积，输入只有一个张量
def firstOctConv(data, settings, ch_in, ch_out, name, kernel):
    alpha_in, alpha_out = settings  # settings=(0, alpha)，alpha取0.5
    hf_ch_in = int(ch_in * (1 - alpha_in))  # 高频输入算出来就是总的输入通道数
    hf_ch_out = int(ch_out * (1 - alpha_out))  # 高频输出算出来就是输出通道数的一半，然后待会直接卷积出这么多个通道就好了

    lf_ch_in = ch_in - hf_ch_in  # 0
    lf_ch_out = ch_out - hf_ch_out  # 低频输出算出来就是输出通道数的一半，也是直接卷积出这么多个通道就好了

    hf_data = data  # 将输入数据定义为高频数据

    # 卷积出输出的高频通道
    out_h = Conv(data=hf_data, num_filter=hf_ch_out, kernel=kernel, name=('%s_hf_conv' % name))

    # 输出的低频通道为输入的高频通道池化后的卷积结果
    hf_pool = Pooling(data=hf_data, pool_type='avg', kernel=(2, 2), stride=(2, 2), name=('%s_hf_pool' % name))
    out_l = Conv(data=hf_pool, num_filter=lf_ch_out, kernel=kernel, name=('%s_hf_pool_conv' % name))

    return out_h, out_l


# 普通八度卷积，输入高频和低频张量，返回叠加后的高频和低频张量
def OctConv(hf_data, lf_data, settings, ch_in, ch_out, kernel, name):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    lf_ch_in = ch_in - hf_ch_in
    lf_ch_out = ch_out - hf_ch_out

    hf_conv = Conv(data=hf_data, num_filter=hf_ch_out, kernel=kernel, name=('%s_hf_conv' % name))
    lf_conv_hf = Conv(data=lf_data, num_filter=hf_ch_out, kernel=kernel, name=('%s_lf_conv_hf' % name))
    lf_upsample = UpSampling(lf_conv_hf, scale=2, name='%s_lf_upsample' % name)

    lf_conv = Conv(data=lf_data, num_filter=lf_ch_out, kernel=kernel, name=('%s_lf_conv' % name))
    hf_pool = Pooling(data=hf_data, pool_type='avg', kernel=(2, 2), stride=(2, 2), name=('%s_hf_pool' % name))
    hf_pool_conv = Conv(data=hf_pool, num_filter=lf_ch_out, kernel=kernel, name=('%s_hf_pool_conv' % name))

    out_h = hf_conv + lf_upsample
    out_l = hf_pool_conv + lf_conv

    return out_h, out_l


# 最后一次八度卷积，输入高频和低频张量，返回两个张量的叠加
def lastOctConv(hf_data, lf_data, settings, ch_in, ch_out, name, kernel=(1, 1)):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    hf_conv = Conv(data=hf_data, num_filter=hf_ch_out, kernel=kernel, name=('%s_hf_last_conv' % name))
    lf_conv = Conv(data=lf_data, num_filter=hf_ch_out, kernel=kernel, name=('%s_lf_last_conv' % name))
    lf_upsample = UpSampling(lf_conv, scale=2, name='%s_lf_upsample' % name)
    out_h = hf_conv + lf_upsample

    return out_h


def firstOctConv_BN_AC(data, alpha, num_filter_in, num_filter_out, kernel, name):
    hf_data, lf_data = firstOctConv(data=data, settings=(0, alpha), ch_in=num_filter_in, ch_out=num_filter_out,
                                    kernel=kernel, name=name)
    out_hf = BN_AC(data=hf_data, name='%s_hf' % name)
    out_lf = BN_AC(data=lf_data, name='%s_lf' % name)
    return out_hf, out_lf


def lastOctConv_BN_AC(hf_data, lf_data, alpha, num_filter_in, num_filter_out, kernel, name):
    conv = lastOctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, 0), ch_in=num_filter_in,
                       ch_out=num_filter_out, name=name, kernel=kernel)
    out = BN_AC(data=conv, name=name)
    return out


def octConv_BN_AC(hf_data, lf_data, alpha, num_filter_in, num_filter_out, kernel, name):
    hf_data, lf_data = OctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, alpha), ch_in=num_filter_in,
                               ch_out=num_filter_out, name=name, kernel=kernel)
    out_hf = BN_AC(data=hf_data, name='%s_hf' % name)
    out_lf = BN_AC(data=lf_data, name='%s_lf' % name)
    return out_hf, out_lf


def firstOctConv_BN(data, alpha, num_filter_in, num_filter_out, kernel, name):
    hf_data, lf_data = firstOctConv(data=data, settings=(0, alpha), ch_in=num_filter_in, ch_out=num_filter_out,
                                    name=name, kernel=kernel)
    out_hf = BN(data=hf_data, name='%s_hf' % name)
    out_lf = BN(data=lf_data, name='%s_lf' % name)
    return out_hf, out_lf


def lastOctConv_BN(hf_data, lf_data, alpha, num_filter_in, num_filter_out, kernel, name):
    conv = lastOctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, 0), ch_in=num_filter_in,
                       ch_out=num_filter_out, name=name, kernel=kernel)
    out = BN(data=conv, name=name)
    return out


def octConv_BN(hf_data, lf_data, alpha, num_filter_in, num_filter_out, kernel, name):
    hf_data, lf_data = OctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, alpha), ch_in=num_filter_in,
                               ch_out=num_filter_out, name=name, kernel=kernel)
    out_hf = BN(data=hf_data, name='%s_hf' % name)
    out_lf = BN(data=lf_data, name='%s_lf' % name)
    return out_hf, out_lf
