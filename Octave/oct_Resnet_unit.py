import tensorflow as tf
from Octave.tf_cnn_basic import *
from Octave.tf_octConv import *


# 单频率的普通残差块
def Residual_Unit_norm(data, num_in, num_mid_1, num_mid_2, num_out, name):
    conv_m1 = Conv_BN_AC(data=data, num_filter=num_mid_1, kernel=(1, 1), name=('%s_conv-m1' % name))
    conv_m2 = Conv_BN(data=conv_m1, num_filter=num_mid_2, kernel=(3, 3), name=('%s_conv-m2' % name))
    conv_m3 = Conv_BN(data=conv_m2, num_filter=num_out, kernel=(1, 1), name=('%s_conv-m3' % name))

    outputs = ElementWiseSum(data, conv_m3, name=('%s_sum' % name))
    return AC(outputs)


# 最后一个残差块，将高低频率合为一个
def Residual_Unit_last(hf_data, lf_data, alpha, num_in, num_mid_1, num_mid_2, num_out, name):
    hf_data_m, lf_data_m = octConv_BN_AC(hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in,
                                         num_filter_out=num_mid_1, kernel=(1, 1), name=('%s_conv-m1' % name))
    hf_data_m, lf_data_m = octConv_BN(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid_1,
                                      num_filter_out=num_mid_2, kernel=(1, 1), name=('%s_conv-m2' % name))
    conv_m2 = lastOctConv_BN_AC(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid_2,
                                num_filter_out=num_out, name=('%s_conv-m2' % name), kernel=(3, 3))
    data = lastOctConv_BN(hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in,
                          num_filter_out=num_out, name=('%s_conv-w1' % name), kernel=(1, 1))

    outputs = ElementWiseSum(data, conv_m2, name=('%s_sum' % name))
    outputs = AC(outputs, name=('%s_act' % name))
    return outputs


# 第一个残差块，输入一个频率，输出高频和低频
def Residual_Unit_first(data, alpha, num_in, num_mid_1, num_mid_2, num_out, name):
    # 1x1的卷积核，将输入的通道卷积成中间通道，卷积时并没有传入控制w和b初始化的东西，firstOctConv_BN_AC和octConv_BN_AC的区别在于输入只有一个高频
    # 因为这是输入上的不一致，所以写成了两个函数
    hf_data_m, lf_data_m = firstOctConv_BN_AC(data=data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_mid_1,
                                              kernel=(1, 1), name=('%s_conv-m1' % name))
    # 1x1的卷积核，将中间通道卷积到中间通道
    hf_data_m, lf_data_m = octConv_BN(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid_1,
                                      num_filter_out=num_mid_2, kernel=(1, 1), name=('%s_conv-m2' % name))
    # 3x3的卷积核，将中间通道卷积到输出通道
    hf_data_m, lf_data_m = octConv_BN(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid_2,
                                      num_filter_out=num_out, kernel=(3, 3), name=('%s_conv-m3' % name))

    # 残差叠加也是两个通道分别进行，所以这里要先卷积出两个通道
    hf_data, lf_data = firstOctConv_BN(data=data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_out,
                                       kernel=(1, 1), name=('%s_conv-w1' % name))

    # 残差叠加
    hf_outputs = ElementWiseSum(hf_data, hf_data_m, name=('%s_hf_sum' % name))
    lf_outputs = ElementWiseSum(lf_data, lf_data_m, name=('%s_lf_sum' % name))

    # 叠加之后再激活，作为残差结构的输出
    hf_outputs = AC(hf_outputs, name=('%s_hf_act' % name))
    lf_outputs = AC(lf_outputs, name=('%s_lf_act' % name))
    return hf_outputs, lf_outputs


# 高低频通用残差块，和第一个残差块区别在于第一次卷积就是高频和低频分开输入了
def Residual_Unit(hf_data, lf_data, alpha, num_in, num_mid_1, num_mid_2, num_out, name):
    hf_data_m, lf_data_m = octConv_BN_AC(hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in,
                                         num_filter_out=num_mid_1, kernel=(1, 1), name=('%s_conv-m1' % name))
    hf_data_m, lf_data_m = octConv_BN(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid_1,
                                      num_filter_out=num_mid_2, kernel=(1, 1), name=('%s_conv-m2' % name))
    hf_data_m, lf_data_m = octConv_BN(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid_2,
                                      num_filter_out=num_out, kernel=(3, 3), name=('%s_conv-m3' % name))

    hf_outputs = ElementWiseSum(hf_data, hf_data_m, name=('%s_hf_sum' % name))
    lf_outputs = ElementWiseSum(lf_data, lf_data_m, name=('%s_lf_sum' % name))

    hf_outputs = AC(hf_outputs, name=('%s_hf_act' % name))
    lf_outputs = AC(lf_outputs, name=('%s_lf_act' % name))
    return hf_outputs, lf_outputs
