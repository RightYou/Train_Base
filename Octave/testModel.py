import numpy as np
import tensorflow as tf
from Octave.tf_octConv import *
from Octave.tf_cnn_basic import *
from Octave.oct_Resnet_unit import *


def model(input_tensor):  # 一开始传入网路的数据是[64,64,64,1]，使用了占位符
    # print(input_tensor)  # Tensor("input_scope/Placeholder:0", shape=(32, 64, 64, 1), dtype=float32)

    # conv1：第一次卷积，用于扩宽通道数，不涉及八度卷积
    # 这里是否批标准化和激活还需要考虑
    conv1 = Conv_BN_AC(data=input_tensor, num_filter=32, kernel=(3, 3), name='conv1_Expand')
    # print(conv1)  # Tensor("input_scope/conv1_Expand__ac__relu:0", shape=(64, 64, 64, 32), dtype=float32)

    alpha = 0.5  # 八度卷积频率比例系数

    # 这里涉及四种残差结构，单频率到双频率，双频率到双频率，双频率到单频率，单频率到单频率

    # conv2：开始八度卷积，第一次类残差结构，主要完成一个频率到高低频率的过程
    # 设置通道数的变化
    num_in = 32
    num_mid_1 = 192
    num_mid_2 = 24
    num_out = 32

    hf_conv2_x, lf_conv2_x = Residual_Unit_first(data=conv1, alpha=alpha, num_in=num_in,
                                                 num_mid_1=num_mid_1, num_mid_2=num_mid_2, num_out=num_out,
                                                 name='conv2_FirstRes')

    # conv3：第二类残差结构
    global hf_conv3_x, lf_conv3_x, conv5_x

    for i in range(4):
        hf_conv3_x, lf_conv3_x = Residual_Unit(
            hf_data=(hf_conv2_x if i == 0 else hf_conv3_x), lf_data=(lf_conv2_x if i == 0 else lf_conv3_x),
            alpha=alpha, num_in=num_in, num_mid_1=num_mid_1, num_mid_2=num_mid_2, num_out=num_out,
            name=('conv3_%02dRes' % i))

    # conv4: 第三类残差结构
    conv4_x = Residual_Unit_last(hf_data=hf_conv3_x, lf_data=lf_conv3_x, alpha=alpha,
                                 num_in=num_in, num_mid_1=num_mid_1, num_mid_2=num_mid_2, num_out=num_out,
                                 name='conv4_ResMerge')

    # conv5：第四类残差结构
    for i in range(2):
        conv5_x = Residual_Unit_norm(
            data=conv4_x, num_in=num_out, num_mid_1=num_mid_1, num_mid_2=num_mid_2, num_out=num_out,
            name=('conv5_%02dRes' % i))

    # conv6：通道归一输出
    conv6 = Conv_BN_AC(data=conv5_x, num_filter=1, kernel=(3, 3), name='conv6_Out')

    return conv6
