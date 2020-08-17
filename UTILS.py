import math
import os
import random
import threading
import numpy as np
import tensorflow as tf
from PIL import Image

# 训练普通模型的时候改成TRAIN就好了
# from TRAIN_CRLC import BATCH_SIZE
# from TRAIN_CRLC import PATCH_SIZE


# 环内测试可能递归加载TRAIN_CRLC，所以直接将BATCH_SIZE和PATCH_SIZE直接赋值在这里
PATCH_SIZE = (64, 64)
BATCH_SIZE = 64


# 截断，将输入控制在min和max中，加_用于与方法区分
def truncate(input_, min_, max_):
    input_ = np.where(input_ > min_, input_, min_)
    input_ = np.where(input_ < max_, input_, max_)
    return input_


# 归一化，把x缩放到0到1之间
def normalize(x):
    x = x / 255.
    return truncate(x, 0., 1.)


# 反归一化，把x还原回0到255
def denormalize(x):
    x = x * 255.
    return truncate(x, 0., 255.)


# 从文件夹路径中读取文件的绝对路径，返回排序后的列表
def load_file_list(directory):
    li = []

    # method 1：只读取文件，不看格式，不会读取子文件夹内文件
    # for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory, y))]:
    #     li.append(os.path.join(directory, filename))

    # method 2：只读取yuv文件，不会读取子文件夹内文件
    # for filename in [y for y in os.listdir(directory) if os.path.isfile(os.path.join(directory, y))]:
    #     if filename.split(".")[-1] == "yuv":
    #         li.append(os.path.join(directory, filename))

    # method 3：只读取yuv文件，会读取子文件夹内的yuv文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_name = os.path.join(root, file)
            if file_name.split(".")[-1] == "yuv":
                li.append(file_name)

    return sorted(li)


# 获取训练集
def get_train_list(low_list, high_list):
    train_list = []
    # method 1：单QP训练，并获取指定index范围内的数据
    # 使用断言，如果训练集数据和标签长度不一样，触发异常，并不适用于多qp重建图像，共用一组label的情况
    # assert len(low_list) == len(high_list), "low:%d, high:%d 数据与标签不等" % (len(low_list), len(high_list))
    # for i in range(len(low_list)):
    #     idx = int(low_list[i].split("\\")[-1].split("_")[0])
    #     if idx <= 500:
    #         train_list.append([low_list[i], high_list[i]])

    # method 2：范围QP训练，yuv文件上层目录名需为 qpxx
    assert len(low_list) % len(high_list) == 0, "low:%d, high:%d" % (len(low_list), len(high_list))
    for i in range(len(low_list)):
        qp = int(low_list[i].split("\\")[-2].split("QP")[-1])
        if 47 <= qp <= 56:
            train_list.append([low_list[i], high_list[i % len(high_list)]])

    return train_list


# 传入图片的绝对路径，通过字符串切割，返回图片的宽高。
def get_w_h(yuv_file_name):
    # method 1：使用正则表达式匹配
    # de_yuv = re.compile(r'(.+?)\.')
    # de_yuv_file_name = de_yuv.findall(yuv_file_name)[0]  # 去yuv文件的后缀名
    # if os.path.basename(de_yuv_file_name).split("_")[0].isdigit():
    #     wxh = os.path.basename(de_yuv_file_name).split('_')[1]
    # else:
    #     wxh = os.path.basename(de_yuv_file_name).split('_')[1]

    # method 2：字符串处理法，对文件命名有要求
    yuv_file_name = os.path.basename(yuv_file_name)  # 获取文件名
    wxh = yuv_file_name.replace('.yuv', '').split('_')[-1]  # 获得number x number的字符串

    w, h = wxh.split('x')
    return int(w), int(h)


# 传入图像路径和宽高，获得y分量数据
def get_y_data(path, size):
    w, h = size[0], size[1]
    with open(path, 'rb') as fp:
        # method 1：间接转换法，yuv图像的三个分量按yuv三个通道依次写在文件中，通过读取前w*h个字节，就能读取出y分量
        fp.seek(0, 0)  # 第一个0表示偏移量，第二个0表示从哪里偏移，为1时从文件末尾开始
        y_read = fp.read()
        temp = Image.frombytes('L', [w, h], y_read)
        y_data = np.asarray(temp, dtype='float32')  # 这里y_data为一个numpy数组，维度是[h,w]，as array为浅拷贝

        # method 2：遍历读取法
        # fp.seek(0, 0)  # 如果读取U分量，就需要偏移量为w*h
        # y_data1 = np.zeros([h, w], dtype="float32", order='C')  # C column代表行优先，即h行w列的数组
        # for n in range(h):
        #     for m in range(w):
        #         y_data1[n, m] = ord(fp.read(1))

    return y_data


# 整合了一下，写了个接口，传入图片的路径，获取宽高，然后调用get_y_data函数，获得yuv图片y分量的数据
def c_get_y_data(path):
    return get_y_data(path, get_w_h(path))


def crop(input_image, gt_image, patch_width, patch_height, img_type):
    assert type(input_image) == type(gt_image), "types are different."
    # return a ndarray object
    if img_type == "ndarray":
        in_row_ind = random.randint(0, input_image.shape[0] - patch_width)
        in_col_ind = random.randint(0, input_image.shape[1] - patch_height)

        input_cropped = input_image[in_row_ind:in_row_ind + patch_width, in_col_ind:in_col_ind + patch_height]
        gt_cropped = gt_image[in_row_ind:in_row_ind + patch_width, in_col_ind:in_col_ind + patch_height]

    # return an "Image" object
    elif img_type == "Image":
        in_row_ind = random.randint(0, input_image.size[0] - patch_width)
        in_col_ind = random.randint(0, input_image.size[1] - patch_height)

        input_cropped = input_image.crop(
            box=(in_row_ind, in_col_ind, in_row_ind + patch_width, in_col_ind + patch_height))
        gt_cropped = gt_image.crop(box=(in_row_ind, in_col_ind, in_row_ind + patch_width, in_col_ind + patch_height))

    else:
        input_cropped = None
        gt_cropped = None
        print("Error in crop!")
    return input_cropped, gt_cropped


class Reader(threading.Thread):
    def __init__(self, file_name, id, input_list, gt_list):
        super(Reader, self).__init__()
        self.file_name = file_name
        self.id = id
        self.input_list = input_list
        self.gt_list = gt_list

    def run(self):
        input_image = c_get_y_data(self.file_name[0])
        gt_image = c_get_y_data(self.file_name[1])
        in_ = []
        gt_ = []
        for j in range(BATCH_SIZE // 8):
            input_imgY, gt_imgY = crop(input_image, gt_image, PATCH_SIZE[0], PATCH_SIZE[1], "ndarray")
            input_imgY = normalize(input_imgY)  # 这里的input_imgY是64x64的
            gt_imgY = normalize(gt_imgY)

            in_.append(input_imgY)
            gt_.append(gt_imgY)

        self.input_list[self.id] = in_
        self.gt_list[self.id] = gt_


def prepare_nn_data(train_list):
    thread_num = 8
    batch_size_random_list = random.sample(range(0, len(train_list)), thread_num)
    input_list = [0 for i in range(thread_num)]
    gt_list = [0 for i in range(thread_num)]
    t = []
    for i in range(thread_num):
        t.append(Reader(train_list[batch_size_random_list[i]], i, input_list, gt_list))
    for i in range(thread_num):
        t[i].start()
    for i in range(thread_num):
        t[i].join()
    input_list = np.reshape(input_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))
    gt_list = np.reshape(gt_list, (BATCH_SIZE, PATCH_SIZE[0], PATCH_SIZE[1], 1))

    return input_list, gt_list


# 计算PSNR
def psnr(hr_image, sr_image, max_value=255.0):
    eps = 1e-10
    if isinstance(hr_image, np.ndarray) or isinstance(hr_image, list):
        hr_image_data = np.asarray(hr_image, 'float32')
        sr_image_data = np.asarray(sr_image, 'float32')

        diff = sr_image_data - hr_image_data
        mse = np.mean(diff * diff)
        mse = np.maximum(eps, mse)
        return float(10 * math.log10(max_value * max_value / mse))
    else:
        assert len(hr_image.shape) == 4 and len(sr_image.shape) == 4
        diff = hr_image - sr_image
        mse = tf.reduce_mean(tf.square(diff))
        mse = tf.maximum(mse, eps)
        return 10 * tf.log(max_value * max_value / mse) / math.log(10)
