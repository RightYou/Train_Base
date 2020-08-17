import sys

if not hasattr(sys, 'argv'):
    sys.argv = ['']

from WARN import model as model
from UTILS import *

tplt1 = "{0:^30}\t{1:^10}\t{2:^10}\t{3:^10}\t{4:^10}"  # \t{4:^10}\t{5:^10}
tplt2 = "{0:^30}\t{1:^10}\t{2:^10}"

config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True


def prepare_test_data(fileOrDir):
    original_ycbcr = []
    imgCbCr = []
    gt_y = []
    fileName_list = []
    # The input is a single file.
    if type(fileOrDir) is str:
        fileName_list.append(fileOrDir)

        # w, h = get_w_h(fileOrDir)
        # imgY = getYdata(fileOrDir, [w, h])
        imgY = c_get_y_data(fileOrDir)
        imgY = normalize(imgY)

        imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
        original_ycbcr.append([imgY, imgCbCr])

    ##The input is one directory of test images.
    elif len(fileOrDir) == 1:
        fileName_list = load_file_list(fileOrDir)
        for path in fileName_list:
            # w, h = get_w_h(path)
            # imgY = getYdata(path, [w, h])
            imgY = c_get_y_data(path)
            imgY = normalize(imgY)

            imgY = np.resize(imgY, (1, imgY.shape[0], imgY.shape[1], 1))
            original_ycbcr.append([imgY, imgCbCr])

    ##The input is two directories, including ground truth.
    elif len(fileOrDir) == 2:

        fileName_list = load_file_list(fileOrDir[0])
        test_list = get_train_list(load_file_list(fileOrDir[0]), load_file_list(fileOrDir[1]))
        for pair in test_list:
            filesize = os.path.getsize(pair[0])
            picsize = get_w_h(pair[0])[0] * get_w_h(pair[0])[0] * 3 // 2
            numFrames = filesize // picsize
            # if numFrames ==1:
            or_imgY = c_get_y_data(pair[0])
            gt_imgY = c_get_y_data(pair[1])

            # normalize
            or_imgY = normalize(or_imgY)

            or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))

            ## act as a placeholder
            or_imgCbCr = 0
            original_ycbcr.append([or_imgY, or_imgCbCr])
            gt_y.append(gt_imgY)
            # else:
            #     while numFrames>0:
            #         or_imgY =getOneFrameY(pair[0])
            #         gt_imgY =getOneFrameY(pair[1])
            #         # normalize
            #         or_imgY = normalize(or_imgY)
            #
            #         or_imgY = np.resize(or_imgY, (1, or_imgY.shape[0], or_imgY.shape[1], 1))
            #         gt_imgY = np.resize(gt_imgY, (1, gt_imgY.shape[0], gt_imgY.shape[1], 1))
            #
            #         ## act as a placeholder
            #         or_imgCbCr = 0
            #         original_ycbcr.append([or_imgY, or_imgCbCr])
            #         gt_y.append(gt_imgY)
    else:
        print("Invalid Inputs.")
        exit(0)

    return original_ycbcr, gt_y, fileName_list


class Predict:
    input_tensor = None
    output_tensor = None
    model = None

    def __init__(self, model, modelpath):
        self.graph = tf.Graph()  # 为每个类(实例)单独创建一个graph
        self.model = model
        with self.graph.as_default():
            self.input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
            # self.output_tensor = tf.make_template('input_scope', self.model)(self.input_tensor)
            # self.output_tensor = self.model(self.input_tensor,is_training=False)
            self.output_tensor = self.model(self.input_tensor)
            self.output_tensor = tf.clip_by_value(self.output_tensor, 0., 1.)
            self.output_tensor = tf.multiply(self.output_tensor, 255)
            self.saver = tf.train.Saver()
        self.sess = tf.Session(graph=self.graph, config=config)  # 创建新的sess
        with self.sess.as_default():
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                self.saver.restore(self.sess, modelpath)  # 从恢复点恢复参数
                print(modelpath)

    def predict(self, fileOrDir):
        if isinstance(fileOrDir, str):
            original_ycbcr, gt_y, fileName_list = prepare_test_data(fileOrDir)
            imgY = original_ycbcr[0][0]

        elif type(fileOrDir) is np.ndarray:
            imgY = fileOrDir

        elif isinstance(fileOrDir, list):
            fileOrDir = np.asarray(fileOrDir, dtype='float32')
            imgY = normalize(np.reshape(fileOrDir, (1, len(fileOrDir), len(fileOrDir[0]), 1)))

        else:
            imgY = None

        with self.sess.as_default():
            with self.sess.graph.as_default():
                out = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: imgY})
                out = np.reshape(out, (out.shape[1], out.shape[2]))
                out = np.around(out)
                out = out.astype('int')
                out = out.tolist()
                return out


def test_all_ckpt(modelPath):
    low_img = r"F:\0wzy_Data\test_set\QP53"
    heigh_img = r"F:\0wzy_Data\test_set\label"

    original_ycbcr, gt_y, fileName_list = prepare_test_data([low_img, heigh_img])
    total_imgs = len(fileName_list)

    tem = [f for f in os.listdir(modelPath) if 'data' in f]
    ckptFiles = sorted([r.split('.data')[0] for r in tem])
    max_psnr = 0
    max_epoch = 0
    max_ckpt_psnr = 0

    for ckpt in ckptFiles:
        cur_ckpt_psnr_sum = 0
        epoch = int(ckpt.split('.')[0].split('_')[-2])

        if epoch != 279:
            continue

        print(os.path.join(modelPath, ckpt))
        predictor = Predict(model, os.path.join(modelPath, ckpt))

        img_index = [14, 17, 4, 2, 7, 10, 12, 3, 0, 13, 16, 5, 6, 1, 15, 8, 9, 11]
        for i in img_index:
            imgY = original_ycbcr[i][0]
            gtY = gt_y[i] if gt_y else 0
            rec = predictor.predict(imgY)

            # showImg(rec)
            # print(np.shape(np.reshape(imgY, [np.shape(imgY)[1],np.shape(imgY)[2]])))
            # cur_psnr[cnnTime]=psnr(denormalize(np.reshape(imgY, [np.shape(imgY)[1],np.shape(imgY)[2]])),np.reshape(gtY, [np.shape(imgY)[1],np.shape(imgY)[2]]))
            cur_psnr = psnr(rec, np.reshape(gtY, np.shape(rec)))

            # cur_psnr = psnr(denormalize(np.reshape(imgY, np.shape(rec))), np.reshape(gtY, np.shape(rec)))
            cur_ckpt_psnr_sum += cur_psnr
            print(tplt2.format(os.path.basename(fileName_list[i]), cur_psnr,
                               psnr(denormalize(np.reshape(imgY, np.shape(rec))), np.reshape(gtY, np.shape(rec)))))

        cur_ckpt_psnr = cur_ckpt_psnr_sum / total_imgs
        if cur_ckpt_psnr > max_ckpt_psnr:
            max_ckpt_psnr = cur_ckpt_psnr
            max_epoch = epoch
        print("______________________________________________________________")
        print(epoch, cur_ckpt_psnr, max_epoch, max_ckpt_psnr)


if __name__ == '__main__':
    test_all_ckpt(r"I:\WZY\2号\Train_Base\checkpoints\RWARN_14w_QP47-56_200810")
