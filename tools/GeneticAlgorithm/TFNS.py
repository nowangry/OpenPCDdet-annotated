import time
import numpy as np
import torchvision.transforms as transforms
from TDE_main_ins_patch_multi_fit import FDE
from darknet_v3 import Darknet
import os
import utils_self
import utils
from PIL import Image
import fnmatch
import TDE_main_ins_patch_multi_fit


print('hyper-paramenters,population size : {}, generations : {}, xmin : {:.4f}, xmax : {:.4f}, per ins dim_full : {:.4f}, per ins dim_sub : {:.4f}'.format(
    TDE_main_ins_patch_multi_fit.population_size, TDE_main_ins_patch_multi_fit.generations,
    TDE_main_ins_patch_multi_fit.xmin, TDE_main_ins_patch_multi_fit.xmax, TDE_main_ins_patch_multi_fit.ins_dim_full,
    TDE_main_ins_patch_multi_fit.ins_dim_sub))

print("start training time : ", time.strftime('%Y-%m-%d %H:%M:%S'))

imgdir = '/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/images'
clean_labdir = "/mnt/share1/tangguijian/Data_storage/Black_AE_Evo_testset/yolo-labels"
savedir = "ablation_study/population_size/plane/N_10"

# savedir = "testset_attack_100/p_30_nearest"     #   最近邻域插值
# savedir = "testset_attack_100/p_30_nearest"     #   双线性插值
# imgdir = 'target_set/origin_imgs'
# clean_labdir = "target_set/yolo-labels"
# savedir = "target_set/instan_patch/less_pixels_per_instance"

# imgdir = 'random_noise/imgs'
# clean_labdir = "random_noise/imgs/yolo-labels"
# savedir = "target_set/instan_patch/less_pixels/p_30_kron_max_base_6_F0.5"
print("savedir : ", savedir)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfgfile = "cfg/yolov3-dota.cfg"
weightfile = "/mnt/share1/tangguijian/DOTA_YOLOv3_patch_AT/weights/yolov3-dota_110000.weights"

model = Darknet(cfgfile)

model.load_darknet_weights(weightfile)
model = model.eval().cuda()

count = 0
total_count = 0
net_correct = 0

img_size = model.height
img_width = model.width
class_names = utils.load_class_names('data/dota.names')

instances_clean = []
instances_after_attack = []

t_begin = time.time()

n_png_images = len(fnmatch.filter(
    os.listdir(imgdir), '*.png'))
n_jpg_images = len(fnmatch.filter(
    os.listdir(imgdir), '*.jpg'))
n_images = n_png_images + n_jpg_images  # 应该和n_images_clean一致
print("Total images in testset : ", n_images)

for imgfile in os.listdir(imgdir):
    print("new image")  #
    t_single_begin = time.time()
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):  # 判断是否为指定文件结尾
        name = os.path.splitext(imgfile)[0]  # image name w/o extension
        # 将文件名分开
        txtname = name + '.txt'  # 将分离出来的文件名重新保存为txt文件
        txtpath = os.path.abspath(os.path.join(clean_labdir, txtname))

        # open beeld en pas aan naar yolo inputs size  # 打开图像并调整为yolo输入大小,对每一张图片进行操作
        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))  # 拼接成绝对路径
        print("image file path is ", imgfile)
        img = utils_self.load_image_file(imgfile)  # 注意这里的不同
        w, h = img.size
        # print("original w = ", w, ", h = ", h)
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2   # dim_to_pad = 1 if w < h else 2  # 根据原始图片的宽高决定怎么填充
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new(
                    'RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
            else:
                padding = (w - h) / 2
                padded_img = Image.new(
                    'RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))

        resize = transforms.Resize((img_size, img_size))
        padded_img = resize(padded_img)
        #   单张图片预测格式

        # boxes_cls = torchie_utils.do_detect(model, padded_img, 0.4, 0.4, True)
        # boxes_cls = torchie_utils.nms(boxes_cls, 0.4)

        # boxes = []  # 定义对特定类别的boxes
        # for box in boxes_cls:
        #     cls_id = box[6]
        #     if (cls_id == 0):
        #         if (box[2] >= 0.1 and box[3] >= 0.1):
        #             boxes.append(box)
        # #   此时得到了干净图片的预测结果，[x,y,w,h,det_conf,cls_conf,id]
        #   读取标签
        tru_lab = utils.read_truths_pre_7(txtpath)  # array数据
        # print("tru_lab : ", tru_lab[0], "int : ", int(tru_lab[0][0]*608))
        # _, pre = torch.max(outputs.data, 1)
        total_count += len(tru_lab)  # 总共的instances个数
        instances_clean.append(len(tru_lab))  # 保存真实标签instances数目

        if len(tru_lab):  # 如果真实标签和预测标签长度相同，即预测的instances相同
            net_correct += len(tru_lab)
            images_to_attack = utils_self.img_transfer(
                padded_img)  # tensor,[3x608x608],这里对w和h进行了交换
            # print("size of : ", img.size())
            # 得到对抗样本，tensor形式[3,608,608]
            images_attack = FDE(images_to_attack, tru_lab)
            #   tru_lab是array格式
            images_attack = transforms.ToPILImage(
                'RGB')(images_attack.cpu())  # 转RGB图片
            #   transforms.ToPILImage()(x)将CHW变成了CWH，但是通道并没有打印

            ############################################################
            #   保存攻击完后的图片，可用于跨检测器迁移攻击
            ############################################################
            # save_name_attacked = name + '.png'
            # attacked_dir = os.path.join(
            #     savedir, 'img_patched/', save_name_attacked)
            # images_attack.save(attacked_dir)

            boxes_cls_attack = utils.do_detect(
                model, images_attack, 0.4, 0.4, True)
            # boxes_cls_attack = torchie_utils.nms(boxes_cls_attack, 0.4)
            boxes_attack = []  # 定义对特定类别的boxes
            for box in boxes_cls_attack:
                cls_id = box[6]
                if (cls_id == 0):
                    if (box[2] >= 0.1 and box[3] >= 0.1):
                        boxes_attack.append(box)

            count += len(boxes_attack)  # 攻击完后还是目标类被的数量
            instances_after_attack.append(
                len(boxes_attack))  # 保存攻击完后仍然是飞机instances数目
        ########################################################################
        #   下面保存攻击后图片，并作图
        ########################################################################

            # pre_name = name + ".png"  # 不添加后缀
            # pre_dir = os.path.join(
            #     savedir, 'patched_pre/', pre_name)

            # torchie_utils.plot_boxes(images_attack, boxes_attack, pre_dir,
            #                  class_names=class_names)  # 这里画出来的预测框是所有的目标，并不是只含有plane

            txtpath_write = os.path.abspath(os.path.join(
                savedir, 'yolo-labels/', txtname))
            textfile = open(txtpath_write, 'w+')
            #   需要注意，此时写入的是攻击后图片的预测结果，不再只写入预测为飞机的instances
            for box in boxes_attack:
                textfile.write(
                    f'{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]} {box[6]}\n')
            textfile.close()
            print("single image tru-instances : ", len(tru_lab), "instances after attack : ", len(boxes_attack), "instances gap : ", (len(tru_lab)-len(boxes_attack)))
        attacked_count = net_correct - count

    # if net_correct ==500:
    #     break
    print("image ", imgfile, "attack done!")
    t_single_end = time.time()
    print('singel attack time: {:.4f} minutes'.format(
        (t_single_end - t_single_begin) / 60))
print("total instances : ", total_count, "correct clean instances : ", net_correct,
      "instances after attack : ", count, "total instances gap : ", attacked_count)
if net_correct > 0:
    print('Accuracy of attack: %f %%' %
          (100 * float(attacked_count) / net_correct))

# np.save("attacked_img_save/instances_count_save/" +
#         "clean.npy", instances_clean)
# np.save("attacked_img_save/instances_count_save/" +
#         "attacked_img.npy", instances_after_attack)
print("All Done!")

t_end = time.time()
print('Total attack time: {:.4f} minutes'.format(
    (t_end - t_begin) / 60))
