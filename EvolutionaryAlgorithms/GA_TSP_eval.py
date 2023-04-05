import argparse
import glob
from pathlib import Path
import datetime
import os
# try:
#     import open3d1
#     from visual_utils import open3d_vis_utils as V
#
#     OPEN3D_FLAG = True
# except:
#     import mayavi.mlab as mlab
#     from visual_utils import visualize_utils as V
#
#     OPEN3D_FLAG = False

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import torch.nn.functional as F


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        if root_path is not None:
            data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
            data_file_list.sort()
            self.sample_file_list = data_file_list
        else:
            self.sample_file_list = []

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        # elif: self.ext == 'ready'
        #     points =
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config(data_path=None):
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str,
                        default='cfgs/kitti_models/pointrcnn-adv.yaml',
                        # default='cfgs/kitti_models/pv_rcnn.yaml',
                        # default='/home/nathan/OpenPCDet/tools/cfgs/kitti_models/second.yaml',
                        # default='/home/nathan/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml',
                        # default='cfgs/kitti_models/CaDDN.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str,
                        # default='/home/nathan/OpenPCDet/data/kitti/testing/velodyne',
                        # default='../data/GeneticAlgorithm/',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str,
                        # default="/home/nathan/OpenPCDet/tools/pointrcnn_7870.pth",
                        # default="/home/nathan/OpenPCDet/weights/caddn_pcdet.pth",
                        # default="/home/nathan/OpenPCDet/weights/pv_rcnn_8369.pth",
                        # default="/home/nathan/OpenPCDet/weights/voxel_rcnn_car_84.54.pth",
                        # default="/home/nathan/OpenPCDet/weights/pointpillar_7728.pth",
                        default="../checkpoints/pointrcnn_7870.pth",
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()
    args.data_path = data_path
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


def initial(exp_name):
    (filepath, filename) = os.path.split(log_file)
    os.makedirs(filepath, exist_ok=True)
    logger = common_utils.create_logger(log_file)
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    return logger


def network_forward(logger, data_path):
    args, cfg = parse_config(data_path)
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    num = 0
    pred_dicts_list = []
    recall_dicts_list = []
    batch_dict_list = []
    car_probability_list = []
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            if idx % 40 == 0:
                logger.info(f'Processing sample index: \t{idx}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, data_dict = model.forward(data_dict)
            # print("打印键值demo")
            # for key in data_dict:
            #     print(key)  # 打印key
            pred_dicts_list.append(pred_dicts)
            # recall_dicts_list.append(recall_dicts)
            batch_dict_list.append(data_dict)

            roi_cls_logit = data_dict['roi_cls_logit']
            softmax_ret = F.softmax(roi_cls_logit[0, 0, :])
            car_probability = softmax_ret[0]
            car_probability_list.append(car_probability.cpu().numpy())

            res = pred_dicts[0]["pred_boxes"][:, -1]
            res_mask = res > 6.29
            if res_mask.any():
                print(res, num)
            num = num + 1

    logger.info('Demo done.')
    return pred_dicts_list, batch_dict_list, car_probability_list

#
if __name__ == '__main__':
    exp_name = 'ga_eval-fixCarPosi'
    writer = SummaryWriter('./GeneticAlgorithm/log_tensorboard/log_eval/' + exp_name)
    bin_dir_base = r'/data/dataset_wujunqi/GeneticAlgorithm/GeneticAlgorithm-v2-20220928-104232/iter'
    bin_dir_base = r'/data/dataset_wujunqi/GeneticAlgorithm/GeneticAlgorithm-v2-20220928-203425-fixCarPosi-RenderPosi/iter'
    logger = initial(exp_name)

    Genations = 92
    car_pro_mean_list = []
    for iter in range(0, Genations):
        bin_dir = bin_dir_base + str(iter)
        _, _, car_probability_list = network_forward(logger, bin_dir)
        car_pro_mean = np.array(car_probability_list).mean()
        car_pro_std = np.array(car_probability_list).std()
        car_pro_mean_list.append(car_pro_mean)
        logger.info('iter: {}. 分类为车的平均概率为： {:.3f}%. STD为：{:.3f}'.format(iter, car_pro_mean * 100, car_pro_std))
        writer.add_scalar('car_probability/car_p_mean', car_pro_mean, iter)
        writer.add_scalar('car_probability/car_p_std', car_pro_mean, iter)

        print(car_pro_mean_list)