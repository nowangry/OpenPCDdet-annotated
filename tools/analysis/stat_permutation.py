import numpy as np
# from det3d.torchie.apis.train import *
# from configs.adv.voxelization_setup import *
from tools.Lib.loss_utils import *
import torch
import tqdm


def is_non_zore_points(point):
    for i in range(point.shape[0]):
        if np.abs(point[i]) > 0.000001:
            return True
    return False

def show_L_norm_logger(logger, dist_l0_list, dist_l1_list, dist_l2_list, dist_l_inf_list, num_points):
    logger.info("修改比例：{:.2f}%; dist_l0: mean={:.2f}, median={:.2f}, std={:.2f}".format(np.array(
        dist_l0_list).mean() / np.array(
        num_points).mean() * 100,
                                                                                           np.array(
                                                                                               dist_l0_list).mean(),
                                                                                           np.array(dist_l0_list).std(),
                                                                                           np.median(
                                                                                               np.array(dist_l0_list)),
                                                                                           ))
    logger.info("dist_l1: mean={:.2f}, median={:.2f}, std={:.2f}".format(np.array(dist_l1_list).mean(),
                                                                         np.median(np.array(dist_l1_list)),
                                                                         np.array(dist_l1_list).std()))
    logger.info("dist_l2: mean={:.2f}, median={:.2f}, std={:.2f}".format(np.array(dist_l2_list).mean(),
                                                                         np.median(np.array(dist_l2_list)),
                                                                         np.array(dist_l2_list).std()))
    logger.info("dist_l_inf: mean={:.2f}, median={:.2f}, var={:.2f}".format(np.array(dist_l_inf_list).mean(),
                                                                            np.median(np.array(dist_l_inf_list)),
                                                                            np.array(dist_l_inf_list).std()))
    logger.info("==============================================================\n")

def show_L_norm(dist_l0_list, dist_l1_list, dist_l2_list, dist_l_inf_list, num_points):
    print("修改比例：{:.2f}%; dist_l0: mean={:.2f}, median={:.2f}, std={:.2f}".format(np.array(
        dist_l0_list).mean() / np.array(
        num_points).mean() * 100,
                                                                                     np.array(dist_l0_list).mean(),
                                                                                     np.array(dist_l0_list).std(),
                                                                                     np.median(np.array(dist_l0_list)),
                                                                                     ))
    print("dist_l1: mean={:.2f}, median={:.2f}, std={:.2f}".format(np.array(dist_l1_list).mean(),
                                                                   np.median(np.array(dist_l1_list)),
                                                                   np.array(dist_l1_list).std()))
    print("dist_l2: mean={:.2f}, median={:.2f}, std={:.2f}".format(np.array(dist_l2_list).mean(),
                                                                   np.median(np.array(dist_l2_list)),
                                                                   np.array(dist_l2_list).std()))
    print("dist_l_inf: mean={:.2f}, median={:.2f}, var={:.2f}".format(np.array(dist_l_inf_list).mean(),
                                                                      np.median(np.array(dist_l_inf_list)),
                                                                      np.array(dist_l_inf_list).std()))
    print("==============================================================\n")


def stat_L_permutations(pc_dir_list):
    for pc_dir in pc_dir_list:
        print(' processing dir: {}'.format(pc_dir))
        num_points = []
        dist_l0_list = []
        dist_l1_list = []
        dist_l2_list = []
        dist_l_inf_list = []
        cnt = 0
        for root, _, filenames in os.walk(pc_dir):
            for filename in filenames:
                if 'innocent' in filename:
                    continue
                # print(' processing {}th file: {}'.format(cnt, filename))
                # cnt += 1
                # if 'a860c02c06e54d9dbe83ce7b694f6c17' not in filename:
                #     continue
                path_adv = os.path.join(root, filename)
                points_adv = np.fromfile(path_adv, dtype=np.float32).reshape(-1, 5)
                if 'PGD' in root or 'MI_FGSM' in root:
                    path_innocent = os.path.join(root, filename.replace('.bin', '-innocent_ori.bin'))
                else:
                    path_innocent = os.path.join(root, filename.replace('.bin', '-innocent.bin'))
                points_innocent = np.fromfile(path_innocent, dtype=np.float32).reshape(-1, 5)

                dist_points = points_adv[:, :3] - points_innocent[:, :3]
                dist_points_l0 = np.abs(dist_points).sum(1)
                dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
                # dist_l0 = 0
                # for i in range(dist_points.shape[0]):
                #     if not is_non_zore_points(dist_points[i, :3]):
                #         dist_l0 += 1
                #     else:
                #         a = 1
                dist_points = dist_points.reshape(-1)
                # dist_l0 = np.linalg.norm(dist_points, ord=0, axis=None)
                dist_l1 = np.linalg.norm(dist_points, ord=1, axis=None)
                dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
                dist_l_inf = np.linalg.norm(dist_points, ord=np.inf, axis=None)
                num_points.append(points_innocent.shape[0])
                # if not dist_l0 / 3 == points_innocent.shape[0]:
                #     a = 1
                dist_l0_list.append(dist_l0)
                dist_l1_list.append(dist_l1)
                dist_l2_list.append(dist_l2)
                dist_l_inf_list.append(dist_l_inf)
        show_L_norm(dist_l0_list, dist_l1_list, dist_l2_list, dist_l_inf_list, num_points)
    print('Done.')


from tools.Lib.chamfer_distance import *
from tools.Lib.hausdorff_distance import *

def stat_L_permutations_all_logger(pc_dir_list, **kwargs):
    if 'cfg' in kwargs:
        cfg = kwargs['cfg']
        logger = cfg.logger
    if 'cfg' in kwargs and cfg.dataset_type == "WaymoDataset":
        collected_token = cfg.collected_token
        logger.info('len(collected_token): {}'.format(len(collected_token)))
    for pc_dir in pc_dir_list:
        logger.info(' processing dir: {}'.format(pc_dir))
        num_points = []
        dist_l0_list = []
        dist_l1_list = []
        dist_l2_list = []
        dist_l_inf_list = []
        dist_chamfer_list = []
        dist_hausdorff_list = []
        cnt = 0
        token_set = set()
        for root, _, filenames in os.walk(pc_dir):
            for filename in filenames:
                token = filename.split('.')[0].split('-')[0]
                if token in token_set:
                    continue
                if 'cfg' in kwargs and cfg.dataset_type == "WaymoDataset" and not (token + '.pkl') in collected_token:
                    continue
                token_set.add(token)

                # if 'innocent' in filename:
                #     continue
                if cnt % 1000 == 0:
                    print(' processing {}th file: {}'.format(cnt, filename))
                cnt += 1
                # if 'a860c02c06e54d9dbe83ce7b694f6c17' not in filename:
                #     continue
                if 'Waymo' in root:
                    token = token + '.pkl'
                    if 'centerpoint-adv-pillar' in root:
                        point_dim = 5
                    else:
                        point_dim = 6
                elif 'KITTI' in root:
                    point_dim = 4
                else:
                    point_dim = 5
                if 'PGD' in root or 'MI' in root or '_iter' in root:
                    path_innocent = os.path.join(root, token + '-innocent_ori.bin')
                else:
                    path_innocent = os.path.join(root, token + '-innocent.bin')
                path_adv = os.path.join(root, token + '.bin')
                points_innocent = np.fromfile(path_innocent, dtype=np.float32).reshape(-1, point_dim)
                points_adv = np.fromfile(path_adv, dtype=np.float32).reshape(-1, point_dim)
                dist_points = points_adv[:, :3] - points_innocent[:, :3]
                dist_points_l0 = np.abs(dist_points).sum(1)
                dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)

                dist_points = dist_points.reshape(-1)
                dist_l1 = np.linalg.norm(dist_points, ord=1, axis=None)
                dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
                dist_l_inf = np.linalg.norm(dist_points, ord=np.inf, axis=None)
                num_points.append(points_innocent.shape[0])

                dist_l0_list.append(dist_l0)
                dist_l1_list.append(dist_l1)
                dist_l2_list.append(dist_l2)
                dist_l_inf_list.append(dist_l_inf)

                # dist_chamfer = chamfer_distance_kdTree_open3d(points_adv[:, :3], points_innocent[:, :3])
                dist_chamfer = chamfer_distance_KDTree_scipy(points_adv[:, :3], points_innocent[:, :3])
                dist_chamfer_list.append(dist_chamfer)
                # dist_hausdorff = hausdorff_distance_kdTree_open3d(points_adv[:, :3], points_innocent[:, :3])
                dist_hausdorff = hausdorff_distance_KDTree_scipy(points_adv[:, :3], points_innocent[:, :3])
                dist_hausdorff_list.append(dist_hausdorff)
                # if cnt >= 3:
                #     break

        logger.info(pc_dir)
        show_L_norm_logger(logger, dist_l0_list, dist_l1_list, dist_l2_list, dist_l_inf_list, num_points)
        logger.info("dist_chamfer_list: mean={:.5f}, median={:.5f}, var={:.5f}".format(np.array(dist_chamfer_list).mean(),
                                                                                 np.median(np.array(dist_chamfer_list)),
                                                                                 np.array(dist_chamfer_list).std()))
        logger.info("dist_hausdorff_list: mean={:.5f}, median={:.5f}, var={:.5f}".format(np.array(dist_hausdorff_list).mean(),
                                                                                   np.median(
                                                                                       np.array(dist_hausdorff_list)),
                                                                                   np.array(dist_hausdorff_list).std()))
        logger.info("==============================================================\n")
    logger.info('Done.')
    logger.info(pc_dir_list)


def stat_L_permutations_all(pc_dir_list, **kwargs):
    if 'cfg' in kwargs:
        cfg = kwargs['cfg']
    if 'cfg' in kwargs and cfg.dataset_type == "WaymoDataset":
        collected_token = cfg.collected_token
        print('len(collected_token): {}'.format(len(collected_token)))
    for pc_dir in pc_dir_list:
        print(' processing dir: {}'.format(pc_dir))
        num_points = []
        dist_l0_list = []
        dist_l1_list = []
        dist_l2_list = []
        dist_l_inf_list = []
        dist_chamfer_list = []
        dist_hausdorff_list = []
        cnt = 0
        token_set = set()
        for root, _, filenames in os.walk(pc_dir):
            for filename in filenames:
                token = filename.split('.')[0].split('-')[0]
                if token in token_set:
                    continue
                if 'cfg' in kwargs and cfg.dataset_type == "WaymoDataset" and not (token + '.pkl') in collected_token:
                    continue
                token_set.add(token)

                # if 'innocent' in filename:
                #     continue
                if cnt % 1000 == 0:
                    print(' processing {}th file: {}'.format(cnt, filename))
                cnt += 1
                # if 'a860c02c06e54d9dbe83ce7b694f6c17' not in filename:
                #     continue
                if 'Waymo' in root:
                    token = token + '.pkl'
                    if 'centerpoint-adv-pillar' in root:
                        point_dim = 5
                    else:
                        point_dim = 6
                elif 'KITTI' in root:
                    point_dim = 4
                else:
                    point_dim = 5
                if 'PGD' in root or 'MI' in root or '_iter' in root:
                    path_innocent = os.path.join(root, token + '-innocent_ori.bin')
                else:
                    path_innocent = os.path.join(root, token + '-innocent.bin')
                path_adv = os.path.join(root, token + '.bin')
                points_innocent = np.fromfile(path_innocent, dtype=np.float32).reshape(-1, point_dim)
                points_adv = np.fromfile(path_adv, dtype=np.float32).reshape(-1, point_dim)
                dist_points = points_adv[:, :3] - points_innocent[:, :3]
                dist_points_l0 = np.abs(dist_points).sum(1)
                dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)

                dist_points = dist_points.reshape(-1)
                dist_l1 = np.linalg.norm(dist_points, ord=1, axis=None)
                dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
                dist_l_inf = np.linalg.norm(dist_points, ord=np.inf, axis=None)
                num_points.append(points_innocent.shape[0])

                dist_l0_list.append(dist_l0)
                dist_l1_list.append(dist_l1)
                dist_l2_list.append(dist_l2)
                dist_l_inf_list.append(dist_l_inf)

                # dist_chamfer = chamfer_distance_kdTree_open3d(points_adv[:, :3], points_innocent[:, :3])
                dist_chamfer = chamfer_distance_KDTree_scipy(points_adv[:, :3], points_innocent[:, :3])
                dist_chamfer_list.append(dist_chamfer)
                # dist_hausdorff = hausdorff_distance_kdTree_open3d(points_adv[:, :3], points_innocent[:, :3])
                dist_hausdorff = hausdorff_distance_KDTree_scipy(points_adv[:, :3], points_innocent[:, :3])
                dist_hausdorff_list.append(dist_hausdorff)
                # if cnt >= 3:
                #     break

        print(pc_dir)
        show_L_norm(dist_l0_list, dist_l1_list, dist_l2_list, dist_l_inf_list, num_points)
        print("dist_chamfer_list: mean={:.5f}, median={:.5f}, var={:.5f}".format(np.array(dist_chamfer_list).mean(),
                                                                                 np.median(np.array(dist_chamfer_list)),
                                                                                 np.array(dist_chamfer_list).std()))
        print("dist_hausdorff_list: mean={:.5f}, median={:.5f}, var={:.5f}".format(np.array(dist_hausdorff_list).mean(),
                                                                                   np.median(
                                                                                       np.array(dist_hausdorff_list)),
                                                                                   np.array(dist_hausdorff_list).std()))
        print("==============================================================\n")
    print('Done.')
    print(pc_dir_list)


def stat_chamfer_permutations(pc_dir_list):
    for dir in pc_dir_list:
        print(' processing dir: {}'.format(dir))
        num_points = []
        dist_l0_list = []
        dist_l1_list = []
        dist_l2_list = []
        dist_l_inf_list = []
        dist_chamfer_list = []
        dist_hausdorff_list = []
        cnt = 0
        for root, _, filenames in os.walk(dir):
            for filename in filenames:
                if 'innocent' in filename:
                    continue
                print(' processing {}th file: {}'.format(cnt, filename))
                cnt += 1
                # if 'a860c02c06e54d9dbe83ce7b694f6c17' not in filename:
                #     continue
                path_adv = os.path.join(root, filename)
                points_adv = np.fromfile(path_adv, dtype=np.float32).reshape(-1, 5)
                if 'PGD' in root:
                    path_innocent = os.path.join(root, filename.replace('.bin', '-innocent_gradient.bin'))
                else:
                    path_innocent = os.path.join(root, filename.replace('.bin', '-innocent.bin'))
                points_innocent = np.fromfile(path_innocent, dtype=np.float32).reshape(-1, 5)

                dist_points = points_adv - points_innocent
                dist_points = dist_points.reshape(-1)
                points_shape = dist_points.shape[0]
                dist_l0 = np.linalg.norm(dist_points, ord=0, axis=None)
                dist_l1 = np.linalg.norm(dist_points, ord=1, axis=None)
                dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
                dist_l_inf = np.linalg.norm(dist_points, ord=np.inf, axis=None)

                # 计算chamfer距离
                # modified_points = []
                # dist_points = dist_points.reshape((-1, 5))
                # for i in range(dist_points.shape[0]):
                #     if np.abs(dist_points[i, :3]).sum() > 0.00001:
                #         modified_points.append(dist_points[i, :3])
                # modified_points = np.array(modified_points).reshape((-1, 3))
                dist_chamfer = chamfer_loss_v1_max(torch.tensor(points_adv[:, :3]),
                                                   torch.tensor(points_innocent[:, :3])).numpy()

                # dist_hausdorff = hausdorff_loss_v1(torch.tensor(modified_points[:, :3]), torch.tensor(points_innocent[:, :3])).numpy()

                num_points.append(points_innocent.shape[0])
                dist_l0_list.append(dist_l0)
                dist_l1_list.append(dist_l1)
                dist_l2_list.append(dist_l2)
                dist_l_inf_list.append(dist_l_inf)
                dist_chamfer_list.append(dist_chamfer)
                # dist_hausdorff_list.append(dist_hausdorff)
                if cnt % 100 == 0:
                    break

        show_L_norm(dist_l0_list, dist_l1_list, dist_l2_list, dist_l_inf_list, num_points)
        print("dist_chamfer_list: mean={:.2f}, median={:.2f}, var={:.2f}".format(np.array(dist_chamfer_list).mean(),
                                                                                 np.median(np.array(dist_chamfer_list)),
                                                                                 np.array(dist_chamfer_list).std()))
        print("==============================================================\n")
    print('Done.')


def stat_hausdorff_permutations(pc_dir_list):
    for dir in pc_dir_list:
        print(' processing dir: {}'.format(dir))
        num_points = []
        dist_l0_list = []
        dist_l1_list = []
        dist_l2_list = []
        dist_l_inf_list = []
        dist_chamfer_list = []
        dist_hausdorff_list = []
        cnt = 0
        for root, _, filenames in os.walk(dir):
            for filename in filenames:
                if 'innocent' in filename:
                    continue
                print(' processing {}th file: {}'.format(cnt, filename))
                cnt += 1
                # if 'a860c02c06e54d9dbe83ce7b694f6c17' not in filename:
                #     continue
                path_adv = os.path.join(root, filename)
                points_adv = np.fromfile(path_adv, dtype=np.float32).reshape(-1, 5)
                if 'PGD' in root:
                    path_innocent = os.path.join(root, filename.replace('.bin', '-innocent_gradient.bin'))
                else:
                    path_innocent = os.path.join(root, filename.replace('.bin', '-innocent.bin'))
                points_innocent = np.fromfile(path_innocent, dtype=np.float32).reshape(-1, 5)

                dist_points = points_adv - points_innocent
                dist_points = dist_points.reshape(-1)
                points_shape = dist_points.shape[0]
                dist_l0 = np.linalg.norm(dist_points, ord=0, axis=None)
                dist_l1 = np.linalg.norm(dist_points, ord=1, axis=None)
                dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
                dist_l_inf = np.linalg.norm(dist_points, ord=np.inf, axis=None)

                # 计算chamfer距离
                # modified_points = []
                # dist_points = dist_points.reshape((-1, 5))
                # for i in range(dist_points.shape[0]):
                #     if np.abs(dist_points[i, :3]).sum() > 0.00001:
                #         modified_points.append(dist_points[i, :3])
                # modified_points = np.array(modified_points).reshape((-1, 3))
                # dist_chamfer = chamfer_loss_v1_max(torch.tensor(modified_points[:, :3]), torch.tensor(points_innocent[:, :3])).numpy()

                dist_hausdorff = hausdorff_loss_v1(torch.tensor(points_adv[:, :3]),
                                                   torch.tensor(points_innocent[:, :3])).numpy()

                num_points.append(points_innocent.shape[0])
                dist_l0_list.append(dist_l0)
                dist_l1_list.append(dist_l1)
                dist_l2_list.append(dist_l2)
                dist_l_inf_list.append(dist_l_inf)
                # dist_chamfer_list.append(dist_chamfer)
                dist_hausdorff_list.append(dist_hausdorff)
                if cnt % 100 == 0:
                    break

        show_L_norm(dist_l0_list, dist_l1_list, dist_l2_list, dist_l_inf_list, num_points)
        print("dist_hausdorff_list: mean={:.2f}, median={:.2f}, var={:.2f}".format(np.array(dist_hausdorff_list).mean(),
                                                                                   np.median(
                                                                                       np.array(dist_hausdorff_list)),
                                                                                   np.array(dist_hausdorff_list).std()))
    print('Done.')


if __name__ == "__main__":
    pc_dir_grad = '/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.5-num_steps_1-Lambda_0.0001-attach_rate_0.04-strategy_gradient_sorted_not_add/'
    pc_dir_grad_leave_one = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted-not_add-voxelize_permutation/'
    pc_dir_grad_leave_one_limitH = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted-not_add-voxelize_permutation-limitH/'
    pc_dir_FGSM = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/FGSM_CoorAdjust/Epsilon_1.0/'
    pc_dir_PGD = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/PGD_CoorAdjust/eps_0.1-eps_iter_0.02-num_steps_10-/'
    pc_dir_IOU = r'/Data4T/Outputs/centerpoint-adv-pillar/IOU_iter/eps_0.1-eps_iter_0.02-num_steps_10-Lambda_0.1-iou_0.1-score_0.1/'
    pc_dir_RoadAdv = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/RoadAdv/Epsilon_0.2/'
    pc_dir_MI_FGSM = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/MI_FGSM/eps_0.1-eps_iter_0.02-num_steps_10-decay_1.0-L_norm_L2/'
    pc_dir_MI_FGSM = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/MI_FGSM/eps_0.2-eps_iter_0.03-num_steps_10-decay_1.0-L_norm_L2/'
    pc_dir_AdaptiveEPS = r'/Data4T/Outputs/centerpoint-adv/AdaptiveEPS/strategy_base-eps_0.4-iter_eps_0.5-num_steps_1-attach_rate_0.5/'
    pc_dir_ball_density = r'/Data4T/Outputs/centerpoint-adv/SpareAdv_Attach/strategy_density_attention_norm-eps_Nan-iter_eps_0.5-num_steps_1-attach_rate_0.1/'

    pc_dir_FGSM_pillar = r'/Data4T/Outputs/centerpoint-adv-pillar/FGSM_CoorAdjust/Epsilon_0.2/'
    pc_dir_MI_AdaptiveEPS_pillar = r'/Data4T/Outputs/centerpoint-adv-pillar/AdaptiveEPS/strategy_MI-eps_0.5-eps_ratio_0.1-num_steps_10-attach_rate_0.1/'
    pc_dir_list = [pc_dir_grad, pc_dir_FGSM, pc_dir_IOU, pc_dir_grad_leave_one]
    # # pc_dir_list = [pc_dir_FGSM]
    # pc_dir_list = [pc_dir_grad_leave_one]
    pc_dir_list = [pc_dir_FGSM]
    pc_dir_list = [pc_dir_AdaptiveEPS]
    pc_dir_list = [pc_dir_MI_FGSM]
    pc_dir_list = [
        r'/Data4T/Outputs/centerpoint-adv/IOU_iter/eps_0.1-eps_iter_0.02-num_steps_10-Lambda_0.1-iou_0.1-score_0.1/',
        r'/Data4T/Outputs/centerpoint-adv/IOU_iter/eps_0.2-eps_iter_0.03-num_steps_10-Lambda_0.1-iou_0.1-score_0.1/',
        r'/Data4T/Outputs/centerpoint-adv/IOU_iter/eps_0.2-eps_iter_0.03-num_steps_20-Lambda_0.1-iou_0.1-score_0.1/',
        r'/Data4T/Outputs/centerpoint-adv/IOU_iter/eps_0.5-eps_iter_0.08-num_steps_10-Lambda_0.1-iou_0.1-score_0.1/',
    ]

    pc_dir_list = [
        r'/Data4T-1/Outputs/Waymo/centerpoint-adv/FGSM_CoorAdjust/Epsilon_0.2/',
    ]

    pc_dir_list = [
        # r'/Data4T-1/Outputs/Waymo/centerpoint-adv/FGSM_CoorAdjust/Epsilon_0.2/',
        r'/Data4T-1/Outputs/Waymo/centerpoint-adv-pillar/FGSM_CoorAdjust/Epsilon_0.2/',
    ]

    dir_save_origin_points = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-origin_points/'  # 干净样本，点云格式，未经过预处理
    dir_save_innocent = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'  # 干净样本，点云格式，经过网络预处理的体素化，未做重新体素化
    dir_save_adv = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2/'  # 对抗样本，点云格式，经过网络预处理的体素化，已做重新体素化
    dir_list = [dir_save_origin_points, dir_save_innocent, dir_save_adv]

    # stat_L_permutations(pc_dir_list)
    stat_L_permutations_all(pc_dir_list)
    # stat_chamfer_permutations(pc_dir_list)
    # stat_hausdorff_permutations(pc_dir_list)
