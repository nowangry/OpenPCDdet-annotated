# from det3d.torchie.apis.train import *
from configs.adv.voxelization_setup import *
from tools.Lib.loss_utils import *
import torch
from tools.analysis.histogram import *


def is_non_zore_points(point):
    for i in range(point.shape[0]):
        if np.abs(point[i]) > 0.000001:
            return True
    return False


def show_L_norm(dist_l0_list, dist_l1_list, dist_l2_list, dist_l_inf_list, num_points):
    print("dist_l0: mean={:.2f}, median={:.2f}, std={:.2f}, 修改比例：{:.2f}%".format(np.array(dist_l0_list).mean(),
                                                                                     np.array(dist_l0_list).std(),
                                                                                     np.median(np.array(dist_l0_list)),
                                                                                     np.array(
                                                                                         dist_l0_list).mean() / np.array(
                                                                                         num_points).mean() * 100))
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


def stat_gradient_L1(pc_dir_list, attach_rate=0.1, save_dir=r'./hist'):
    for dir in pc_dir_list:
        print(' processing dir: {}'.format(dir))
        time_stamps_array = np.array([])
        cnt = 0
        for root, _, filenames in os.walk(dir):
            for filename in filenames:
                if '-innocent_gradient.bin' not in filename:
                    continue
                print(' processing {}th filename: {}'.format(cnt, filename))
                cnt += 1
                path_innocent = os.path.join(root, filename.replace('-innocent_gradient.bin', '-innocent.bin'))
                points_innocent = np.fromfile(path_innocent, dtype=np.float32).reshape(-1, 5)
                path_grad = os.path.join(root, filename)
                grad_points = np.fromfile(path_grad, dtype=np.float32).reshape(-1, 5)
                grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
                for j in range(grad_points.shape[0]):
                    grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
                grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
                                                  descending=True).numpy()  # 梯度排序
                attach_n = int(grad_points.shape[0] * attach_rate)
                time_stamp = points_innocent[grad_points_order[:attach_n], 3]

                time_stamp = time_stamp.reshape(-1)
                time_stamps_array = np.hstack((time_stamps_array, time_stamp))
                print(time_stamps_array.shape)
                MyHist(time_stamp, title_name='top 10% intensity stamps', bins=100,
                       save_path=os.path.join(save_dir, filename.replace('-innocent_gradient.bin', '.png')))

        MyHist(time_stamps_array, title_name='top 10% intensity stamps',
               save_path=os.path.join(save_dir, '0000-sum.png'))

    print('Done.')


if __name__ == "__main__":
    pc_dir_grad = '/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.5-strategy_gradient_sorted_not_add/'
    pc_dir_grad_leave_one = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted-not_add-voxelize_permutation/'
    pc_dir_grad_leave_one_limitH = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted-not_add-voxelize_permutation-limitH/'
    pc_dir_FGSM = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/FGSM_CoorAdjust/Epsilon_0.1/'
    pc_dir_PGD = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/PGD_CoorAdjust/eps_0.1-eps_iter_0.02-num_steps_10-/'
    pc_dir_IOU = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/IOU_CoorAdjust/eps_0.1-eps_iter_0.1-num_steps_1-Lambda_0.1-iou_0.1-score_0.1/'
    pc_dir_RoadAdv = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/RoadAdv/Epsilon_0.1/'
    pc_dir_MI_FGSM = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/MI_FGSM/eps_0.1-eps_iter_0.02-num_steps_10-decay_1.0-L_norm_L1/'

    pc_dir_list = [pc_dir_grad, pc_dir_FGSM, pc_dir_IOU, pc_dir_grad_leave_one]
    # # pc_dir_list = [pc_dir_FGSM]
    # pc_dir_list = [pc_dir_grad_leave_one]
    pc_dir_list = [pc_dir_FGSM]

    dir_save_origin_points = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-origin_points/'  # 干净样本，点云格式，未经过预处理
    dir_save_innocent = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'  # 干净样本，点云格式，经过网络预处理的体素化，未做重新体素化
    dir_save_adv = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2/'  # 对抗样本，点云格式，经过网络预处理的体素化，已做重新体素化
    dir_list = [dir_save_origin_points, dir_save_innocent, dir_save_adv]

    save_dir = r'./save/hist/time_stamps/'
    save_dir = r'./save/hist/intensity/'
    os.makedirs(save_dir, exist_ok=True)
    stat_gradient_L1(pc_dir_list, save_dir=save_dir)
    # stat_chamfer_permutations(pc_dir_list)
    # stat_hausdorff_permutations(pc_dir_list)
