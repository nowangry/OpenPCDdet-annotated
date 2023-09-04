from det3d.torchie.apis.train import *
from configs.adv.voxelization_setup import *
from configs.adv.voxelization_setup import *
from configs.adv_pillar.pillar_voxelation_setup import *


def stat_points_1():
    dir_save_origin_points = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-origin_points/'  # 干净样本，点云格式，未经过预处理
    dir_save_innocent = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'  # 干净样本，点云格式，经过网络预处理的体素化，未做重新体素化
    dir_save_adv = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2/'  # 对抗样本，点云格式，经过网络预处理的体素化，已做重新体素化
    dir_list = [dir_save_origin_points, dir_save_innocent, dir_save_adv]

    for dir in dir_list:
        print(' processing dir: {}'.format(dir))
        points_n_list = []
        voxels_n_list = []
        points_n_in_voxels_list = []
        for root, _, filenames in os.walk(dir):
            for filename in filenames:
                if 'a860c02c06e54d9dbe83ce7b694f6c17' not in filename:
                    continue
                path = os.path.join(root, filename)
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
                voxels, coordinates, num_points = Voxelize.voxel_generator.generate(
                    points, max_voxels=max_voxels
                )

                points_n_list.append(points.shape[0])
                voxels_n_list.append(voxels.shape[0])
                points_n_in_voxels_list.append(num_points.sum())

        print("points_n: mean={}, std={}".format(np.array(points_n_list).mean(), np.array(points_n_list).std()))
        print("voxels_n: mean={}, std={}".format(np.array(voxels_n_list).mean(), np.array(voxels_n_list).std()))
        print("points_n_in_voxels: mean={}, var={}".format(np.array(points_n_in_voxels_list).mean(),
                                                           np.array(points_n_in_voxels_list).std()))
        print("==============================================================\n")
    print('Done.')


def stat_points_2():
    dir_save_origin_points = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-origin_points/'  # 干净样本，点云格式，未经过预处理
    dir_save_innocent = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'  # 干净样本，点云格式，经过网络预处理的体素化，未做重新体素化
    dir_save_adv = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2/'  # 对抗样本，点云格式，经过网络预处理的体素化，已做重新体素化
    dir_save_adv_reVoxelize = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2-reVoxelize/'  # 对抗样本，点云格式，经过网络预处理的体素化，已做重新体素化
    # dir_list = [dir_save_origin_points, dir_save_innocent, dir_save_adv, dir_save_adv_reVoxelize]
    dir_list = [dir_save_origin_points]

    voxel_range = voxelization_cfg.cfg['range']  # [-54, -54, -5.0, 54, 54, 3.0]
    for dir in dir_list:
        print(' processing dir: {}'.format(dir))
        points_n_in_voxelization_range_list = []
        points_n_out_voxelization_range_list = []
        for root, _, filenames in os.walk(dir):
            for filename in filenames:

                # if 'a860c02c06e54d9dbe83ce7b694f6c17' not in filename:
                #     continue
                path = os.path.join(root, filename)
                print('processing {}'.format(path))
                points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)

                points_n_in_voxelization_range = 0
                points_n_out_voxelization_range = 0
                for i in range(points.shape[0]):
                    if voxel_range[0] <= points[i][0] and points[i][0] <= voxel_range[3] and \
                            voxel_range[1] <= points[i][1] and points[i][1] <= voxel_range[4] and \
                            voxel_range[2] <= points[i][2] and points[i][2] <= voxel_range[5]:
                        points_n_in_voxelization_range += 1
                    else:
                        points_n_out_voxelization_range += 1

                points_n_in_voxelization_range_list.append(points_n_in_voxelization_range)
                points_n_out_voxelization_range_list.append(points_n_out_voxelization_range)
                print(
                    "points_n_in_voxels = {}, points_n_out_voxels = {}. sum = {}".format(points_n_in_voxelization_range,
                                                                                         points_n_out_voxelization_range,
                                                                                         points_n_in_voxelization_range + points_n_out_voxelization_range))

                # points_n_in_revoxels = 0
                # points_n_out_revoxels = 0
                # voxels, coordinates, num_points = Voxelize.voxel_generator.generate(
                #     points, max_voxels=max_voxels
                # )
                # points = voxel_to_point_numba(voxels, num_points)
                # for i in range(points.shape[0]):
                #     if voxel_range[0] <= points[i][0] and points[i][0] <= voxel_range[3] and \
                #             voxel_range[1] <= points[i][1] and points[i][1] <= voxel_range[4] and \
                #             voxel_range[2] <= points[i][2] and points[i][2] <= voxel_range[5]:
                #         points_n_in_revoxels += 1
                #     else:
                #         points_n_out_revoxels += 1
                # # break

                # print("points_n_in_revoxels = {}, points_n_out_revoxels = {}. sum = {}".format(points_n_in_revoxels,
                #                                                                                points_n_out_revoxels,
                #                                                                                points_n_in_revoxels + points_n_out_revoxels))
                #
            points_n_in_voxelization_range_array = np.array(points_n_in_voxelization_range_list)
            points_n_out_voxelization_range_array = np.array(points_n_out_voxelization_range_list)
            print("points_n_in_voxelization_range_list: mean = {}, std = {}".format(
                points_n_in_voxelization_range_array.mean(),
                points_n_in_voxelization_range_array.std()))
            print("points_n_out_voxelization_range_array: mean = {}, std = {}".format(
                points_n_out_voxelization_range_array.mean(),
                points_n_out_voxelization_range_array.std()))
            # points_n_in_voxelization_range_list: mean = 246846.11145510836, std = 10407.957621817393
            # points_n_out_voxelization_range_array: mean = 21132.346749226006, std = 8792.659725591775


if __name__ == "__main__":
    stat_points_2()
