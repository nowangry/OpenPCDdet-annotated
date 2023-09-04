# from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import pickle
import json
import pathlib
import platform

plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

from tools.analysis.visual_utily import *

# from configs.v1.0-mini.adv.voxelization_setup import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--path', help='path to visualization file', type=str, default=None)
    parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.3)
    args = parser.parse_args()

    rgb_colors = ncolors(10)

    # pc_dir = r'D:\data\adv_examples\IOU-score-target\iter_eps_0.12-num_steps_20-Lambda_0.1-points_clouds_visualization'
    # pkl_path = r'../../work_dirs/ADV/IOU/eps_iter_0.06-num_steps_20-Lambda_0.1-fixLoss-logIOU-target0/prediction.pkl'
    # json_path = r'../../work_dirs/ADV/IOU/eps_iter_0.06-num_steps_20-Lambda_0.1-fixLoss-logIOU-target0/infos_train_10sweeps_withvelo_filter_True.json'
    #
    # pc_dir = r'D:\data\adv_examples\re-Voxelize\PGD\eps_1.6-iter_eps_0.12-num_steps_10-points_clouds_visualization'
    # pkl_path = r'../../work_dirs/ADV/PGD_CoorAdjust/eps_1.6-eps_iter_0.12-num_steps_10/prediction.pkl'
    # json_path = r'../../work_dirs/ADV/PGD_CoorAdjust/eps_1.6-eps_iter_0.12-num_steps_10/infos_train_10sweeps_withvelo_filter_True.json'

    # pc_dir = r'D:\data\adv_examples\re-Voxelize\IOU\iter_eps_0.1-num_steps_1-Lambda_0.1-points_clouds_visualization'
    # pkl_path = r'../../work_dirs\ADV_reVoxelize\IOU\eps_iter_0.1-num_steps_1-Lambda_0.1/prediction.pkl'
    # json_path = r'../../work_dirs\ADV_reVoxelize\IOU\eps_iter_0.1-num_steps_1-Lambda_0.1/infos_train_10sweeps_withvelo_filter_True.json'

    # pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-origin_points/'
    # work_dir = r'../../work_dirs/v1.0-mini/ADV_reVoxelize/test_reVoxelize-backup/innocent/'

    pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'
    work_dir = '../../work_dirs/v1.0-mini/ADV_reVoxelize/test_reVoxelize-backup/innocent/'
    # #
    # pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2/'
    # pkl_path = '../../work_dirs/v1.0-mini/ADV_reVoxelize/test_reVoxelize-backup/adv/prediction.pkl'
    # json_path = '../../work_dirs/v1.0-mini/ADV_reVoxelize/test_reVoxelize-backup/adv/infos_train_10sweeps_withvelo_filter_True.json'

    # pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted_not_add/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/PointAttach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted_not_add/'

    pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/FGSM_CoorAdjust/Epsilon_0.5/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/FGSM/Epsilon_0.5/'
    #
    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/PGD_CoorAdjust/eps_0.1-eps_iter_0.02-num_steps_10/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/PGD_CoorAdjust/eps_0.1-eps_iter_0.02-num_steps_10/'
    # #
    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/IOU_CoorAdjust/eps_0.1-eps_iter_0.1-num_steps_1-Lambda_0.1-iou_0.1-score_0.1/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/IOU/eps_0.1-eps_iter_0.1-num_steps_1-Lambda_0.1-iou_0.1-score_0.1/'
    #
    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted-not_add-voxelize_permutation/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/PointAttach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted-not_add-voxelize_permutation/'
    #
    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted-not_add-voxelize_permutation-limitH/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/PointAttach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted-not_add-voxelize_permutation-limitH/'
    #
    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/RoadAdv/Epsilon_0.1/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/RoadAdv/Epsilon_0.1/'

    pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/MI_FGSM/eps_0.2-eps_iter_0.03-num_steps_10-decay_1.0-L_norm_L2/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/MI_FGSM/eps_0.2-eps_iter_0.03-num_steps_10-decay_1.0-L_norm_L2/'
    #
    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.5-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted_not_add/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/PointAttach/eps_Nan-iter_eps_0.5-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted_not_add/'

    # pc_dir = r'/Data4T/Outputs/centerpoint-adv/Point_Attach/strategy_random-eps_Nan-iter_eps_0.1-num_steps_1-Lambda_0.0001-attach_rate_0.1/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/PointAttach/strategy_random-eps_Nan-iter_eps_0.1-num_steps_1-Lambda_0.0001-attach_rate_0.1/'
    #
    pc_dir = r'/Data4T/Outputs/centerpoint-adv/AdaptiveEPS/strategy_base-eps_0.5-iter_eps_0.5-num_steps_1-attach_rate_0.5/'
    work_dirs = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/AdaptiveEPS/strategy_base-eps_0.5-iter_eps_0.5-num_steps_1-attach_rate_0.5/'

    # pc_dir = r'/Data4T/Outputs/centerpoint-adv/GSDA/strategy_base_low_freq_2000-eps_0.5-eps_ratio_0.5-num_steps_1-attach_rate_0.1/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/GSDA/strategy_base_low_freq_2000-eps_0.5-eps_ratio_0.5-num_steps_1-attach_rate_0.1/'
    #
    pc_dir = r'/Data4T/Outputs/centerpoint-adv/GSDA/strategy_base_low_freq-eps_0.5-eps_GFT_0.5-max_points_6000/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize/GSDA/strategy_base_low_freq-eps_0.5-eps_GFT_0.5-max_points_6000/'

    # -----------------------------------
    pc_dir = r'/Data4T/Outputs/centerpoint-adv-pillar/FGSM_CoorAdjust/Epsilon_0.2/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/v1.0-mini/ADV_reVoxelize-pillar/FGSM/Epsilon_0.2/'

    pc_dir = r'/Data4T/Outputs/centerpoint-adv-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10/'

    pc_dir = r'/Data4T/Outputs/centerpoint-adv-pillar/AdaptiveEPS/strategy_PGD-filterOnce-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_0.3/'

    pc_dir = r'/Data4T/Outputs/centerpoint-adv/AdaptiveEPS/strategy_PGD-filterOnce-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_0.2/'

    # # ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pkl_path = os.path.join(work_dir, 'prediction.pkl')
    json_path = os.path.join(work_dir, 'infos_train_10sweeps_withvelo_filter_True.json')

    # a860c02c06e54d9dbe83ce7b694f6c17-origin.ply
    # samples/CAM_FRONT/n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489304412404.jpg
    # samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489304446846.pcd.bin

    # 7ff999a1242e4573a81dd3a6f0c158a2

    with open(json_path) as f:
        json_data = json.load(f)

    with open(pkl_path, 'rb+') as f:
        print(f)
        data_dicts = pickle.load(f)

    cnt = 0
    vis = o3d.visualization.Visualizer()
    for key, data in data_dicts.items():
        if '7ff999a1242e4573a81dd3a6f0c158a2' not in key:
            continue

        print("========= processing {}".format(key))
        # ply_file = os.path.join(pc_dir, key + '-origin.ply')
        # ply_file = os.path.join(pc_dir, key + '-adv.ply')
        # # ply_file = os.path.join(pc_dir, key + '-mean_origin.ply')
        # ply_file = os.path.join(pc_dir, key + '-mean_adv.ply')
        #
        # ply = o3d.io.read_triangle_mesh(ply_file)
        # points = np.array(ply.vertices)

        points_path = os.path.join(pc_dir, key + '.bin')
        # points_path = os.path.join(pc_dir, key + '-innocent.bin')
        # points_path = os.path.join(pc_dir, key + '-innocent_gradient.bin')
        # points_path = os.path.join(r'/home/jqwu/Datasets/nuScenes/v1.0-mini/', 'samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489304446846.pcd.bin')

        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)
        print(points.shape)
        visualize_points_colorGradient(vis, data, pc_dir, key)  # 根据梯度上颜色
        # visualize_points_customized_color(vis, data, pc_dir, key)
        visualize_points_colorModification(vis, data, pc_dir, key, is_show_adv=True)  # 根据扰动位置上颜色
        # visualize_points_colorModification(vis, data, pc_dir, key, is_show_adv=False)  # 根据扰动位置上颜色
        # visualize_points_colorHeight(vis, data, pc_dir, key) # 根据高度上颜色

        # visualize_points_without_color(points, data)  # 普通可视化
        visualize_points_without_color_vis(points, vis, data)  # 普通可视化
        visualize_points_without_color_without_detection(points, vis, pc_dir.split('/')[-2])
