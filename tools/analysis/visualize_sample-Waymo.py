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

    pc_dir = r'/Data4T-1/Outputs/Waymo/centerpoint-adv/FGSM_CoorAdjust/Epsilon_0.2/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/FGSM/Epsilon_0.2/'

    pc_dir = r'/Data4T-1/Outputs/Waymo/centerpoint-adv-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/'
    #
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/Eval/innocent/waymo_centerpoint_pp_two_pfn_stride1_3x/'
    #
    pc_dir = r'/Data4T-1/Outputs/Waymo/centerpoint-adv/FGSM_CoorAdjust/Epsilon_0.2/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/FGSM/Epsilon_0.2/'

    # pc_dir = r'/Data4T-1/Outputs/Waymo/centerpoint-adv/AdaptiveEPS/strategy_PGD-filterOnce-PCSel_323-fixedEPS_0.3-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_0.3/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/AdaptiveEPS/strategy_PGD-filterOnce-PCSel_323-fixedEPS_0.3-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_0.3/'

    ## innocent
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/Eval/innocent/waymo_centerpoint_voxelnet_two_sweeps_3x_with_velo/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV-pillar/Eval/innocent/waymo_centerpoint_pp_two_pfn_stride1_3x/'

    pc_dir = r'/Data4T-1/Outputs/Waymo/centerpoint-adv-pillar/FGSM_CoorAdjust/Epsilon_0.2/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV-pillar/FGSM/Epsilon_0.2/'
    #
    # pc_dir = r'/Data4T-1/Outputs/Waymo/centerpoint-adv-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/'

    # # ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # pkl_path = os.path.join(work_dir, 'prediction-6091.pkl')
    pkl_path = os.path.join(work_dir, 'prediction-323.pkl')
    print('pkl_path: {}'.format(pkl_path))

    if 'Waymo/centerpoint-adv' in pc_dir:
        if 'Waymo/centerpoint-adv-pillar' in pc_dir:
            point_dim = 5
        else:
            point_dim = 6


    # seq_0_frame_0.pkl.bin
    # seq_29_frame_163.pkl
    # seq_175_frame_182.pkl

    with open(pkl_path, 'rb+') as f:
        print(f)
        data_dicts = pickle.load(f)

    cnt = 0
    vis = o3d.visualization.Visualizer()

    # # def visualize_light(data):
    # key = 'seq_0_frame_0.pkl'
    # # points_path = os.path.join(pc_dir, key + '.bin')
    # points_path = os.path.join(pc_dir, key + '-innocent.bin')
    # points = np.fromfile(points_path, dtype=np.float32).reshape(-1, point_dim)
    # print(points.shape)
    # data = data_dicts[key]
    # # visualize_points_colorGradient(vis, data, pc_dir, key)  # 根据梯度上颜色
    # # visualize_points_customized_color(vis, data, pc_dir, key)
    # # visualize_points_colorModification(vis, data, pc_dir, key, is_show_adv=True)  # 根据扰动位置上颜色
    # # visualize_points_colorModification(vis, data, pc_dir, key, is_show_adv=False)  # 根据扰动位置上颜色
    # # visualize_points_colorHeight(vis, data, pc_dir, key) # 根据高度上颜色
    #
    # visualize_points_without_color(points, data)  # 普通可视化
    # # visualize_points_without_color_vis(points, vis, data)  # 普通可视化
    # # visualize_points_without_color_without_detection(points, vis, pc_dir.split('/')[-2])

    # for key, data in data_dicts.items():
    for key in sorted(data_dicts):
        data = data_dicts[key]
        # if 'seq_0_frame_0.pkl' not in key:
        #     continue

        print("========= processing {}".format(key))
        # ply_file = os.path.join(pc_dir, key + '-origin.ply')
        # ply_file = os.path.join(pc_dir, key + '-adv.ply')
        # # ply_file = os.path.join(pc_dir, key + '-mean_origin.ply')
        # ply_file = os.path.join(pc_dir, key + '-mean_adv.ply')
        #
        # ply = o3d.io.read_triangle_mesh(ply_file)
        # points = np.array(ply.vertices)

        points_path = os.path.join(pc_dir, key + '.bin')

        if 'innocent' in work_dir:
            if 'PGD' in pc_dir or 'MI' in pc_dir or '_iter' in pc_dir:
                points_path = os.path.join(pc_dir, key + '-innocent_ori.bin')
            else:
                points_path = os.path.join(pc_dir, key + '-innocent.bin')

        # points_path = os.path.join(pc_dir, key + '-innocent_gradient.bin')
        # points_path = os.path.join(r'/home/jqwu/Datasets/nuScenes/v1.0-mini/', 'samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489304446846.pcd.bin')

        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, point_dim)
        print(points.shape)
        # visualize_points_colorGradient(vis, data, pc_dir, key)  # 根据梯度上颜色
        # visualize_points_customized_color(vis, data, pc_dir, key)
        # visualize_points_colorModification(vis, data, pc_dir, key, is_show_adv=True)  # 根据扰动位置上颜色
        # visualize_points_colorModification(vis, data, pc_dir, key, is_show_adv=False)  # 根据扰动位置上颜色
        # visualize_points_colorHeight(vis, data, pc_dir, key) # 根据高度上颜色

        # visualize_points_without_color(points, data)  # 普通可视化
        visualize_points_without_color_vis(points, vis, data)  # 普通可视化
        # visualize_points_without_color_without_detection(points, vis, pc_dir.split('/')[-2])
