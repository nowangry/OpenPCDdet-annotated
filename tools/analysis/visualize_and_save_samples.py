# from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import os.path
import pickle
import json
import pathlib
import platform

import numpy as np

plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
from tools.analysis.visual_utily import *
# from configs.adv.voxelization_setup import *
import os


def split_10_samples(points):
    time_set = set()
    for i in range(points.shape[0]):
        time = points[i, -1]
        time_set.add(time)

    time_arr = np.array(list(time_set))
    value = np.sort(time_arr)
    time_to_index = {}
    for i in range(value.shape[0]):
        time_to_index[value[i]] = i

    points_list = [[] for i in range(10)]
    for i in range(points.shape[0]):
        time = points[i, -1]
        index = time_to_index[time]
        points_list[index].append(points[i, :])

    return points_list


def visualize_points_segment_plane(dir_save_images):
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

    pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-origin_points/'
    pkl_path = '../../work_dirs/ADV_reVoxelize/test_reVoxelize-backup/innocent/prediction.pkl'
    json_path = '../../work_dirs/ADV_reVoxelize/test_reVoxelize-backup/innocent/infos_train_10sweeps_withvelo_filter_True.json'

    pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'
    work_dir = '../../work_dirs/ADV_reVoxelize/test_reVoxelize-backup/innocent/'
    # #
    # pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2/'
    # pkl_path = '../../work_dirs/ADV_reVoxelize/test_reVoxelize-backup/adv/prediction.pkl'
    # json_path = '../../work_dirs/ADV_reVoxelize/test_reVoxelize-backup/adv/infos_train_10sweeps_withvelo_filter_True.json'

    # pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/Point_Attach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted_not_add/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/ADV_reVoxelize/PointAttach/eps_Nan-iter_eps_0.2-num_steps_1-Lambda_0.0001-attach_rate_0.1-strategy_gradient_sorted_not_add/'

    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/FGSM_CoorAdjust/Epsilon_0.1/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/ADV_reVoxelize/FGSM/Epsilon_0.1/'
    #
    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/PGD_CoorAdjust/eps_0.1-eps_iter_0.02-num_steps_10/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/ADV_reVoxelize/PGD_CoorAdjust/eps_0.1-eps_iter_0.02-num_steps_10/'
    #
    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/IOU_CoorAdjust/eps_0.1-eps_iter_0.1-num_steps_1-Lambda_0.1-iou_0.1-score_0.1/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/ADV_reVoxelize/IOU/eps_0.1-eps_iter_0.1-num_steps_1-Lambda_0.1-iou_0.1-score_0.1/'

    #
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pkl_path = os.path.join(work_dir, 'prediction.pkl')
    json_path = os.path.join(work_dir, 'infos_train_10sweeps_withvelo_filter_True.json')

    # a860c02c06e54d9dbe83ce7b694f6c17-origin.ply
    # samples/CAM_FRONT/n008-2018-08-28-16-43-51-0400__CAM_FRONT__1535489304412404.jpg
    # samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489304446846.pcd.bin

    with open(json_path) as f:
        json_data = json.load(f)

    with open(pkl_path, 'rb+') as f:
        print(f)
        data_dicts = pickle.load(f)

    cnt = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    for key, data in data_dicts.items():
        # if 'a860c02c06e54d9dbe83ce7b694f6c17' not in key:
        #     continue

        print("========= processing {}th point cloud: {}".format(cnt, key))
        cnt += 1
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
        # visualize_points_colored_v1(points, vis, data)
        # visualize_points_without_color(points, vis, data)
        # visualize_points_without_color_save_image(points, vis, data)
        # visualize_points_without_color_without_detection(points, vis, pc_dir.split('/')[-2])

        detections = {
            'boxes': data['box3d_lidar'].numpy(),
            'scores': data['scores'].numpy(),
            'classes': data['label_preds'].numpy()
        }

        # points_list = split_10_samples(points)
        points_list = [points]
        for pc_idx in range(len(points_list)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points_list[pc_idx])[:, :3])

            plane_model, inliers = pcd.segment_plane(distance_threshold=0.30,
                                                     ransac_n=5,
                                                     num_iterations=1000)
            # 模型参数
            [a, b, c, d] = plane_model
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
            # colors = np.zeros_like(np.asarray(pcd.points))
            colors = np.stack(np.array([[0, 1.0, 0]]).repeat(np.asarray(pcd.points).shape[0], axis=0), axis=0)
            for inliers_index in inliers:
                colors[inliers_index, :] = [1.0, 0, 0]
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

            visual = [pcd]
            num_dets = detections['scores'].shape[0]
            # visual += plot_boxes(detections, args.thresh, rgb_colors)
            # o3d.visualization.draw_geometries(visual)

            for i in range(len(visual)):
                vis.add_geometry(visual[i])

            # 调整视角
            '''
            set_lookat设置的是：拖动模型旋转时，围绕哪个点进行旋转。
            set_front设置的是：垂直指向屏幕外的向量，三维空间中有无数向量，垂直指向屏幕外的只有一个。
            set_up设置的是：是设置指向屏幕上方的向量，当设置了垂直指向屏幕外的向量后，模型三维空间中的哪个面和屏幕平行就确定了（垂直屏幕的向量相当于法向量），还剩下一个旋转自由度，设置指向屏幕上方的向量后，模型的显示方式就确定了。
            注：set_front和set_up设置的向量应该要满足一定的约束关系，否则可能得不到想要的效果。
            ————————————————
            版权声明：本文为CSDN博主「不解不惑」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
            原文链接：https://blog.csdn.net/qq_24815615/article/details/121929203
            '''
            view_control = vis.get_view_control()
            # view_control.rotate(30.0, 20.0)
            # view_control.translate(-1.0, 0.5, 0.0)
            view_control.set_front([0.025108506476634779, -0.56206476785211124, 0.8267120173566278])
            view_control.set_lookat([1.5899741522181399, 4.2570619399321821, 1.1817619248494213])
            view_control.set_up([0.014860928327451707, 0.82709121438455047, 0.56187122714829929])
            view_control.set_zoom(0.31999999999999962)

            render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
            render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
            render_option.point_size = 2.0  # 设置渲染点的大小

            # 更新场景和渲染器
            # vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # dir_save = os.path.join(dir_save_images, key)
            # os.makedirs(dir_save, exist_ok=True)
            path_save = os.path.join(dir_save_images, "{}.png".format(key))
            vis.capture_screen_image(path_save)

            # o3d.visualization.draw_geometries(visual)
            # vis.run()
            # input("Press Enter to continue...")
            vis.clear_geometries()
    vis.destroy_window()


def visualize_points_saved_waymo(dir_save_images):
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--path', help='path to visualization file', type=str, default=None)
    parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.3)
    args = parser.parse_args()
    rgb_colors = ncolors(10)

    pc_dir = r'/Data4T-1/Outputs/Waymo/centerpoint-adv-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset-0506/ADV-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/'
    #
    ## innocent
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset-0506/ADV/Eval/innocent/waymo_centerpoint_voxelnet_two_sweeps_3x_with_velo/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset-0506/ADV-pillar/Eval/innocent/waymo_centerpoint_pp_two_pfn_stride1_3x/'
    #
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    pkl_path = os.path.join(work_dir, 'prediction-323.pkl')
    gt_pkl_path = os.path.join('/Data4T-1/Data/WaymoOpenDataset/', 'infos_val_01sweeps_filter_zero_gt.pkl')

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

    with open(gt_pkl_path, 'rb+') as f:
        print(f)
        gt_data_dicts_origin = pickle.load(f)
        gt_data_dicts = {}
        WAYMO_CLASSES = ['UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'TRUCK', 'CYCLIST', 'SIGN']
        for scence in gt_data_dicts_origin:
            if scence['token'] in data_dicts:
                ob = {}
                ob['box3d_lidar'] = torch.from_numpy(scence['gt_boxes'][:, [0, 1, 2, 3, 4, 5, -1]])
                ob['scores'] = torch.from_numpy(np.ones(scence['gt_names'].shape[0], dtype=np.float32))
                box_name = scence['gt_names']
                ob['label_preds'] = torch.from_numpy(
                    np.array([WAYMO_CLASSES.index(name) for i, name in enumerate(box_name)]).reshape(-1))
                gt_data_dicts[scence['token']] = ob
        data_dicts = gt_data_dicts

    cnt = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    for key, data in data_dicts.items():
        # if 'a860c02c06e54d9dbe83ce7b694f6c17' not in key:
        #     continue

        print("========= processing {}th point cloud: {}".format(cnt, key))
        cnt += 1

        points_path = os.path.join(pc_dir, key + '.bin')
        ## innocent
        if 'innocent' in work_dir:
            if 'PGD' in pc_dir or 'MI' in pc_dir or '_iter' in pc_dir:
                points_path = os.path.join(pc_dir, key + '-innocent_ori.bin')
            else:
                points_path = os.path.join(pc_dir, key + '-innocent.bin')

        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, point_dim)
        print(points.shape)

        detections = {
            'boxes': data['box3d_lidar'].numpy(),
            'scores': data['scores'].numpy(),
            'classes': data['label_preds'].numpy()
        }

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        visual = [pcd]
        num_dets = detections['scores'].shape[0]
        visual += plot_boxes(detections, args.thresh, rgb_colors)
        # o3d.visualization.draw_geometries(visual)

        for i in range(len(visual)):
            vis.add_geometry(visual[i])

        # 调整视角
        '''
        set_lookat设置的是：拖动模型旋转时，围绕哪个点进行旋转。
        set_front设置的是：垂直指向屏幕外的向量，三维空间中有无数向量，垂直指向屏幕外的只有一个。
        set_up设置的是：是设置指向屏幕上方的向量，当设置了垂直指向屏幕外的向量后，模型三维空间中的哪个面和屏幕平行就确定了（垂直屏幕的向量相当于法向量），还剩下一个旋转自由度，设置指向屏幕上方的向量后，模型的显示方式就确定了。
        注：set_front和set_up设置的向量应该要满足一定的约束关系，否则可能得不到想要的效果。
        ————————————————
        版权声明：本文为CSDN博主「不解不惑」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
        原文链接：https://blog.csdn.net/qq_24815615/article/details/121929203
        '''
        view_control = vis.get_view_control()
        # view_control.rotate(30.0, 20.0)
        # view_control.translate(-1.0, 0.5, 0.0)
        view_control.set_front([0.025108506476634779, -0.56206476785211124, 0.8267120173566278])
        view_control.set_lookat([1.5899741522181399, 4.2570619399321821, 1.1817619248494213])
        view_control.set_up([0.014860928327451707, 0.82709121438455047, 0.56187122714829929])
        view_control.set_zoom(0.31999999999999962)

        render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
        render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
        render_option.point_size = 2.0  # 设置渲染点的大小

        # 更新场景和渲染器
        # vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # dir_save = os.path.join(dir_save_images, key)
        # os.makedirs(dir_save, exist_ok=True)
        path_save = os.path.join(dir_save_images, "{}.png".format(key))
        vis.capture_screen_image(path_save)

        vis.clear_geometries()
    vis.destroy_window()


if __name__ == '__main__':
    # dir_save_images = r'./save/images-segment_plane/'
    # os.makedirs(dir_save_images, exist_ok=True)
    # visualize_points_segment_plane(dir_save_images)

    dir_save_images = r'./save/images-waymo-pillar-innocent/with_gt_3dbox-test'
    os.makedirs(dir_save_images, exist_ok=True)
    visualize_points_saved_waymo(dir_save_images)
