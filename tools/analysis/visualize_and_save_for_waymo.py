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

target_key = None


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


def generate_images_with_pred_3dbox(dir_save_images, pc_dir, pkl_path, thresh=0.3):
    dir_save_images = os.path.join(dir_save_images, 'with_pred_3dbox')
    os.makedirs(dir_save_images, exist_ok=True)

    rgb_colors = ncolors(6)

    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if 'Waymo/centerpoint-adv' in pc_dir:
        if 'Waymo/centerpoint-adv-pillar' in pc_dir:
            point_dim = 5
        else:
            point_dim = 6

    # seq_0_frame_0.pkl.bin
    # seq_29_frame_163.pkl
    # seq_175_frame_182.pkl

    data_dicts = read_data_dict(pkl_path)

    cnt = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    for key, data in data_dicts.items():
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
        visual += plot_boxes(detections, thresh, rgb_colors)
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


def read_data_dict(pkl_path):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb+') as f:
            print(f)
            data_dicts = pickle.load(f)
    else:
        pkl_path_6091 = pkl_path.replace('323', '6091')
        with open(pkl_path_6091, 'rb+') as f:
            print(f)
            data_dicts_6091 = pickle.load(f)

        collected_token_path = r'/home/jqwu/Codes/CenterPoint-adv/tools/waymo/collected_token-323-04280144.pkl'
        with open(collected_token_path, 'rb+') as f:
            print(f)
            collected_token = set(pickle.load(f))

        data_dicts = {}
        for key, data in data_dicts_6091.items():
            if key in collected_token:
                data_dicts[key] = data

    return data_dicts


def generate_images_without_3dbox(dir_save_images, pc_dir, pkl_path, thresh=0.3):
    dir_save_images = os.path.join(dir_save_images, 'without_3dbox')
    os.makedirs(dir_save_images, exist_ok=True)

    rgb_colors = ncolors(6)

    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if 'Waymo/centerpoint-adv' in pc_dir:
        if 'Waymo/centerpoint-adv-pillar' in pc_dir:
            point_dim = 5
        else:
            point_dim = 6

    # seq_0_frame_0.pkl.bin
    # seq_29_frame_163.pkl
    # seq_175_frame_182.pkl

    data_dicts = read_data_dict(pkl_path)

    cnt = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    for key, data in data_dicts.items():
        if target_key != None and target_key not in key:
            continue
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

        # detections = {
        #     'boxes': data['box3d_lidar'].numpy(),
        #     'scores': data['scores'].numpy(),
        #     'classes': data['label_preds'].numpy()
        # }

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        visual = [pcd]
        # num_dets = detections['scores'].shape[0]
        # visual += plot_boxes(detections, thresh, rgb_colors)
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

        vis.run()

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


def generate_images_with_modification_positions(dir_save_images, pc_dir, pkl_path, thresh=0.3, is_heatmap=False):
    if is_heatmap:
        dir_save_images = os.path.join(dir_save_images, 'modification_heatmap')
    else:
        dir_save_images = os.path.join(dir_save_images, 'modification_positions')
    os.makedirs(dir_save_images, exist_ok=True)

    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    if 'Waymo/centerpoint-adv' in pc_dir:
        if 'Waymo/centerpoint-adv-pillar' in pc_dir:
            point_dim = 5
        else:
            point_dim = 6

    data_dicts = read_data_dict(pkl_path)

    cnt = 0
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    for key, data in data_dicts.items():
        if target_key != None and target_key not in key:
            continue
        print("========= processing {}th/{} point cloud: {}".format(cnt, len(data_dicts), key))
        cnt += 1

        points_adv_path = os.path.join(pc_dir, key + '.bin')
        ## innocent
        if 'PGD' in pc_dir or 'MI' in pc_dir or '_iter' in pc_dir:
            points_innocent_path = os.path.join(pc_dir, key + '-innocent_ori.bin')
        else:
            points_innocent_path = os.path.join(pc_dir, key + '-innocent.bin')

        points_innocent = np.fromfile(points_innocent_path, dtype=np.float32).reshape(-1, point_dim)
        points_adv = np.fromfile(points_adv_path, dtype=np.float32).reshape(-1, point_dim)
        points = points_innocent
        points_diff = points_adv[:, :3] - points_innocent[:, :3]
        print(points.shape)

        if is_heatmap:
            colors = color_generation(np.abs(points_diff[:, :3]).sum(-1))
        else:
            cnt_modify = 0
            colors = np.ones_like(points[:, :3]) * 0.5
            for i in range(points_diff.shape[0]):
                if np.abs(points_diff[i, :]).sum() > 0:
                    colors[i, :] = [1, 1, 0]
                    cnt_modify += 1
            print('MR: {:02.1f}'.format(cnt_modify / points_diff.shape[0] * 100))

        # detections = {
        #     'boxes': data['box3d_lidar'].numpy(),
        #     'scores': data['scores'].numpy(),
        #     'classes': data['label_preds'].numpy()
        # }

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        visual = [pcd]
        # num_dets = detections['scores'].shape[0]
        # visual += plot_boxes(detections, thresh, rgb_colors)
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

        vis.run()

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


def generate_images_with_gt_3dbox(dir_save_images, pc_dir, pkl_path, thresh=0.3):
    dir_save_images = os.path.join(dir_save_images, 'with_gt_3dbox')
    os.makedirs(dir_save_images, exist_ok=True)

    rgb_colors = ncolors(6)
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    gt_pkl_path = os.path.join('/Data4T-1/Data/WaymoOpenDataset/', 'infos_val_01sweeps_filter_zero_gt.pkl')

    if 'Waymo/centerpoint-adv' in pc_dir:
        if 'Waymo/centerpoint-adv-pillar' in pc_dir:
            point_dim = 5
        else:
            point_dim = 6

    # seq_0_frame_0.pkl.bin
    # seq_29_frame_163.pkl
    # seq_175_frame_182.pkl

    data_dicts = read_data_dict(pkl_path)

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
        print("========= processing {}th/{} point cloud: {}".format(cnt, len(data_dicts), key))
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
        visual += plot_boxes(detections, thresh, rgb_colors)
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


def visualize_points_saved_waymo(dir_save_images, pc_dir, pkl_path, thresh=0.3):
    generate_images_without_3dbox(dir_save_images, pc_dir, pkl_path, thresh)
    # generate_images_with_gt_3dbox(dir_save_images, pc_dir, pkl_path, thresh)
    # generate_images_with_pred_3dbox(dir_save_images, pc_dir, pkl_path, thresh)

    # generate_images_with_modification_positions(dir_save_images, pc_dir, pkl_path, thresh, is_heatmap=False)
    generate_images_with_modification_positions(dir_save_images, pc_dir, pkl_path, thresh, is_heatmap=True)

if __name__ == '__main__':
    pc_dir = r'/Data4T-1/Outputs/Waymo/centerpoint-adv-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/'
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset-0506/ADV-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/'
    #
    ###====================================== waymo innocent =============================================
    work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset-0506/ADV-pillar/Eval/innocent/waymo_centerpoint_pp_two_pfn_stride1_3x/'
    #

    ####====================================== waymo voxel =============================================

    ## innocent
    innocent_voxel = dict(
        dir_save_images=r'./save/images-waymo-voxel/innocent',
        pc_dir=r'/Data4T-1/Outputs/Waymo/centerpoint-adv/AdaptiveEPS/strategy_PGD-filterOnce-fixedEPS_0.3-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_0.2',
        work_dir=r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/Eval/innocent/waymo_centerpoint_voxelnet_two_sweeps_3x_with_velo/',
    )

    ## FGSM
    FGSM_voxel = dict(
        dir_save_images=r'./save/images-waymo-voxel/FGSM',
        pc_dir=r'/Data4T-1/Outputs/Waymo/centerpoint-adv/FGSM_CoorAdjust/Epsilon_0.2/',
        work_dir=r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/FGSM/Epsilon_0.2/',
    )
    ## PGD
    PGD_voxel = dict(
        dir_save_images=r'./save/images-waymo-voxel/PGD',
        pc_dir=r'/Data4T-1/Outputs/Waymo/centerpoint-adv/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/',
        work_dir=r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart',
    )

    ## IOU
    IOU_voxel = dict(
        dir_save_images=r'./save/images-waymo-voxel/IOU',
        pc_dir=r'/Data4T-1/Outputs/Waymo/centerpoint-adv/IOU_iter/eps_0.2-eps_iter_0.2-num_steps_1-Lambda_0.1-iou_0.1-score_0.1',
        work_dir=r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/IOU_iter/eps_0.2-eps_iter_0.2-num_steps_1-Lambda_0.1-iou_0.1-score_0.1',
    )

    ## AdaptiveEPS
    AdaptiveEPS_voxel = dict(
        dir_save_images=r'./save/images-waymo-voxel/AdaptiveEPS',
        pc_dir=r'/Data4T-1/Outputs/Waymo/centerpoint-adv/AdaptiveEPS/strategy_PGD-filterOnce-fixedEPS_0.3-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_0.2',
        work_dir=r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV/AdaptiveEPS/strategy_PGD-filterOnce-fixedEPS_0.3-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_0.2',
    )

    ####====================================== waymo pillar =============================================

    ## PGD
    PGD_pillar = dict(
        dir_save_images=r'./save/images-waymo-pillar/PGD',
        pc_dir=r'/Data4T-1/Outputs/Waymo/centerpoint-adv-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart/',
        work_dir=r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV-pillar/PGD_CoorAdjust/eps_0.2-eps_iter_0.03-num_steps_10-randStart',
    )

    ## IOU
    IOU_pillar = dict(
        dir_save_images=r'./save/images-waymo-pillar/IOU',
        pc_dir=r'/Data4T-1/Outputs/Waymo/centerpoint-adv-pillar/IOU_iter/eps_0.2-eps_iter_0.2-num_steps_1-Lambda_0.1-iou_0.1-score_0.1',
        work_dir=r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV-pillar/IOU_iter/eps_0.2-eps_iter_0.2-num_steps_1-Lambda_0.1-iou_0.1-score_0.1',
    )

    ## AdaptiveEPS
    AdaptiveEPS_pillar = dict(
        dir_save_images=r'./save/images-waymo-pillar/AdaptiveEPS',
        pc_dir=r'/Data4T-1/Outputs/Waymo/centerpoint-adv-pillar/AdaptiveEPS/strategy_PGD-filterOnce-fixedEPS_0.1-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_0.2',
        work_dir=r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/WaymoDataset/ADV-pillar/AdaptiveEPS/strategy_PGD-filterOnce-fixedEPS_0.1-eps_0.5-eps_ratio_0.5-num_steps_10-attach_rate_0.2',
    )

    # seq_25_frame_160.pkl
    target_key = 'seq_25_frame_160.pkl'

    adv_list = [PGD_voxel, IOU_voxel]
    adv_list = [PGD_pillar, IOU_pillar, AdaptiveEPS_pillar]
    adv_list = [AdaptiveEPS_voxel, AdaptiveEPS_pillar]
    adv_list = [AdaptiveEPS_pillar]

    adv_list = [AdaptiveEPS_voxel]
    # adv_list = [PGD_voxel]
    # adv_list = [innocent_voxel]

    adv_list = [AdaptiveEPS_voxel]
    # adv_list = [PGD_voxel]
    # adv_list = [innocent_voxel]

    adv_list = [AdaptiveEPS_pillar]
    # adv_list = [PGD_pillar]
    # adv_list = [innocent_voxel]

    for adv_alg in adv_list:
        dir_save_images = adv_alg['dir_save_images']
        pc_dir = adv_alg['pc_dir']
        work_dir = adv_alg['work_dir']
        pkl_path = os.path.join(work_dir, 'prediction-323.pkl')
        visualize_points_saved_waymo(dir_save_images, pc_dir, pkl_path)

        # seq_37_frame_164.pkl.png
