# from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import os.path
import pickle
import json
import pathlib
import platform

plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
from tools.analysis.visual_utily import *
from configs.adv.voxelization_setup import *
import os
import cv2


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


def generate_images(dir_save_images):
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

        points_list = split_10_samples(points)

        for pc_idx in range(len(points_list)):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points_list[pc_idx])[:, :3])

            visual = [pcd]
            num_dets = detections['scores'].shape[0]
            visual += plot_boxes(detections, args.thresh, rgb_colors)
            # o3d.visualization.draw_geometries(visual)

            for i in range(len(visual)):
                vis.add_geometry(visual[i])

            # 调整视角
            view_control = vis.get_view_control()
            view_control.rotate(30.0, 20.0)
            view_control.translate(-1.0, 0.5, 0.0)
            view_control.set_zoom(0.4)

            render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
            render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
            render_option.point_size = 2.0  # 设置渲染点的大小
            # json_path = r'./angle_json/angle_xie.json'
            # if os.path.exists(json_path):
            #     render_option.load_from_json(json_path)
            #     print("已加载视角")
            # vis.update_renderer()

            # vis.run()
            # vis.destroy_window()
            # vis.capture_screen_image('pc_test_vis.png')

            # 更新场景和渲染器
            # vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # # 捕获当前帧
            # image = vis.capture_screen_float_buffer(do_render=True)
            # image = np.asarray(image)
            #
            # # 将浮点数缓冲区转换为图像，并保存为文件
            # image = Image.fromarray((image * 255).astype(np.uint8))
            # image.save("frame_{:02d}.png".format(i))

            dir_save = os.path.join(dir_save_images, key)
            os.makedirs(dir_save, exist_ok=True)
            path_save = os.path.join(dir_save, "{}_{:02d}.png".format(key, pc_idx))
            vis.capture_screen_image(path_save)
            # o3d.visualization.draw_geometries()
            vis.clear_geometries()
    vis.destroy_window()


# 图片转视频
def image_to_video_fun(dir_images, path_video, fps):
    file_list = os.listdir(dir_images)
    # 此处必须sort，因为读取的文件顺序按照字符串ASSIC码排序而非数字。

    file_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    # fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWriter = cv2.VideoWriter(path_video, fourcc, fps, (1920, 1080))
    for file_name in file_list:
        img = cv2.imread(os.path.join(dir_images, file_name), 1)
        videoWriter.write(img)
    videoWriter.release()


def generate_videos(dir_save_images, dir_save_videos):
    os.makedirs(dir_save_videos, exist_ok=True)
    # 要被合成的多张图片所在文件夹
    # 路径分隔符最好使用“/”,而不是“\”,“\”本身有转义的意思；或者“\\”也可以。
    # 因为是文件夹，所以最后还要有一个“/”
    dirs_images = os.listdir(dir_save_images)
    for floder in dirs_images:
        token = floder
        path_output = os.path.join(dir_save_videos, token + '.mp4')
        image_to_video_fun(os.path.join(dir_save_images, floder), path_output, 2)


if __name__ == '__main__':
    dir_save_images = r'./save/images-innocent/'
    dir_save_videos = r'./save/videos-innocent/'
    generate_images(dir_save_images)
    generate_videos(dir_save_images, dir_save_videos)
