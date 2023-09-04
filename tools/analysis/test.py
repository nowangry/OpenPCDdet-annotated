import open3d as o3d
import numpy as np
import os

print("->正在加载点云... ")
pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'
input_path = os.path.join(pc_dir, 'a860c02c06e54d9dbe83ce7b694f6c17.bin')
# input_path = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/FGSM-no_sign/FGSM-eps_0.2/0a3abc33048d46f9bd78151d1df4b004.bin'
points = np.fromfile(input_path, dtype=np.float32).reshape(-1, 5)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points[:, :3])
# pcd = o3d.io.read_point_cloud("")
print(pcd)


def test_DBSCAN():
    print("->正在DBSCAN聚类...")
    eps = 0.5  # 同一聚类中最大点间距
    min_points = 50  # 有效聚类的最小点数
    # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    # labels = np.array(pcd.cluster_dbscan(eps, min_points, print_progress=True))
    labels = np.random.randint(0, 100, (points.shape[0]))
    max_label = labels.max()  # 获取聚类标签的最大值 [-1,0,1,2,...,max_label]，label = -1 为噪声，因此总聚类个数为 max_label + 1
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # labels = -1 的簇为噪声，以黑色显示
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def segment_plane():
    # test_data_dir = '/home/pi/PycharmProjects/learn/Open3D/examples/test_data'
    # point_cloud_file_name = 'fragment.pcd'
    # point_cloud_file_path = os.path.join(test_data_dir, point_cloud_file_name)
    point_cloud_file_path = input_path
    # 读取点云
    # pcd = o3d.io.read_point_cloud(point_cloud_file_path)
    # 平面分割
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.30,
                                             ransac_n=5,
                                             num_iterations=1000)
    # 模型参数
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    # 平面内的点
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # 平面外的点
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    outlier_cloud.paint_uniform_color([0, 1.0, 0])
    # 可视化
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def generate_random_numbers1():
    import numpy as np

    # 生成一个长度为n的不重复随机数数组
    def generate_random_numbers(n):
        arr = np.random.permutation(n)
        random_numbers = np.random.choice(arr, size=n, replace=False)
        return random_numbers

    # 生成10个不重复随机数
    random_numbers = generate_random_numbers(10)
    print(random_numbers)

    # 生成不重复的随机数
    def generate_random_numbers(n, size):
        return np.random.choice(n, size=size, replace=False)

    # 生成10个不重复的随机数，范围为0到99
    random_numbers = generate_random_numbers(100, 10)
    print(random_numbers)


def check_point_in_box(pts, box):
    """
	pts[x,y,z]
	box[c_x,c_y,c_z,dx,dy,dz,heading]
    """
    shift_x = np.abs(pts[0] - box[0])
    shift_y = np.abs(pts[1] - box[1])
    shift_z = np.abs(pts[2] - box[2])
    cos_a = np.cos(box[6])
    sin_a = np.sin(box[6])
    dx, dy, dz = box[3], box[4], box[5]
    local_x = shift_x * cos_a + shift_y * sin_a
    # local_y = shift_y * cos_a - shift_x * sin_a
    local_y = shift_y * cos_a - shift_x * sin_a
    if (np.abs(shift_z) > dz / 2.0 or np.abs(local_x) > dx / 2.0 or np.abs(local_y) > dy / 2.0):
        return False
    return True


import pickle
import json
import pathlib
import platform

plt = platform.system()
if plt != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

from tools.analysis.visual_utily import *
from configs.adv.voxelization_setup import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--path', help='path to visualization file', type=str, default=None)
    parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.3)
    args = parser.parse_args()

    rgb_colors = ncolors(10)

    pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'
    work_dir = '../../work_dirs/ADV_reVoxelize/test_reVoxelize-backup/innocent/'

    # pc_dir = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/FGSM_CoorAdjust/Epsilon_0.1/'
    # work_dir = r'/home/jqwu/Codes/CenterPoint-adv/work_dirs/ADV_reVoxelize/FGSM/Epsilon_0.1/'

    # # ## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
    for key, data in data_dicts.items():
        if 'a860c02c06e54d9dbe83ce7b694f6c17' not in key:
            continue
        print("========= processing {}".format(key))

        points_path = os.path.join(pc_dir, key + '.bin')
        # points_path = os.path.join(pc_dir, key + '-innocent.bin')
        # points_path = os.path.join(pc_dir, key + '-innocent_gradient.bin')
        # points_path = os.path.join(r'/home/jqwu/Datasets/nuScenes/v1.0-mini/', 'samples/LIDAR_TOP/n008-2018-08-28-16-43-51-0400__LIDAR_TOP__1535489304446846.pcd.bin')

        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)
        print(points.shape)

        # detections = {
        #     'boxes': data['box3d_lidar'].numpy(),
        #     'scores': data['scores'].numpy(),
        #     'classes': data['label_preds'].numpy()
        # }
        detections = {
            'boxes': data['box3d_lidar'][0, :].numpy().reshape(1, -1),
            'scores': data['scores'][0].numpy().reshape(1),
            'classes': data['label_preds'][0].numpy().reshape(1)
        }

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        visual = [pcd]
        thresh = 0.3
        visual += plot_boxes(detections, thresh, rgb_colors)
        # o3d.visualization.draw_geometries(visual)
        #
        vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
        render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
        render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
        render_option.point_size = 2.0  # 设置渲染点的大小
        for i in range(len(visual)):
            vis.add_geometry(visual[i])
        vis.run()
        vis.destroy_window()
