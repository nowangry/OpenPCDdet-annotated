import os

import open3d as o3d
import numpy as np
from tools.analysis.histogram import *
from tools.analysis.visual_utily import *


def compute_density(pcd, radius, knn):
    """
    计算点云中每个点的局部密度值
    :param pcd: Open3D点云对象
    :param radius: 球形领域的半径
    :param knn: 球形领域内的最近邻点数量
    :return density: 点云中每个点的局部密度值
    """
    # 使用Open3D库的KDTree实现最近邻搜索
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    density = np.zeros(np.asarray(pcd.points).shape[0])
    for i in range(np.asarray(pcd.points).shape[0]):
        l2_dist = np.linalg.norm(np.asarray(pcd.points)[i, :3])
        radius_norm = radius * l2_dist
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius_norm)
        density[i] = k / (4 / 3 * np.pi * radius ** 3)
        # if k < knn:
        #     density[i] = 0
        # else:
        #     density[i] = k / (4 / 3 * np.pi * radius ** 3)
    return density

def compute_density_attention(pcd, radius, knn, alpha):
    """
    计算点云中每个点的注意力系数
    :param pcd: Open3D点云对象
    :param radius: 球形领域的半径
    :param knn: 球形领域内的最近邻点数量
    :param alpha: 注意力系数的超参数
    :return density_attention: 点云中每个点的注意力系数
    """
    # 计算点云中每个点的局部密度值
    density = compute_density(pcd, radius, knn)

    MyHist(density, bins=1000, title_name='density')
    # 计算点云中每个点的注意力系数
    # density_attention = np.exp(-alpha * density)
    # MyHist(density_attention, bins=1000, title_name='density_attention')
    #
    # density_attention_log = np.log(density * 0.01 + 1)
    # MyHist(density_attention_log, bins=1000, title_name='density_attention - log')
    return density


if __name__ == "__main__":
    pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'
    work_dir = '../../work_dirs/ADV_reVoxelize/test_reVoxelize-backup/innocent/'
    cnt = 0
    save_dir = r'../save/attentions/images_density_attention_norm'
    os.makedirs(save_dir, exist_ok=True)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口

    for root, _, filenames in os.walk(pc_dir):
        for filename in filenames:
            points_path = os.path.join(root, filename)
            print("========= processing {}th point cloud: {}".format(cnt, points_path))
            cnt += 1

            points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)

            # 加载点云数据
            # pcd_data = np.loadtxt('point_cloud.xyz')
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            # 计算点云中每个点的注意力系数
            density_attention = compute_density_attention(pcd=pcd, radius=0.15, knn=3, alpha=1)

            # 打印结果
            print(density_attention)

            curvature = density_attention
            # 根据曲率生成点云注意力权重
            # alpha = 1 - (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))
            alpha = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))

            # 可视化
            colors = color_generation(alpha)
            # colors = np.tile(alpha, (3, 1)).T
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pcd])

            save_path = os.path.join(save_dir, filename.replace('.bin', '.png'))
            save_3d_visualization(vis, pcd, save_path)
    vis.destroy_window()
