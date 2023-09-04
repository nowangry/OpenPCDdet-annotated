import open3d as o3d
import numpy as np
import os
from tools.analysis.histogram import *
from tools.analysis.visual_utily import *


def compute_curvature(pcd, knn):
    """
    计算点云中每个点的曲率
    :param pcd: Open3D点云对象
    :param knn: 曲率估计所需的最近邻点数量
    :return curvature: 点云中每个点的曲率
    """
    # 使用Open3D库的KDTree实现最近邻搜索
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    curvature = []
    for i in range(pcd.points.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)
        if k < knn:
            curvature.append(0)
        else:
            cov_matrix = np.cov(pcd.points[idx].T)
            eig_values, _, _ = np.linalg.svd(cov_matrix)
            curvature.append(eig_values[2] / (eig_values[0] + eig_values[1] + eig_values[2]))
    return np.array(curvature)

def compute_curvature_attention(pcd, knn, alpha):
    """
    计算点云中每个点的注意力系数
    :param pcd: Open3D点云对象
    :param knn: 曲率估计所需的最近邻点数量
    :param alpha: 注意力系数的超参数
    :return curvature_attention: 点云中每个点的注意力系数
    """
    # 计算点云中每个点的曲率
    curvature = compute_curvature(pcd, knn)
    # 计算点云中每个点的注意力系数
    curvature_attention = np.exp(-alpha * curvature)
    return curvature_attention

def main():
    import open3d as o3d
    import numpy as np

    # 加载点云
    pcd = o3d.io.read_point_cloud("point_cloud.pcd")

    # 计算曲率
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    curvature = np.asarray(pcd.compute_point_cloud_normals()).T[2]

    # 根据曲率生成点云注意力权重
    alpha = 1 - (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))

    # 可视化
    pcd.colors = o3d.utility.Vector3dVector(np.tile(alpha, (3, 1)).T)
    o3d.visualization.draw_geometries([pcd])

def main1():
    import open3d as o3d
    import pyntcloud
    pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'
    pc_path = os.path.join(pc_dir, 'a860c02c06e54d9dbe83ce7b694f6c17.bin')
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)

    # 加载点云数据
    # cloud = pyntcloud.PyntCloud.from_file(pc_path, sep=' ', header=None, names=['x', 'y', 'z'])
    cloud = pyntcloud.PyntCloud(points=points[:, :3])

    # 计算曲率
    curvatures = cloud.compute_curvature()

    # 将曲率数据保存到点云颜色中
    colors = [[curvatures.loc[i, "k1"], curvatures.loc[i, "k2"], 0] for i in range(len(curvatures))]
    cloud.points.colors = colors

    # 将点云数据转换为Open3D格式并可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud.xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main1()
    pc_dir = '/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'
    work_dir = '../../work_dirs/ADV_reVoxelize/test_reVoxelize-backup/innocent/'
    cnt = 0
    save_dir = r'../save/attentions/images_normal_attention'
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
            # density_attention = compute_density_attention(pcd=pcd, radius=0.15, knn=3, alpha=1)
            curvature_attention = compute_curvature_attention(pcd, 10, 1)
            curvature = curvature_attention

            # 打印结果
            print(curvature)

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
