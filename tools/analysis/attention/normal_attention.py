import open3d as o3d
import numpy as np
import os
from tools.analysis.histogram import *
from tools.analysis.visual_utily import *


def compute_normal(pcd, knn):
    """
    计算点云中每个点的法向量
    :param pcd: Open3D点云对象
    :param knn: 法向量估计所需的最近邻点数量
    :return normals: 点云中每个点的法向量
    """
    # 使用Open3D库的KDTree实现最近邻搜索
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = []
    for i in range(np.asarray(pcd.points).shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)
        if k < knn:
            normals.append([0, 0, 0])
        else:
            cov_matrix = np.cov(np.asarray(pcd.points)[idx].T)
            _, eig_vectors, _ = np.linalg.svd(cov_matrix)
            # normals.append(eig_vectors[:, 2])
            normals.append(eig_vectors[:2])
    return np.array(normals)


def compute_normal_norm(pcd, knn):
    """
    计算点云中每个点的法向量
    :param pcd: Open3D点云对象
    :param knn: 法向量估计所需的最近邻点数量
    :return normals: 点云中每个点的法向量
    """
    # 使用Open3D库的KDTree实现最近邻搜索
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = []
    for i in range(np.asarray(pcd.points).shape[0]):
        # l2_dist = np.linalg.norm(np.asarray(pcd.points)[i, :3])
        # knn_norm = knn * l2_dist
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], knn)

        cov_matrix = np.cov(np.asarray(pcd.points)[idx].T)
        _, eig_vectors, _ = np.linalg.svd(cov_matrix)
        # normals.append(eig_vectors[:, 2])
        normals.append(eig_vectors[:2])
    return np.array(normals)


def compute_normal_attention(pcd, knn, alpha):
    """
    计算点云中每个点的注意力系数
    :param pcd: Open3D点云对象
    :param knn: 法向量估计所需的最近邻点数量
    :param alpha: 注意力系数的超参数
    :return normal_attention: 点云中每个点的注意力系数
    """
    # 计算点云中每个点的法向量
    normals = compute_normal_norm(pcd, knn)
    # 计算点云中每个点的法向量模长
    norms = np.linalg.norm(normals, axis=1)
    MyHist(norms, bins=1000, title_name='norms')
    # # 计算点云中每个点的注意力系数
    # normal_attention = np.exp(-alpha * norms)
    # MyHist(normal_attention, bins=1000, title_name='normal_attention')
    # norms_attention_log = np.log(norms * 0.01 + 0.5)
    # MyHist(norms_attention_log, bins=1000, title_name='norms_attention - log')
    return norms



def main():
    import open3d as o3d

    # 读取点云
    pcd = o3d.io.read_point_cloud("chair.pcd")
    print(pcd)
    # 法线估计
    radius = 0.01  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    o3d.visualization.draw_geometries([pcd], window_name="法线估计",
                                      point_show_normal=True,
                                      width=800,  # 窗口宽度
                                      height=600)  # 窗口高度

if __name__ == "__main__":
    main()

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
            normal_attention = compute_normal_attention(pcd, 10, 1)

            # 打印结果
            print(normal_attention)

            curvature = normal_attention
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
