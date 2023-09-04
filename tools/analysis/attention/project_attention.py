import open3d as o3d
import numpy as np

# 加载点云
pcd = o3d.io.read_point_cloud("point_cloud.pcd")

# 定义投影面
plane1 = o3d.geometry.Plane([0, 0, 1], [0, 0, 0])
plane2 = o3d.geometry.Plane([0, 1, 0], [0, 0, 0])
planes = [plane1, plane2]

# 投影并计算点云注意力权重
alphas = []
for plane in planes:
    pcd_proj = pcd.project_plane(plane, project_distance=0.01)
    dist = np.sqrt(np.sum(np.asarray(pcd_proj.points) ** 2, axis=1))
    alpha = 1 - (dist - np.min(dist)) / (np.max(dist) - np.min(dist))
    alphas.append(alpha)

# 将两个平面的注意力权重相加
alpha = np.mean(alphas, axis=0)

# 可视化
pcd.colors = o3d.utility.Vector3dVector(np.tile(alpha, (3, 1)).T)
o3d.visualization.draw_geometries([pcd])
