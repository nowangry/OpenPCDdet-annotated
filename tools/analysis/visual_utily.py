import os.path
import open3d as o3d
import numpy as np
import pathlib
import platform

platform = platform.system()
if platform != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
import colorsys
import random
import torch
# from configs.adv.voxelization_setup import *
import argparse
from tools.analysis.histogram import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--path', help='path to visualization file', type=str, default=None)
    parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.3)
    args = parser.parse_args()


def plt_GFT(gft):
    dim = np.array(range(gft.shape[0]))
    x = gft[:, 0]
    y = gft[:, 1]
    z = gft[:, 2]

    plt.figure()
    plt.subplot(221)
    plt.plot(dim, x, color='r', label='x')  # label每个plot指定一个字符串标签
    plt.plot(dim, y, '-.', color='b', label='y')
    plt.plot(dim, z, '--', color='g', label='z')
    plt.legend(loc='best')

    plt.subplot(222)
    plt.plot(dim, x, color='r', label='x')
    plt.legend(loc='best')

    plt.subplot(223)
    plt.plot(dim, y, '-.', color='b', label='y')
    plt.legend(loc='best')

    plt.subplot(224)
    plt.plot(dim, z, '--', color='g', label='z')
    plt.legend(loc='best')

    plt.show()
    # plt.savefig('.//result//3.7.png')

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        # r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        r, g, b = [x for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


rgb_colors = ncolors(10)


def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point.

    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.

    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim), axis=1
    ).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2 ** ndim, ndim])
    return corners


def center_to_corner_box3d(centers, dims, angles=None, origin=(0.5, 0.5, 0.5), axis=2):
    """convert kitti locations, dimensions and angles to corners

    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def label2color(label):
    colors = [[204 / 255, 0, 0], [52 / 255, 101 / 255, 164 / 255],
              [245 / 255, 121 / 255, 0], [115 / 255, 210 / 255, 22 / 255]]

    return colors[label]


def label2color_v2(label):
    colors = [[204 / 255, 0, 0], [52 / 255, 101 / 255, 164 / 255],
              [245 / 255, 121 / 255, 0], [115 / 255, 210 / 255, 22 / 255], [], [], [], [], [], []]

    return colors[label]


def corners_to_lines(qs, color=[204 / 255, 0, 0]):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    idx = [(1, 0), (5, 4), (2, 3), (6, 7), (1, 2), (5, 6), (0, 3), (4, 7), (1, 5), (0, 4), (2, 6), (3, 7)]
    cl = [color for i in range(12)]

    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(qs),
    #     lines=np.asarray(o3d.utility.Vector2iVector(idx)),
    # )
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(qs)
    line_set.lines = o3d.utility.Vector2iVector(idx)
    line_set.colors = o3d.utility.Vector3dVector(cl)

    return line_set


def plot_boxes(boxes, score_thresh, rgb_colors):
    visuals = []
    num_det = boxes['scores'].shape[0]
    for i in range(num_det):
        score = boxes['scores'][i]
        if score < score_thresh:
            continue

        box = boxes['boxes'][i:i + 1]
        label = boxes['classes'][i]
        corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
        color = rgb_colors[label]
        visuals.append(corners_to_lines(corner, color))
    return visuals


def visualize_points(points):
    vis = o3d.visualization.Visualizer()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    visual = [pcd]
    o3d.visualization.draw_geometries(visual)

    vis.create_window()  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    for i in range(len(visual)):
        vis.add_geometry(visual[i])
    vis.run()
    vis.destroy_window()


def visualize_points_colored(points, color):
    vis = o3d.visualization.Visualizer()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(color[:, :3])
    visual = [pcd]
    o3d.visualization.draw_geometries(visual)

    vis.create_window()  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    for i in range(len(visual)):
        vis.add_geometry(visual[i])
    vis.run()
    vis.destroy_window()


def visualize_points_colorGradient(vis, data, pc_dir, key):
    '''
    可视化梯度位置
    '''
    start = 0
    thresh = 0.3
    ## visulize with color
    attach_rate = 0.10
    points_innocent = np.fromfile(os.path.join(pc_dir, key + '-innocent.bin'), dtype=np.float32).reshape(-1, 5)
    grad_points = np.fromfile(os.path.join(pc_dir, key + '-innocent_gradient.bin'), dtype=np.float32).reshape(-1, 5)
    assert grad_points.shape == points_innocent.shape
    grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
    for i in range(grad_points.shape[0]):
        grad_points_abs_sum[i] = np.abs(grad_points[i, :2]).sum()
        # grad_points_abs_sum[i] = np.abs(grad_points[i, 3]).sum()
        # grad_points_abs_sum[i] = np.abs(points_innocent[i, 4]).sum()
    grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1, descending=True).numpy()  # 梯度排序
    assert np.max(grad_points_order) < points_innocent.shape[0]
    # MyHist(points_innocent[grad_points_order[attach_num * start: attach_num * (start + 1)], 4], title_name='{}/10~{}/10'.format(start, start + 1))

    attach_num = int(points_innocent.shape[0] * attach_rate)
    colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
    colors[grad_points_order[attach_num * start: attach_num * (start + 1)], :] = [1, 1, 0]
    # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
    # visualize_points_colored(points, colors)
    # visualize_points(points)

    detections = {
        'boxes': data['box3d_lidar'].numpy(),
        'scores': data['scores'].numpy(),
        'classes': data['label_preds'].numpy()
    }

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_innocent[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    visual = [pcd]
    # visual += plot_boxes(detections, thresh, rgb_colors)
    # o3d.visualization.draw_geometries(visual)

    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    for i in range(len(visual)):
        vis.add_geometry(visual[i])
    vis.run()
    vis.destroy_window()


def color_generation(scaler):
    '''
    青色 --> 蓝色 --> 紫色
    '''
    colors = np.zeros([scaler.shape[0], 3])
    gscaler_max = np.max(scaler[:])
    gscaler_min = np.min(scaler[:])
    delta_c = abs(gscaler_max - gscaler_min) / (255 * 2)
    for j in range(scaler.shape[0]):
        color_n = (scaler[j] - gscaler_min) / delta_c
        if color_n <= 255:
            colors[j, :] = [0, 1 - color_n / 255, 1]
        else:
            colors[j, :] = [(color_n - 255) / 255, 0, 1]
    return colors

def color_generation_black2white(scaler):
    '''
    青色 --> 蓝色 --> 紫色
    '''
    colors = np.zeros([scaler.shape[0], 3])
    gscaler_max = np.max(scaler[:])
    gscaler_min = np.min(scaler[:])
    delta_c = abs(gscaler_max - gscaler_min) / (255 * 1)
    for j in range(scaler.shape[0]):
        color_n = (scaler[j] - gscaler_min) / delta_c
        colors[j, :] = [color_n, color_n, color_n]
    return colors

def visualize_points_customized_color(vis, data, pc_dir, key):
    '''
    可视化梯度位置
    '''
    thresh = 0.3
    ## visulize with color
    attach_rate = 0.10
    points_innocent = np.fromfile(os.path.join(pc_dir, key + '-innocent.bin'), dtype=np.float32).reshape(-1, 5)
    grad_points = np.fromfile(os.path.join(pc_dir, key + '-innocent_gradient.bin'), dtype=np.float32).reshape(-1, 5)
    assert grad_points.shape == points_innocent.shape
    grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
    for i in range(grad_points.shape[0]):
        grad_points_abs_sum[i] = np.abs(grad_points[i, :2]).sum()
        grad_points_abs_sum[i] = np.linalg.norm(grad_points[i, :2], ord=2).sum()
        # grad_points_abs_sum[i] = np.abs(grad_points[i, :3]).sum()
        # grad_points_abs_sum[i] = np.abs(grad_points[i, :4]).sum()
    grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1, descending=True).numpy()  # 梯度排序
    assert np.max(grad_points_order) < points_innocent.shape[0]

    # color generation
    print(grad_points_abs_sum.mean(), grad_points_abs_sum.std(), np.median(grad_points_abs_sum))
    MyHist(grad_points_abs_sum, bins=1000, title_name='L1 grad')
    MyHist(grad_points_abs_sum[grad_points_order[:int(points_innocent.shape[0] * 0.1)]], bins=1000,
           title_name='10% L1 grad')
    MyHist(grad_points_abs_sum[grad_points_order[:int(points_innocent.shape[0] * 0.04)]], bins=1000,
           title_name='4% L1 grad')
    MyHist(grad_points_abs_sum[grad_points_order[:int(points_innocent.shape[0] * 0.02)]], bins=1000,
           title_name='2% L1 grad')
    MyHist(grad_points_abs_sum[grad_points_order[:int(points_innocent.shape[0] * 0.01)]], bins=1000,
           title_name='1% L1 grad')
    MyHist(grad_points_abs_sum[grad_points_order[:int(points_innocent.shape[0] * 0.001)]], bins=1000,
           title_name='0.1% L1 grad')
    MyHist(min_max_normalize(grad_points_abs_sum), bins=1000, title_name='min_max_normalize')
    MyHist(l2_normalize(grad_points_abs_sum), bins=1000, title_name='l2_normalize')
    MyHist(z_score_normalize(grad_points_abs_sum), bins=1000, title_name='z_score_normalize')
    colors = color_generation(z_score_normalize(grad_points_abs_sum))

    detections = {
        'boxes': data['box3d_lidar'].numpy(),
        'scores': data['scores'].numpy(),
        'classes': data['label_preds'].numpy()
    }

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_innocent[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    visual = [pcd]
    # visual += plot_boxes(detections, thresh, rgb_colors)
    # o3d.visualization.draw_geometries(visual)

    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    for i in range(len(visual)):
        vis.add_geometry(visual[i])
    vis.run()
    vis.destroy_window()


def visualize_points_colorModification(vis, data, pc_dir, key, is_show_adv=True):
    '''
    可视化修改位置
    '''
    ## visulize with color
    thresh = 0.1
    points_path_adv = os.path.join(pc_dir, key + '.bin')
    points_adv = np.fromfile(points_path_adv, dtype=np.float32).reshape(-1, 5)
    points_innocent_path = os.path.join(pc_dir, key + '-innocent.bin')
    points_innocent = np.fromfile(points_innocent_path, dtype=np.float32).reshape(-1, 5)

    points_diff = points_adv - points_innocent
    attach_num = int(np.linalg.norm(np.abs(points_diff[:, :3]).sum(-1), ord=0))
    print('attach_num = {}'.format(attach_num))
    colors = color_generation(np.abs(points_diff[:, :3]).sum(-1))

    detections = {
        'boxes': data['box3d_lidar'].numpy(),
        'scores': data['scores'].numpy(),
        'classes': data['label_preds'].numpy()
    }

    pcd = o3d.geometry.PointCloud()
    if is_show_adv:
        pcd.points = o3d.utility.Vector3dVector(points_adv[:, :3])
    else:
        pcd.points = o3d.utility.Vector3dVector(points_innocent[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    visual = [pcd]
    # visual += plot_boxes(detections, thresh, rgb_colors)
    # o3d.visualization.draw_geometries(visual)

    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    for i in range(len(visual)):
        vis.add_geometry(visual[i])
    vis.run()
    vis.destroy_window()

def visualize_points_colorModification_debug(points_adv, points_innocent, data=None, is_show_adv=True):
    '''
    可视化修改位置
    '''
    ## visulize with color
    vis = o3d.visualization.Visualizer()

    thresh = 0.1
    points_diff = points_adv - points_innocent
    attach_num = int(np.linalg.norm(np.abs(points_diff[:, :3]).sum(-1), ord=0))
    print('attach_num = {}'.format(attach_num))
    colors = color_generation(np.abs(points_diff[:, :3]).sum(-1))

    pcd = o3d.geometry.PointCloud()
    if is_show_adv:
        pcd.points = o3d.utility.Vector3dVector(points_adv[:, :3])
    else:
        pcd.points = o3d.utility.Vector3dVector(points_innocent[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    visual = [pcd]
    if not data == None:
        detections = {
            'boxes': data['box3d_lidar'].numpy(),
            'scores': data['scores'].numpy(),
            'classes': data['label_preds'].numpy()
        }
        visual += plot_boxes(detections, thresh, rgb_colors)
    # o3d.visualization.draw_geometries(visual)

    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    for i in range(len(visual)):
        vis.add_geometry(visual[i])
    vis.run()
    vis.destroy_window()

def visualize_points_colorHeight(vis, data, pc_dir, key):
    '''
    可视化高度位置
    '''
    thresh = 0.3
    points_path = os.path.join(pc_dir, key + '-innocent.bin')
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)
    print(points.shape)

    ## visulize with color
    attach_rate = 0.02
    points_innocent = points

    h_low = -2.0
    h_up = -1.5
    colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
    # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
    for i in range(points.shape[0]):
        if h_low <= points[i, 2] and points[i, 2] <= h_up:
            colors[i, :] = [1, 1, 0]
    # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
    # visualize_points_colored(points, colors)
    # visualize_points(points)

    detections = {
        'boxes': data['box3d_lidar'].numpy(),
        'scores': data['scores'].numpy(),
        'classes': data['label_preds'].numpy()
    }

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    visual = [pcd]
    # visual += plot_boxes(detections, thresh, rgb_colors)
    # o3d.visualization.draw_geometries(visual)

    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    for i in range(len(visual)):
        vis.add_geometry(visual[i])
    vis.run()
    vis.destroy_window()


def visualize_points_without_color(points, data):
    detections = {
        'boxes': data['box3d_lidar'].numpy(),
        'scores': data['scores'].numpy(),
        'classes': data['label_preds'].numpy()
    }

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    visual = [pcd]
    num_dets = detections['scores'].shape[0]
    thresh = 0.0
    visual += plot_boxes(detections, thresh, rgb_colors)
    o3d.visualization.draw_geometries(visual)


def visualize_points_without_color_vis(points, vis, data):
    detections = {
        'boxes': data['box3d_lidar'].numpy(),
        'scores': data['scores'].numpy(),
        'classes': data['label_preds'].numpy()
    }

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    visual = [pcd]
    num_dets = detections['scores'].shape[0]
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

    vis.poll_events()
    vis.update_renderer()
    vis.clear_geometries()
    # vis.destroy_window()


import open3d.visualization.rendering as rendering


def visualize_points_without_color_save_image(points, vis, data):
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

    vis.create_window(width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    json_path = r'./angle_json/view_angle.json'
    if os.path.exists(json_path):
        render_option.load_from_json(r'./angle_json/view_angle.json')
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小

    for i in range(len(visual)):
        vis.add_geometry(visual[i])

    # vis.run()
    # vis.update_geometry(source)
    # vis.poll_events()
    # vis.update_renderer()
    vis.capture_screen_image('pc_test_vis.png')

    # vis.destroy_window()

    # render.scene.add_geometry()
    # render = rendering.OffscreenRenderer(1920, 1080)
    # center = [0, 0, 0]  # look_at target
    # eye = [0, 0, 80]  # camera position
    # up = [0, 1, 0]  # camera orientation
    # render.scene.camera.look_at(center, eye, up)
    # render.scene.set_background([0, 0, 0, 0])
    #
    # img = render.render_to_image()
    # o3d.io.write_image("pc_test.png", img, 9)


def visualize_points_without_color_without_detection(points, vis, window_name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    visual = [pcd]
    vis.create_window(window_name=window_name, width=1920, height=1080, left=50, top=50, visible=True)  # 创建窗口
    # if cnt == 0 or True:
    render_option: o3d.visualization.RenderOption = vis.get_render_option()  # 设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])  # 设置背景色（这里为黑色）
    render_option.point_size = 2.0  # 设置渲染点的大小
    for i in range(len(visual)):
        vis.add_geometry(visual[i])
    vis.run()
    ## o3d.visualization.clear_geometries()
    vis.destroy_window()


def save_3d_visualization(vis, pcd, save_path):
    visual = [pcd]
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
    # vis.run()
    # 更新场景和渲染器
    # vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    vis.capture_screen_image(save_path)
    # o3d.visualization.draw_geometries()
    vis.clear_geometries()
