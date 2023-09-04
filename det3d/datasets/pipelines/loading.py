import numpy as np

from pathlib import Path
import pickle
import os

from ..registry import PIPELINES


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


def read_file(path, tries=2, num_point_feature=4, virtual=False):
    if virtual:
        # WARNING: hard coded for nuScenes 
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]
        tokens = path.split('/')
        seg_path = os.path.join(*tokens[:-2], tokens[-2]+"_VIRTUAL", tokens[-1]+'.pkl.npy')
        data_dict = np.load(seg_path, allow_pickle=True).item()

        # remove reflectance as other virtual points don't have this value  
        virtual_points1 = data_dict['real_points'][:, [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] 
        virtual_points2 = data_dict['virtual_points']

        points = np.concatenate([points, np.ones([points.shape[0], 15-num_point_feature])], axis=1)
        virtual_points1 = np.concatenate([virtual_points1, np.zeros([virtual_points1.shape[0], 1])], axis=1)
        virtual_points2 = np.concatenate([virtual_points2, -1 * np.ones([virtual_points2.shape[0], 1])], axis=1)
        points = np.concatenate([points, virtual_points1, virtual_points2], axis=0).astype(np.float32)
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, virtual=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), virtual=virtual).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T

def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


def get_obj(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def get_num_voxels(voxel):
    num_points_per_voxel = np.zeros(shape=(voxel.shape[0],), dtype=np.int32)
    eps = 0.000001
    for voxel_id in range(voxel.shape[0]):
        for num in range(voxel.shape[1]):
            if np.abs(voxel[voxel_id, num, 0]) > eps:
                num_points_per_voxel[voxel_id] += 1
            elif np.sum(np.abs(voxel[voxel_id, num, :])) < 0.000001:
                break
    return num_points_per_voxel


# import tools.global_var as global_var
global_cfg = {}  # quick fix
import torch
from det3d.ops.point_cloud.point_cloud_ops import points_to_voxel


@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

        # adv
        self.adv_input_dir = kwargs.get("adv_input_dir", None)
        self.num_input_features = kwargs.get("num_input_features", None)
        self.is_adv_eval_entire_pc = kwargs.get("is_adv_eval_entire_pc", None)

    def __call__(self, res, info, **kwargs):

        res["type"] = self.type

        # if self.type == "NuScenesDataset" and global_cfg.is_point_interpolate:
        if self.type == "NuScenesDataset" and False:  # adv

            token = info['token']
            sample_dir = global_cfg.adv_eval_dir
            sample_path = os.path.join(sample_dir, token + '.bin')
            voxel_sample = np.fromfile(sample_path, dtype=np.float32).reshape(-1, 10, 5)
            voxel_sample

            num_points_per_voxel = get_num_voxels(voxel_sample)

            # points_mean = voxel_sample[:, :, :3].sum(
            #     dim=1, keepdim=False
            # ) / num_points_per_voxel.type_as(voxel_sample).view(-1, 1)
            adv_points_mean = torch.tensor(voxel_sample[:, :, :3]).sum(
                dim=1, keepdim=False
            ) / torch.tensor(num_points_per_voxel).type_as(torch.tensor(voxel_sample)).view(-1, 1)

            # 生成原始样本的point_mean
            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), virtual=res["virtual"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )

            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            # res["lidar"]["points"] = points
            # res["lidar"]["times"] = times
            # res["lidar"]["combined"] = np.hstack([points, times])

            # points_to_voxel()的输入
            voxel_size = np.array([0.075, 0.075, 0.2])
            coors_range = np.array([-54., -54., -5., 54., 54., 3.])
            max_points = 10
            reverse_index = True
            max_voxels = 120000
            voxels, _, num_points = points_to_voxel(np.hstack([points, times]),
                                                    voxel_size,
                                                    coors_range,
                                                    max_points,
                                                    reverse_index,
                                                    max_voxels)

            ori_points_mean = torch.tensor(voxels[:, :, :3]).sum(
                dim=1, keepdim=False
            ) / torch.tensor(num_points).type_as(torch.tensor(voxels)).view(-1, 1)

            # 对每个体素计算其内部的平均扰动量
            permutation_mean = adv_points_mean - ori_points_mean

        elif self.type == "NuScenesDataset" and False:  # 测试 re-Voxelize
            dir_save_innocent = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'  # 干净样本，点云格式，经过网络预处理的体素化，未做重新体素化
            dir_save_adv = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2/'  # 干净样本，点云格式，经过网络预处理的体素化，已做重新体素化
            token = info['token']

            # points_path = os.path.join(dir_save_innocent, token + '.bin')
            points_path = os.path.join(dir_save_adv, token + '.bin')
            print("points_path: {}".format(points_path))
            points_conbined = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)
            points = points_conbined[:, :4].reshape(-1, 4)
            times = points_conbined[:, -1].reshape(-1, 1)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])

            sweep_order = []
            lidar_path = ""
            res["lidar"]["sweep_order"] = np.array(sweep_order)
            res["lidar"]["lidar_path"] = lidar_path

        elif self.type == "NuScenesDataset" and False:  # 计算攻击前后的点云、voxel的统计特征
            dir_save_innocent = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'  # 干净样本，点云格式，经过网络预处理的体素化，未做重新体素化
            dir_save_origin_points = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-origin_points/'  # 干净样本，点云格式，未经过预处理
            dir_save_adv = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2/'  # 对抗样本，点云格式，经过网络预处理的体素化，已做重新体素化
            token = info['token']

            # points_path = os.path.join(dir_save_innocent, token + '.bin')
            points_path = os.path.join(dir_save_adv, token + '.bin')
            print("points_path: {}".format(points_path))
            points_conbined = np.fromfile(points_path, dtype=np.float32).reshape(-1, 5)
            points = points_conbined[:, :4].reshape(-1, 4)
            times = points_conbined[:, -1].reshape(-1, 1)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])

            sweep_order = []
            lidar_path = ""
            res["lidar"]["sweep_order"] = np.array(sweep_order)
            res["lidar"]["lidar_path"] = lidar_path

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), virtual=res["virtual"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )
            sweep_order = []
            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)
                sweep_order.append(i)  # adv

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])

            res["lidar"]["sweep_order"] = np.array(sweep_order)
            res["lidar"]["lidar_path"] = lidar_path

        elif self.type == "NuScenesDataset" and False:

            nsweeps = res["lidar"]["nsweeps"]

            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), virtual=res["virtual"])

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )
            sweep_order = []
            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)
                sweep_order.append(i)  # adv

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])

            res["lidar"]["sweep_order"] = np.array(sweep_order)
            res["lidar"]["lidar_path"] = lidar_path


        elif self.type == "NuScenesDataset":

            nsweeps = res["lidar"]["nsweeps"]
            lidar_path = Path(info["lidar_path"])

            if self.adv_input_dir == None:
                points = read_file(str(lidar_path), virtual=res["virtual"])

                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should equal to list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )
                sweep_order = []
                for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_sweep(sweep, virtual=res["virtual"])
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)
                    sweep_order.append(i)  # adv

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])

                res["lidar"]["sweep_order"] = np.array(sweep_order)
                res["lidar"]["lidar_path"] = lidar_path

            else:
                # token = os.path.basename(lidar_path)
                token = res["metadata"]["token"]
                if self.is_adv_eval_entire_pc == False:
                    adv_eval_path = os.path.join(self.adv_input_dir, token + '.bin')
                else:
                    adv_eval_path = os.path.join(self.adv_input_dir, token + '-conbined_adv.bin')
                points_adv = np.fromfile(adv_eval_path, dtype=np.float32).reshape(-1, self.num_input_features)
                res["lidar"]["points"] = points_adv[:, :-1]
                res["lidar"]["times"] = points_adv[:, -1]
                res["lidar"]["combined"] = points_adv

        elif self.type == "WaymoDataset" and False:
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points

            if nsweeps > 1:
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]

                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )

                sweep_order = []
                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)
                    sweep_order.append(i)  # adv

                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])

        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]

            if self.adv_input_dir == None:
                obj = get_obj(path)
                points = read_single_waymo(obj)

                res["lidar"]["points"] = points

                if nsweeps > 1:
                    sweep_points_list = [points]
                    sweep_times_list = [np.zeros((points.shape[0], 1))]

                    assert (nsweeps - 1) == len(
                        info["sweeps"]
                    ), "nsweeps {} should be equal to the list length {}.".format(
                        nsweeps, len(info["sweeps"])
                    )

                    sweep_order = []
                    for i in range(nsweeps - 1):
                        sweep = info["sweeps"][i]
                        points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                        sweep_points_list.append(points_sweep)
                        sweep_times_list.append(times_sweep)
                        sweep_order.append(i)  # adv

                    points = np.concatenate(sweep_points_list, axis=0)
                    times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

                    res["lidar"]["points"] = points
                    res["lidar"]["times"] = times
                    res["lidar"]["combined"] = np.hstack([points, times])

                    res["lidar"]["sweep_order"] = np.array(sweep_order)
                    res["lidar"]["lidar_path"] = path
            else:
                token = os.path.basename(path)
                if self.is_adv_eval_entire_pc == False:
                    adv_eval_path = os.path.join(self.adv_input_dir, token + '.bin')
                else:
                    adv_eval_path = os.path.join(self.adv_input_dir, token + '-conbined_adv.bin')
                points_adv = np.fromfile(adv_eval_path, dtype=np.float32).reshape(-1, self.num_input_features)
                res["lidar"]["points"] = points_adv[:, :-1]
                res["lidar"]["times"] = points_adv[:, -1]
                res["lidar"]["combined"] = points_adv
        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        else:
            pass 

        return res, info
