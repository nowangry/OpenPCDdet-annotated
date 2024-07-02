from __future__ import division

import re
from collections import OrderedDict, defaultdict
from functools import partial

try:
    import apex
except:
    print("No APEX!")

import numpy as np
import torch
from det3d.builder import _create_learning_rate_scheduler

# from det3d.datasets.kitti.eval_hooks import KittiDistEvalmAPHook, KittiEvalmAPHookV2
from det3d.core import DistOptimizerHook
from det3d.datasets import DATASETS, build_dataloader
from det3d.solver.fastai_optim import OptimWrapper
from det3d.torchie.trainer import DistSamplerSeedHook, Trainer, obj_from_dict
from det3d.utils.print_utils import metric_to_str
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from .env import get_root_logger
from pathlib import Path
import open3d as o3d

# from tools.analysis.visual_utily import *
from det3d.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from det3d.ops.point_cloud.point_cloud_ops import points_to_voxel
from scipy.interpolate import interpn
import numba


def example_to_device(example, device=None, non_blocking=False) -> dict:
    assert device is not None

    example_torch = {}
    float_names = ["voxels", "bev_map"]
    for k, v in example.items():
        if k in ["anchors", "anchors_mask", "reg_targets", "reg_weights", "labels", 'points',
                 "hm", "anno_box", "ind", "mask", 'cat', 'points']:  # adv
            example_torch[k] = [res.to(device, non_blocking=non_blocking) for res in v]
        elif k in [
            "voxels",
            "bev_map",
            "coordinates",
            "num_points",
            "num_voxels",
            "cyv_voxels",
            "cyv_num_voxels",
            "cyv_coordinates",
            "cyv_num_points"
        ]:
            example_torch[k] = v.to(device, non_blocking=non_blocking)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                # calib[k1] = torch.tensor(v1, dtype=dtype, device=device)
                calib[k1] = torch.tensor(v1).to(device, non_blocking=non_blocking)
            example_torch[k] = calib
        else:
            example_torch[k] = v

    return example_torch


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError("{} is not a tensor or list of tensors".format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

    log_vars["loss"] = loss
    for name in log_vars:
        log_vars[name] = log_vars[name].item()

    return loss, log_vars


def parse_second_losses(losses):
    log_vars = OrderedDict()
    loss = sum(losses["loss"])
    for loss_name, loss_value in losses.items():
        if loss_name == "loc_loss_elem":
            log_vars[loss_name] = [[i.item() for i in j] for j in loss_value]
        else:
            log_vars[loss_name] = [i.item() for i in loss_value]

    return loss, log_vars


import os
from pathlib import Path
import open3d as o3d


def save_point_cloud(point_could, save_path):
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_could[:, :3])
    o3d.io.write_point_cloud(save_path, pcd)


def save_sample(origin_voxels, example, save_path):
    dict = {
        'origin_voxels': origin_voxels,
        'num_points': example['num_points'],
        'coordinates': example['coordinates'],
        'adv_voxels': example['voxels']
    }
    np.save(save_path, dict)
    del dict


def permutation_mask(voxel_shape, num_points):
    per_mask = permutation_mask_numba(tuple(list(voxel_shape)), num_points.detach().cpu().numpy())
    return torch.from_numpy(per_mask).cuda()


@numba.jit(nopython=True)
def permutation_mask_numba(voxel_shape, num_points):
    # voxel = np.zeros(shape=voxel_shape, dtype=np.float32)
    per_mask = np.zeros(shape=voxel_shape, dtype=np.bool_)
    for voxel_index in range(voxel_shape[0]):
        per_mask[voxel_index, 0:num_points[voxel_index], 0:3] = True
    return per_mask


def permutation_mask_cuda(voxel_shape, num_points):
    per_mask = torch.zeros(size=(voxel_shape), dtype=torch.bool).cuda()
    for voxel_index in range(voxel_shape[0]):
        per_mask[voxel_index, 0:num_points[voxel_index], 0:3] = True
    return per_mask

def adv_evaluation(model, example, device, token, **kwargs):
    cfg = kwargs['cfg']
    args = kwargs['args']
    # token = data['metadata'][0]['token']
    # token = kwargs['token']
    adv_eval_dir = cfg.adv_eval_dir
    is_adv_eval_entire_pc = cfg.get('is_adv_eval_entire_pc', False)
    if not is_adv_eval_entire_pc:
        adv_eval_path = os.path.join(adv_eval_dir, str(token) + '.bin')
    else:
        adv_eval_path = os.path.join(args.outputs_dir, str(token) + '-conbined_adv.bin')
    assert os.path.exists(adv_eval_path)
    # if not os.path.exists(adv_eval_path):
    #     print('=== None model output ===')
    #     return None
    points_adv = np.fromfile(adv_eval_path, dtype=np.float32).reshape(-1, cfg.model.reader.num_input_features)
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(points_adv)
    save_to_example(example, voxels, coordinates, num_points, device)

    # data['voxels'] = adv_sample.reshape(data['voxels'].shape)
    # data 迁移到cuda上
    # example = example_to_device(data, device, non_blocking=False)
    # del data
    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs)
    return predictions


def save_to_example(example, voxels, coordinates, num_points, device):
    example['voxels'] = torch.from_numpy(voxels).to(device, non_blocking=False)
    Coors = collate_kitti_v0(coordinates)
    example['coordinates'] = Coors.to(device, non_blocking=False)
    example['num_points'] = torch.from_numpy(num_points).to(device, non_blocking=False)
    example['num_voxels'] = np.array([example['voxels'].shape[0]], dtype=np.int64)


def save_to_example_light(example, voxels, device):
    example['voxels'] = torch.from_numpy(voxels).to(device, non_blocking=False)
    example['num_voxels'] = np.array([example['voxels'].shape[0]], dtype=np.int64)


def FGSM_Attack_v0(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.FGSM.Epsilon

    example['voxels'].requires_grad = True
    origin_voxels = example['voxels'].clone()
    losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)

    model.zero_grad()
    grad_voxel = \
        torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]

    # mask voxel内空白的点、intensity、timestamp
    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
    permutation = Epsilon * grad_voxel.sign() * perm_mask
    example['voxels'] = example['voxels'] + permutation

    if cfg.is_point_interpolate:
        print(" === point_interpolate === ")
        adv_points_mean = example['voxels'][:, :, :3].sum(
            dim=1, keepdim=False
        ) / example['num_points'].type_as(example['voxels']).view(-1, 1)

        voxel_size = np.array([0.075, 0.075, 0.2])
        coors_range = np.array([-54., -54., -5., 54., 54., 3.])
        max_points = 10
        reverse_index = True
        max_voxels = 120000
        voxels, _, num_points = points_to_voxel(example['points'][0].numpy(),
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

        # interpn(points, values, point)
        # interpn(points, values, example['points'][0][:, :3])

        ndim = 3
        # for j in range(ndim):  # 遍历x/y/z维度
        #     c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])

    token = example['metadata'][0]['token']
    save_dir = os.path.join(cfg.FGSM.save_dir, 'eps_{}'.format(Epsilon))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, str(token) + '.npy')
    save_sample(origin_voxels, example, save_path)
    # load_dict = np.load(save_path, allow_pickle=True).item()
    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def test_reVoxelize(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    points_adv = example['points'][0].detach().cpu().numpy()
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(points_adv)
    save_to_example(example, voxels, coordinates, num_points, device)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def FGSM_Attack_test_reVoxelize(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.FGSM.Epsilon
    dir_save_origin_points = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-origin_points/'  # 干净样本，点云格式，未经过预处理
    dir_save_innocent = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-innocent/'  # 干净样本，点云格式，经过网络预处理的体素化，未做重新体素化
    dir_save_adv = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2/'  # 对抗样本，点云格式，经过网络预处理的体素化，已做重新体素化
    dir_save_adv_reVoxelize = r'/home/jqwu/Datasets/Outputs/centerpoint-adv/test_reVoxelize/v1.0-mini-FGSM-eps_0.2-reVoxelize/'
    token = example['metadata'][0]['token']
    # [( '/home/jqwu/Datasets/nuScenes/v1.0-mini/samples/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542800861447555.pcd.bin')]

    # 保存 points_origin
    # points_origin = example['points'][0].cpu().detach().numpy().reshape(-1, 5)
    # points_origin.tofile(os.path.join(dir_save_origin_points, token + '.bin'))

    # 保存points_innocent
    # points_innocent = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
    #                               example['num_points'].cpu().detach().numpy())
    # points_innocent.tofile(os.path.join(dir_save_innocent, token + '.bin'))

    # 保存 points_adv
    example['voxels'].requires_grad = True
    origin_voxels = example['voxels'].clone()
    losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)

    model.zero_grad()
    grad_voxel = \
        torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]

    # mask voxel内空白的点、intensity、timestamp
    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
    permutation = Epsilon * grad_voxel.sign() * perm_mask
    example['voxels'] = example['voxels'] + permutation

    points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                      example['num_points'].cpu().detach().numpy())
    # points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))

    # 保存 points_adv_reVoxelize
    ## coordinate adjustment
    voxelization_cfg = ConfigDict(
        {'cfg': {'range': [-54, -54, -5.0, 54, 54, 3.0], 'voxel_size': [0.075, 0.075, 0.2], 'max_points_in_voxel': 10,
                 'max_voxel_num': [120000, 160000]}})
    Voxelize = Voxelization(**voxelization_cfg)
    max_voxels = 120000
    voxels, coordinates, num_points = Voxelize.voxel_generator.generate(
        points_adv, max_voxels=max_voxels
    )
    Coors = collate_kitti_v0(coordinates)
    example['voxels'] = torch.from_numpy(voxels).to(device, non_blocking=False)
    example['coordinates'] = Coors.to(device, non_blocking=False)
    example['num_points'] = torch.from_numpy(num_points).to(device, non_blocking=False)
    example['num_voxels'] = np.array([example['voxels'].shape[0]], dtype=np.int64)

    points_adv_reVoxelize = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                                 example['num_points'].cpu().detach().numpy())
    points_adv_reVoxelize.tofile(os.path.join(dir_save_adv_reVoxelize, token + '.bin'))

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def stat_points_voxelized(example):
    points = example['points'][0].cpu().detach().numpy().reshape(-1, 5)

    voxel_range = voxelization_cfg.cfg.range  # [-54, -54, -5.0, 54, 54, 3.0]
    voxels, coordinates, num_points = Voxelize.voxel_generator.generate(
        points, max_voxels=max_voxels
    )
    points_voxelized = voxel_to_point_numba(voxels, num_points)

    points_out_of_range = []
    for i in range(points.shape[0]):
        if (voxel_range[0] <= points[i, 0] and points[i, 0] <= voxel_range[3]) \
                and (voxel_range[1] <= points[i, 1] and points[i, 1] <= voxel_range[4]) \
                and (voxel_range[2] <= points[i, 2] and points[i, 2] <= voxel_range[5]):
            continue
        points_out_of_range.append(points[i, :])
    points_out_of_range = np.array(points_out_of_range)
    print(points.shape)
    print(points_out_of_range.shape)
    print(points_voxelized.shape)
    points_conbined = np.concatenate([points_out_of_range, points_voxelized], axis=0)
    print(points_conbined.shape)
    visualize_points(points_conbined)


def FGSM_Attack_debug(model, example, device, **kwargs):
    # token = example['metadata'][0]['token']
    # if 'a860c02c06e54d9dbe83ce7b694f6c17' not in token:
    #     with torch.no_grad():
    #         predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    #     return predictions
    # stat_points_voxelized(example)

    # cfg = kwargs['cfg']
    # Epsilon = cfg.FGSM.Epsilon
    #
    # example['voxels'].requires_grad = True
    # origin_voxels = example['voxels'].clone()
    # losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
    #
    # model.zero_grad()
    # grad_voxel = \
    #     torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
    #
    # # mask voxel内空白的点、intensity、timestamp
    # perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
    # permutation = Epsilon * grad_voxel.sign() * perm_mask
    # example['voxels'] = example['voxels'] + permutation
    #
    # ## coordinate adjustment
    # points = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
    #                               example['num_points'].cpu().detach().numpy())
    # voxels, coordinates, num_points = Voxelize.voxel_generator.generate(
    #     points, max_voxels=max_voxels
    # )
    # example['voxels'] = torch.from_numpy(voxels).to(device, non_blocking=False)
    # Coors = collate_kitti_v0(coordinates)
    # example['coordinates'] = Coors.to(device, non_blocking=False)
    # example['num_points'] = torch.from_numpy(num_points).to(device, non_blocking=False)
    # example['num_voxels'] = np.array([example['voxels'].shape[0]], dtype=np.int64)
    #
    # # save
    # token = example['metadata'][0]['token']
    # save_dir = os.path.join(cfg.FGSM.save_dir, 'eps_{}'.format(Epsilon))
    # os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, str(token) + '.npy')
    # save_sample(origin_voxels, example, save_path)
    # # load_dict = np.load(save_path, allow_pickle=True).item()

    cfg = kwargs['cfg']
    voxelization = cfg.voxelization
    Epsilon = cfg.FGSM.Epsilon
    dir_save_adv = cfg.FGSM.save_dir

    example['voxels'].requires_grad = True
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    # voxel to point
    # points_innocent_v0 = voxel_to_point_torch(example['voxels'],
    #                                           example['num_points'])

    # example_m2 = example['voxels'] * 2
    # example_m2.backward()
    # print(example['voxels'].grad)
    # points_innocent_v0_m3 = points_innocent_v0 * 3
    # points_innocent_v0_m3.backward()
    # print(example['voxels'].grad)

    points_adv = points_innocent_v0.copy()

    # Get gradient
    example['voxels'].requires_grad = True
    losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
    model.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
    grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                 example['num_points'].cpu().detach().numpy())
    # Add permutation
    permutation = Epsilon * np.sign(grad_points[:, :3])
    points_adv[:, :3] = points_adv[:, :3] + permutation

    # point to voxel
    # if kwargs['cfg'].model.type == 'PointPillars':
    voxels, coordinates, num_points = voxelization.voxel_generator.generate(
        points_adv, max_voxels=max_voxels
    )
    save_to_example(example, voxels, coordinates, num_points, device)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def FGSM_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.FGSM.Epsilon
    dir_save_adv = cfg.FGSM.save_dir

    # voxel to point
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    points_adv = points_innocent_v0.copy()

    # Get gradient
    example['voxels'].requires_grad = True
    losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
    model.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
    grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                 example['num_points'].cpu().detach().numpy())
    # Add permutation
    permutation = Epsilon * np.sign(grad_points[:, :3])
    points_adv[:, :3] = points_adv[:, :3] + permutation

    # point to voxel
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(points_adv)
    save_to_example(example, voxels, coordinates, num_points, device)

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    # grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    # if "NuScenesDataset" in cfg.dataset_type:
    save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, example['points'][0].shape[1]),
                       points_innocent_v0, points_adv,
                       token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def FGSM_Attack_sign(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.FGSM.Epsilon
    print(' ===== no sign(): permutation = Epsilon * grad_voxel * perm_mask =======')
    dir_save_adv = cfg.FGSM.save_dir  # 对抗样本，点云格式，经过网络预处理的体素化，已做重新体素化

    example['voxels'].requires_grad = True
    origin_voxels = example['voxels'].clone()
    losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)

    model.zero_grad()
    grad_voxel = \
        torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]

    # mask voxel内空白的点、intensity、timestamp
    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
    # permutation = Epsilon * grad_voxel.sign() * perm_mask
    permutation = Epsilon * grad_voxel * perm_mask
    example['voxels'] = example['voxels'] - permutation

    ## coordinate adjustment
    voxelization_cfg = ConfigDict(
        {'cfg': {'range': [-54, -54, -5.0, 54, 54, 3.0], 'voxel_size': [0.075, 0.075, 0.2], 'max_points_in_voxel': 10,
                 'max_voxel_num': [120000, 160000]}})
    Voxelize = Voxelization(**voxelization_cfg)
    max_voxels = 120000
    points = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                  example['num_points'].cpu().detach().numpy())
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
        points, max_voxels=max_voxels
    )
    example['voxels'] = torch.from_numpy(voxels).to(device, non_blocking=False)
    Coors = collate_kitti_v0(coordinates)
    example['coordinates'] = Coors.to(device, non_blocking=False)
    example['num_points'] = torch.from_numpy(num_points).to(device, non_blocking=False)
    example['num_voxels'] = np.array([example['voxels'].shape[0]], dtype=np.int64)

    # save
    token = example['metadata'][0]['token']
    points.tofile(os.path.join(dir_save_adv, token + '.bin'))

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions

def PGD_Attack_v0(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.PGD.eps
    eps_iter = cfg.PGD.eps_iter
    num_steps = cfg.PGD.num_steps
    ori_sample = example['voxels'].clone()
    save_dir = cfg.PGD.save_dir
    # adv_sample = ori_sample.clone()

    # if cfg.PGD.random_start:
    #     # Starting at a uniformly random point
    #     adv_sample = adv_sample + torch.empty_like(adv_sample).uniform_(-eps, eps)
    #     adv_sample = torch.clamp(adv_sample, min=0, max=1).detach()

    example['voxels'].requires_grad = True
    # mask voxel内空白的点、intensity、timestamp
    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
    for i in range(num_steps):
        model.zero_grad()
        # Calculate loss
        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)

        # Update adversarial images
        grad = torch.autograd.grad(sum(losses['loss']), example['voxels'],
                                   retain_graph=False, create_graph=False)[0]

        permutation = eps_iter * grad.sign() * perm_mask
        example['voxels'] = example['voxels'] + permutation
        delta = torch.clamp(example['voxels'] - ori_sample, min=-eps, max=eps)
        example['voxels'] = ori_sample + delta
        cfg.logger.info('Iter {}: detection_loss={:.6f}'.format(i, sum(losses['loss'])))

    # save
    token = example['metadata'][0]['token']
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, str(token) + '.npy')
    save_sample(ori_sample, example, save_path)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def RoadAdv_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.RoadAdv.Epsilon
    dir_save_adv = cfg.RoadAdv.save_dir
    os.makedirs(dir_save_adv, exist_ok=True)

    # voxel to point
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    points_adv = points_innocent_v0.copy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_innocent_v0[:, :3])
    _, inliers = pcd.segment_plane(distance_threshold=0.30,
                                   ransac_n=5,
                                   num_iterations=1000)

    # Get gradient
    example['voxels'].requires_grad = True
    losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
    model.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
    grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                 example['num_points'].cpu().detach().numpy())
    # Add permutation
    permutation = Epsilon * np.sign(grad_points[:, :3])
    # points_adv[:, :3] = points_adv[:, :3] + permutation
    points_adv[inliers, :3] = points_adv[inliers, :3] + permutation[inliers, :3]

    # point to voxel
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
        points_adv
    )
    save_to_example(example, voxels, coordinates, num_points, device)

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


@numba.jit(nopython=True)
def voxel_to_point_numba(voxels, num_points):
    points = np.zeros(shape=(num_points.sum(), voxels.shape[2]), dtype=np.float32)
    cnt = 0
    for voxel_idx, num_point in enumerate(num_points):
        for point_idx in range(num_point):
            points[cnt, :] = voxels[voxel_idx, point_idx, :]
            cnt += 1
    assert cnt == num_points.sum()
    return points


def voxel_to_point_torch(voxels, num_points):
    points = torch.zeros(size=(num_points.sum(), voxels.shape[2]), dtype=torch.float32, requires_grad=False)
    cnt = 0
    for voxel_idx, num_point in enumerate(num_points):
        points[cnt: cnt + num_point, :] = voxels[voxel_idx, :num_point, :].clone()
        cnt += num_point
    assert cnt == num_points.sum()
    points.requires_grad = True
    return points


@numba.jit(nopython=True)
def voxel_to_point_numba_voxelMap(voxels, num_points):
    points = np.zeros(shape=(num_points.sum(), voxels.shape[2]), dtype=np.float32)
    cnt = 0
    voxelMap = {}
    for voxel_idx, num_point in enumerate(num_points):
        for point_idx in range(num_point):
            points[cnt, :] = voxels[voxel_idx, point_idx, :]
            voxelMap[cnt] = (voxel_idx, point_idx)
            cnt += 1
    assert cnt == num_points.sum()
    return points, voxelMap


@numba.jit(nopython=True)
def points_to_voxel_numba_voxelMap(points, voxelMap, voxel_shape):
    voxel = np.zeros(shape=voxel_shape, dtype=np.float32)
    for cnt in range(points.shape[0]):
        (voxel_idx, point_idx) = voxelMap[cnt]
        voxel[voxel_idx, point_idx, :] = points[cnt, :]
    return voxel


@numba.jit(nopython=True)
def voxel_to_point_numba_for_iterAttack(voxels, num_points, voxels_innocent_v0):
    points = np.zeros(shape=(num_points.sum(), voxels.shape[2]), dtype=np.float32)
    points_innocent_ori = np.zeros(shape=(num_points.sum(), voxels.shape[2]), dtype=np.float32)
    cnt = 0
    for voxel_idx, num_point in enumerate(num_points):
        for point_idx in range(num_point):
            points[cnt, :] = voxels[voxel_idx, point_idx, :]
            points_innocent_ori[cnt, :] = voxels_innocent_v0[voxel_idx, point_idx, :]
            cnt += 1
    assert cnt == num_points.sum()
    return points, points_innocent_ori


def voxel_to_point_numba_for_iterAttack_torch(voxels, num_points, voxels_innocent_v0):
    points = torch.tensor(np.zeros(shape=(num_points.sum(), voxels.shape[2]), dtype=np.float32))
    points_innocent_ori = torch.tensor(np.zeros(shape=(num_points.sum(), voxels.shape[2]), dtype=np.float32))
    cnt = 0
    for voxel_idx, num_point in enumerate(num_points):
        for point_idx in range(num_point):
            points[cnt, :] = voxels[voxel_idx, point_idx, :].clone()
            points_innocent_ori[cnt, :] = voxels_innocent_v0[voxel_idx, point_idx, :].clone()
            cnt += 1
    assert cnt == num_points.sum()
    return points, points_innocent_ori


@numba.jit(nopython=True)
def voxel_to_point_numba_for_iterAttack_momentum(voxels, num_points, voxels_innocent_v0, voxel_momentum):
    points = np.zeros(shape=(num_points.sum(), voxels.shape[2]), dtype=np.float32)
    points_innocent_ori = np.zeros(shape=(num_points.sum(), voxels.shape[2]), dtype=np.float32)
    momentum = np.zeros(shape=(num_points.sum(), voxels.shape[2]), dtype=np.float32)
    cnt = 0
    for voxel_idx, num_point in enumerate(num_points):
        for point_idx in range(num_point):
            points[cnt, :] = voxels[voxel_idx, point_idx, :]
            points_innocent_ori[cnt, :] = voxels_innocent_v0[voxel_idx, point_idx, :]
            momentum[cnt, :] = voxel_momentum[voxel_idx, point_idx, :]
            cnt += 1
    assert cnt == num_points.sum()
    return points, points_innocent_ori, momentum


@numba.jit(nopython=True)
def grad_voxel_to_grad_point_numba(grad_voxels, num_points):
    grad_points = np.zeros(shape=(num_points.sum(), grad_voxels.shape[2]), dtype=np.float32)
    cnt = 0
    for voxel_idx, num_point in enumerate(num_points):
        for point_idx in range(num_point):
            grad_points[cnt, :] = grad_voxels[voxel_idx, point_idx, :]
            cnt += 1
    assert cnt == num_points.sum()
    return grad_points


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


def object_point_filter(points, gt_boxes_and_cls):
    is_object_point = np.zeros(points.shape[0], dtype=np.bool)
    for i in range(points.shape[0]):
        for j in range(gt_boxes_and_cls.shape[0]):
            if np.abs(gt_boxes_and_cls[j, :]).sum() == 0:
                break
            is_object_point[i] = check_point_in_box(points[i, :3], gt_boxes_and_cls[j, :7])
    return is_object_point


@numba.jit(nopython=True)
def object_point_filter_numba(points, gt_boxes_and_cls):
    is_object_point = np.zeros(points.shape[0], dtype=np.bool)
    for i in range(points.shape[0]):
        for j in range(gt_boxes_and_cls.shape[0]):
            if np.abs(gt_boxes_and_cls[j, :]).sum() == 0:
                break
            # is_object_point[i] = check_point_in_box(points[i, :3], gt_boxes_and_cls[j, :7])
            pts = points[i, :3]
            box = gt_boxes_and_cls[j, :7]
            shift_x = pts[0] - box[0]
            shift_y = pts[1] - box[1]
            shift_z = pts[2] - box[2]
            cos_a = np.cos(box[6])
            sin_a = np.sin(box[6])
            dx, dy, dz = box[3], box[4], box[5]
            local_x = shift_x * cos_a + shift_y * sin_a
            local_y = shift_y * cos_a - shift_x * sin_a
            if (abs(shift_z) > dz / 2.0 or abs(local_x) > dx / 2.0 or abs(local_y) > dy / 2.0):
                is_object_point[i] = False
            else:
                is_object_point[i] = True

    return is_object_point


@numba.jit(nopython=True)
def voxel_to_point_with_order_numba(voxels, num_points):
    points = np.zeros(shape=(num_points.sum(), 5), dtype=np.float32)
    order = np.zeros(shape=(num_points.sum(), 2), dtype=np.float32)
    cnt = 0
    for voxel_idx, num_point in enumerate(num_points):
        for point_idx in range(num_point):
            points[cnt, :] = voxels[voxel_idx, point_idx, :]
            order[cnt, 0] = voxel_idx
            order[cnt, 1] = num_point
            cnt += 1
    assert cnt == num_points.sum()
    return points, order


def collate_kitti_v0(coordinates):
    coors = []
    for i, coor in enumerate([coordinates]):
        coor_pad = np.pad(
            coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
        )
        coors.append(coor_pad)
    Coors = torch.tensor(np.concatenate(coors, axis=0))
    return Coors

from det3d.torchie.parallel.collate import collate_kitti

def PGD_CoorAdjust_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.PGD_CoorAdjust.eps
    eps_iter = cfg.PGD_CoorAdjust.eps_iter
    num_steps = cfg.PGD_CoorAdjust.num_steps
    dir_save_adv = cfg.PGD_CoorAdjust.save_dir
    os.makedirs(dir_save_adv, exist_ok=True)

    # voxel to point
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    for n_step in range(num_steps):
        if not n_step == 0:
            # voxel to point
            points_adv, points_innocent_ori = voxel_to_point_numba_for_iterAttack(
                example['voxels'].cpu().detach().numpy(),
                example['num_points'].cpu().detach().numpy(),
                voxel_innocent_ori)
            # points_innocent_ori 跟随迭代更新，但是保存最初的信息
            # np.abs(points_adv - points_innocent_ori)[:, :3].mean()
            # (np.abs(points_adv - points_innocent_ori)[:, :3] > 0.10001).sum()
        else:
            points_adv = points_innocent_v0.copy()
            points_innocent_ori = points_innocent_v0.copy()

            if cfg.PGD_CoorAdjust.random_start:
                # Starting at a uniformly random point
                points_adv[:, :3] = points_adv[:, :3] + torch.empty_like(torch.tensor(points_adv[:, :3])).uniform_(-eps,
                                                                                                                   eps).numpy()
                # point to voxel
                voxels, coordinates, num_points, voxel_innocent_ori = \
                    cfg.voxelization.voxel_generator.generate_for_iterAttack(
                        points_adv, points_innocent_ori=points_innocent_ori
                    )
                save_to_example(example, voxels, coordinates, num_points, device)
                # voxel to point
                points_adv, points_innocent_ori = voxel_to_point_numba_for_iterAttack(
                    example['voxels'].cpu().detach().numpy(),
                    example['num_points'].cpu().detach().numpy(),
                    voxel_innocent_ori)

        # np.abs(points_innocent_ori - points_innocent).sum() / points_innocent_ori.shape[0] / 3
        # np.abs(points_innocent_ori[:, :3] - points_adv[:, :3]).max()
        # Get gradient
        example['voxels'].requires_grad = True
        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        grad_voxel = torch.autograd.grad(sum(losses['loss']), example['voxels'],
                                         retain_graph=False, create_graph=False)[0]
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())

        # Add permutation
        permutation = eps_iter * np.sign(grad_points[:, :3])
        points_adv[:, :3] = points_adv[:, :3] + permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent_ori[:, :3]), min=-eps,
                            max=eps).numpy()
        points_adv[:, :3] = points_innocent_ori[:, :3] + delta

        cfg.logger.info('Iter {}: detection_loss={:.6f}'.format(n_step, sum(losses['loss'])))

        # point to voxel
        voxels, coordinates, num_points, voxel_innocent_ori = \
            cfg.voxelization.voxel_generator.generate_for_iterAttack(
                points_adv, points_innocent_ori=points_innocent_ori
            )
        save_to_example(example, voxels, coordinates, num_points, device)

    # save
    assert points_adv.shape == points_innocent_ori.shape
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_ori.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))
    # points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    # grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    if "NuScenesDataset" in cfg.dataset_type:
        save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, example['points'][0].shape[1]),
                           points_innocent_ori, points_adv,
                           token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def PGD_CoorAdjust_Attack_light(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.PGD_CoorAdjust.eps
    eps_iter = cfg.PGD_CoorAdjust.eps_iter
    num_steps = cfg.PGD_CoorAdjust.num_steps
    dir_save_adv = cfg.PGD_CoorAdjust.save_dir
    os.makedirs(dir_save_adv, exist_ok=True)

    voxel_innocent_ori = example['voxels'].clone()
    example['voxels'].requires_grad = True
    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])

    for n_step in range(num_steps):
        if n_step == 0:
            if cfg.PGD_CoorAdjust.random_start:
                # Starting at a uniformly random point
                perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
                permutation = torch.empty_like(example['voxels']).uniform_(-eps, eps) * perm_mask
                example['voxels'] = example['voxels'] + permutation

        # Calculate loss
        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)

        # Update adversarial images
        grad = torch.autograd.grad(sum(losses['loss']), example['voxels'],
                                   retain_graph=False, create_graph=False)[0]
        # mask voxel内空白的点、intensity、timestamp
        permutation = eps_iter * grad.sign() * perm_mask
        example['voxels'] = example['voxels'] + permutation
        delta = torch.clamp(example['voxels'] - voxel_innocent_ori, min=-eps, max=eps)
        example['voxels'] = voxel_innocent_ori + delta
        cfg.logger.info('Iter {}: detection_loss={:.6f}'.format(n_step, sum(losses['loss'])))

    # light version
    ## coordinate adjustment
    points_adv, points_innocent_ori = voxel_to_point_numba_for_iterAttack(
        example['voxels'].cpu().detach().numpy(),
        example['num_points'].cpu().detach().numpy(),
        voxel_innocent_ori.cpu().detach().numpy())
    # point to voxel
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(points_adv)
    save_to_example(example, voxels, coordinates, num_points, device)

    # save
    assert points_adv.shape == points_innocent_ori.shape
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_ori.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def point_in_range(point, voxel_range):
    if (voxel_range[0] <= point[0] and point[0] < voxel_range[3]) \
            and (voxel_range[1] <= point[1] and point[1] < voxel_range[4]) \
            and (voxel_range[2] <= point[2] and point[2] < voxel_range[5]):
        return True
    else:
        return False


def save_origin_points(cfg, points, points_innocent_ori, points_adv, token, dir_save_adv):
    return
    # if "NuScenesDataset" in cfg.dataset_type:
    voxel_range = cfg.voxel_generator.range
    points_out_of_range = []
    for i in range(points.shape[0]):
        if point_in_range(points[i, :], voxel_range):
            continue
        else:
            points_out_of_range.append(points[i, :])
    # if "NuScenesDataset" in cfg.dataset_type:
    # points_conbined_innocent = np.concatenate([points_out_of_range, points_innocent_ori], axis=0)
    # points_conbined_innocent.tofile(os.path.join(dir_save_adv, token + '-conbined_innocent_ori.bin'))
    points_conbined_adv = np.concatenate([points_out_of_range, points_adv], axis=0)
    points_conbined_adv.tofile(os.path.join(dir_save_adv, token + '-conbined_adv.bin'))

def save_origin_points_cuda(cfg, points, points_innocent_ori, points_adv, token, dir_save_adv, device):
    voxel_range = cfg.voxel_generator.range
    points_out_of_range = []
    for i in range(points.shape[0]):
        if point_in_range(points[i, :], voxel_range):
            continue
        else:
            points_out_of_range.append(points[i, :].cpu().detach().numpy())
    points_out_of_range = torch.from_numpy(np.asarray(points_out_of_range)).to(device)
    points_conbined_innocent = torch.cat([points_out_of_range, points_innocent_ori], dim=0)
    points_conbined_adv = torch.cat([points_out_of_range, points_adv], dim=0)
    points_conbined_innocent.cpu().detach().numpy().tofile(
        os.path.join(dir_save_adv, token + '-conbined_innocent_ori.bin'))
    points_conbined_adv.cpu().detach().numpy().tofile(os.path.join(dir_save_adv, token + '-conbined_adv.bin'))

def MI_FGSM_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.MI_FGSM.eps
    eps_iter = cfg.MI_FGSM.eps_iter
    num_steps = cfg.MI_FGSM.num_steps
    decay = cfg.MI_FGSM.decay
    L_norm = cfg.MI_FGSM.L_norm
    dir_save_adv = cfg.MI_FGSM.save_dir

    ##### voxel to point
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    momentum = np.zeros_like(points_innocent_v0)
    for n_step in range(num_steps):
        if not n_step == 0:
            #### voxel to point
            points_adv, points_innocent_ori, momentum = voxel_to_point_numba_for_iterAttack_momentum(
                example['voxels'].cpu().detach().numpy(),
                example['num_points'].cpu().detach().numpy(),
                voxel_innocent_ori, voxel_momentum)
            # points_innocent_ori 跟随迭代更新，但是保存最初的信息
            # np.abs(points_adv - points_innocent_ori)[:, :3].mean()
            # (np.abs(points_adv - points_innocent_ori)[:, :3] > 0.10001).sum()
        else:
            points_adv = points_innocent_v0.copy()
            points_innocent_ori = points_innocent_v0.copy()

        # np.abs(points_innocent_ori - points_innocent).sum() / points_innocent_ori.shape[0] / 3
        # np.abs(points_innocent_ori[:, :3] - points_adv[:, :3]).max()
        ###### Get gradient
        example['voxels'].requires_grad = True
        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        grad_voxel = torch.autograd.grad(sum(losses['loss']), example['voxels'],
                                         retain_graph=False, create_graph=False)[0]
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        if 'L1' in L_norm:
            grad_points_norm = torch.mean(torch.abs(torch.tensor(grad_points)), dim=(1), keepdim=True).numpy()
        elif 'L2' in L_norm:
            grad_points_norm = torch.norm(torch.tensor(grad_points).view(grad_points.shape[0], -1), dim=1,
                                          keepdim=True).numpy()
        grad_points = grad_points / (grad_points_norm + 1e-10)
        grad_points = grad_points + momentum * decay
        momentum = grad_points

        ##### Add permutation
        permutation = eps_iter * np.sign(grad_points[:, :3])
        points_adv[:, :3] = points_adv[:, :3] + permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent_ori[:, :3]), min=-eps,
                            max=eps).numpy()
        points_adv[:, :3] = points_innocent_ori[:, :3] + delta

        cfg.logger.info('Iter {}: detection_loss={:.6f}'.format(n_step, sum(losses['loss'])))

        ##### point to voxel
        voxels, coordinates, num_points, voxel_innocent_ori, voxel_momentum = \
            cfg.voxelization.voxel_generator.generate_for_iterAttack_momentum(
                points_adv, points_innocent_ori=points_innocent_ori, momentum=momentum
            )
        save_to_example(example, voxels, coordinates, num_points, device)

    # save
    assert points_adv.shape == points_innocent_ori.shape
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_ori.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))
    # points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    # grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save origin points
    if "NuScenesDataset" in cfg.dataset_type:
        save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, example['points'][0].shape[1]),
                           points_innocent_ori, points_adv,
                           token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def MI_FGSM_Attack_light(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.MI_FGSM.eps
    eps_iter = cfg.MI_FGSM.eps_iter
    num_steps = cfg.MI_FGSM.num_steps
    decay = cfg.MI_FGSM.decay
    L_norm = cfg.MI_FGSM.L_norm
    dir_save_adv = cfg.MI_FGSM.save_dir

    voxel_innocent_ori = example['voxels'].clone()
    momentum = torch.zeros_like(voxel_innocent_ori)
    example['voxels'].requires_grad = True
    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])

    for n_step in range(num_steps):

        ###### Get gradient
        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        grad_voxel = torch.autograd.grad(sum(losses['loss']), example['voxels'],
                                         retain_graph=False, create_graph=False)[0]

        if 'L1' in L_norm:
            grad_voxel_norm = torch.mean(torch.abs(grad_voxel), dim=(2), keepdim=True)
        elif 'L2' in L_norm:
            grad_voxel_norm = torch.norm(grad_voxel, dim=2, keepdim=True)
            # grad_points_norm = torch.norm(torch.tensor(grad_points).view(grad_points.shape[0], -1), dim=1,
            #                               keepdim=True).numpy()

        grad_voxel = grad_voxel / (grad_voxel_norm + 1e-10)
        grad_voxel = grad_voxel + momentum * decay
        momentum = grad_voxel

        ##### Add permutation
        # mask voxel内空白的点、intensity、timestamp
        permutation = eps_iter * grad_voxel.sign() * perm_mask
        example['voxels'] = example['voxels'] + permutation
        delta = torch.clamp(example['voxels'] - voxel_innocent_ori, min=-eps, max=eps)
        example['voxels'] = voxel_innocent_ori + delta
        cfg.logger.info('Iter {}: detection_loss={:.6f}'.format(n_step, sum(losses['loss'])))

    # light version
    ## coordinate adjustment
    points_adv, points_innocent_ori = voxel_to_point_numba_for_iterAttack(
        example['voxels'].cpu().detach().numpy(),
        example['num_points'].cpu().detach().numpy(),
        voxel_innocent_ori.cpu().detach().numpy())
    # point to voxel
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(points_adv)
    save_to_example(example, voxels, coordinates, num_points, device)

    dist_points = points_adv - points_innocent_ori
    dist_points_l0 = np.abs(dist_points).sum(1)
    dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
    dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
    cfg.logger.info(
        'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    assert points_adv.shape == points_innocent_ori.shape
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_ori.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))

    # # save origin points
    # if "NuScenesDataset" in cfg.dataset_type:
    #     save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, example['points'][0].shape[1]),
    #                        points_innocent_ori, points_adv,
    #                        token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def IOU_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.IOU.eps
    eps_iter = cfg.IOU.eps_iter
    num_steps = cfg.IOU.num_steps
    Lambda = cfg.IOU.Lambda
    iou_thre = cfg.IOU.iou_thre
    score_thre = cfg.IOU.score_thre
    origin_voxels_v0 = example['voxels'].clone()
    dir_save_adv = cfg.IOU.save_dir
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for i in range(num_steps):
        # voxel to point
        if i == 0:
            points_innocent = points_innocent_v0.copy()
        else:
            points_innocent = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                                   example['num_points'].cpu().detach().numpy())

        example['voxels'].requires_grad = True

        # 计算loss
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pre_labels = predictions[0]['label_preds']

        rot_anno = torch.atan2(example['gt_boxes_and_cls'][0, :, -2], example['gt_boxes_and_cls'][0, :, -1])  # 500
        boxes_anno = torch.cat([example['gt_boxes_and_cls'][0, :, :6].view(-1, 6), rot_anno.view(-1, 1)],
                               dim=1)  # 500 * 7
        iou3d = boxes_iou3d_gpu(boxes_anno.to(device, non_blocking=False), pred_boxes)  # 500 * N

        loss_iou = 0
        for pre_index in range(iou3d.shape[1]):
            # loss_iou += -iou3d[:, pre_index].sum() * torch.log2(1 - pred_scores[pre_index])
            # if not pre_labels[pre_index] == cfg.target_adv_label:
            #     continue
            iou = iou3d[:, pre_index].sum()
            score = pred_scores[pre_index]
            if iou > iou_thre and score > score_thre:
                loss_iou += torch.log2(iou) * (- torch.log2(1 - score))  # log(IOU)

        # loss_l2 = torch.norm(example['voxels'] - origin_voxels, p=2)
        # total_loss = loss_iou + Lambda * loss_l2  # 坏loss，需要最小化
        loss_l2 = 0
        total_loss = loss_iou  # 坏loss，需要最小化

        # get gradient
        grad_voxel = \
            torch.autograd.grad(total_loss, example['voxels'], retain_graph=False, create_graph=False)[0]
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())

        permutation = -1.0 * eps_iter * np.sign(grad_points[:, :3])
        points_adv = points_innocent.copy()
        points_adv[:, :3] = points_innocent[:, :3] + permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent[:, :3]), min=-eps,
                            max=eps).numpy()
        points_adv[:, :3] = points_innocent[:, :3] + delta

        cfg.logger.info(
            'Iter {}: total_loss={:.6f}, loss_iou={:.6f}, loss_l2={:.6f}, Lambda={}'.format(i, total_loss, loss_iou,
                                                                                            loss_l2, Lambda))

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def   IOU_Attack_iter(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.IOU.eps
    eps_iter = cfg.IOU.eps_iter
    num_steps = cfg.IOU.num_steps
    Lambda = cfg.IOU.Lambda
    iou_thre = cfg.IOU.iou_thre
    score_thre = cfg.IOU.score_thre
    strategy = cfg.IOU.get('strategy', '')
    origin_voxels_v0 = example['voxels'].clone()
    dir_save_adv = cfg.IOU.save_dir
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):
        if not n_step == 0:
            # voxel to point
            points_adv, points_innocent_ori = voxel_to_point_numba_for_iterAttack(
                example['voxels'].cpu().detach().numpy(),
                example['num_points'].cpu().detach().numpy(),
                voxel_innocent_ori)
            # points_innocent_ori 跟随迭代更新，但是保存最初的信息
            # np.abs(points_adv - points_innocent_ori)[:, :3].mean()
            # (np.abs(points_adv - points_innocent_ori)[:, :3] > 0.10001).sum()
        else:
            points_adv = points_innocent_v0.copy()
            points_innocent_ori = points_innocent_v0.copy()

        example['voxels'].requires_grad = True

        # 计算loss
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pre_labels = predictions[0]['label_preds']

        rot_anno = torch.atan2(example['gt_boxes_and_cls'][0, :, -2], example['gt_boxes_and_cls'][0, :, -1])  # 500
        boxes_anno = torch.cat([example['gt_boxes_and_cls'][0, :, :6].view(-1, 6), rot_anno.view(-1, 1)],
                               dim=1)  # 500 * 7
        iou3d = boxes_iou3d_gpu(boxes_anno.to(device, non_blocking=False), pred_boxes)  # 500 * N

        loss_iou = torch.tensor(0.0, dtype=torch.float32)
        is_loss_non_zore = False
        for pre_index in range(iou3d.shape[1]):
            # loss_iou += -iou3d[:, pre_index].sum() * torch.log2(1 - pred_scores[pre_index])
            # if not pre_labels[pre_index] == cfg.target_adv_label:
            #     continue
            iou = iou3d[:, pre_index].sum()
            score = pred_scores[pre_index]
            if iou > iou_thre and score > score_thre:
                loss_iou = loss_iou + torch.log2(iou) * (- torch.log2(1 - score))  # log(IOU)
                is_loss_non_zore = True
        if 'no_lossl2' in strategy:
            loss_l2 = 0
            total_loss = loss_iou
        else:
            if n_step == 0:
                loss_l2 = torch.tensor(0.0, dtype=torch.float32)
                total_loss = loss_iou
            else:
                loss_l2 = torch.norm((example['voxels'] - torch.tensor(voxel_innocent_ori).to(device)), p=2)
                total_loss = loss_iou + Lambda * loss_l2  # 坏loss，需要最小化
        # total_loss = loss_iou  # 坏loss，需要最小化

        # get gradient
        if is_loss_non_zore:
            model.zero_grad()
            grad_voxel = \
                torch.autograd.grad(total_loss, example['voxels'], retain_graph=False, create_graph=False)[0]
            grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                         example['num_points'].cpu().detach().numpy())

            permutation = -1.0 * eps_iter * np.sign(grad_points[:, :3])
            points_adv[:, :3] = points_adv[:, :3] + permutation
            delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent_ori[:, :3]), min=-eps,
                                max=eps).numpy()
            points_adv[:, :3] = points_innocent_ori[:, :3] + delta
        else:
            grad_points = np.zeros_like(points_innocent_ori[:, :3])
            points_adv[:, :3] = points_innocent_ori[:, :3]

        cfg.logger.info(
            'Iter {}: total_loss={:.6f}, loss_iou={:.6f}, loss_l2={:.6f}, Lambda={}'.format(n_step, total_loss,
                                                                                            loss_iou,
                                                                                            loss_l2, Lambda))

        dist_points = points_adv - points_innocent_ori
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg.logger.info(
            '         l0={:02.1f}%, l2={:.1f})'.format(dist_l0 / dist_points.shape[0] * 100, dist_l2))

        # point to voxel
        voxels, coordinates, num_points, voxel_innocent_ori = \
            cfg.voxelization.voxel_generator.generate_for_iterAttack(
                points_adv, points_innocent_ori=points_innocent_ori
            )
        save_to_example(example, voxels, coordinates, num_points, device)

    # save
    assert points_adv.shape == points_innocent_ori.shape
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_ori.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))
    # points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    # grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, example['points'][0].shape[1]),
                       points_innocent_ori, points_adv,
                       token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def IOU_Attack_iter_light(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.IOU.eps
    eps_iter = cfg.IOU.eps_iter
    num_steps = cfg.IOU.num_steps
    Lambda = cfg.IOU.Lambda
    iou_thre = cfg.IOU.iou_thre
    score_thre = cfg.IOU.score_thre
    origin_voxels_v0 = example['voxels'].clone()
    dir_save_adv = cfg.IOU.save_dir
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):

        # 计算loss
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pre_labels = predictions[0]['label_preds']

        rot_anno = torch.atan2(example['gt_boxes_and_cls'][0, :, -2], example['gt_boxes_and_cls'][0, :, -1])  # 500
        boxes_anno = torch.cat([example['gt_boxes_and_cls'][0, :, :6].view(-1, 6), rot_anno.view(-1, 1)],
                               dim=1)  # 500 * 7
        iou3d = boxes_iou3d_gpu(boxes_anno.to(device, non_blocking=False), pred_boxes)  # 500 * N

        loss_iou = 0
        for pre_index in range(iou3d.shape[1]):
            # loss_iou += -iou3d[:, pre_index].sum() * torch.log2(1 - pred_scores[pre_index])
            # if not pre_labels[pre_index] == cfg.target_adv_label:
            #     continue
            iou = iou3d[:, pre_index].sum()
            score = pred_scores[pre_index]
            if iou > iou_thre and score > score_thre:
                loss_iou = loss_iou + torch.log2(iou) * (- torch.log2(1 - score))  # log(IOU)

        if n_step == 0:
            loss_l2 = 0
        else:
            loss_l2 = torch.norm((example['voxels'] - origin_voxels_v0), p=2)
        total_loss = loss_iou + Lambda * loss_l2  # 坏loss，需要最小化
        # total_loss = loss_iou  # 坏loss，需要最小化

        # get gradient
        example['voxels'].requires_grad = True
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(total_loss, example['voxels'], retain_graph=False, create_graph=False)[0]

        # mask voxel内空白的点、intensity、timestamp
        perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
        permutation = eps_iter * grad_voxel.sign() * perm_mask
        example['voxels'] = example['voxels'] + permutation
        delta = torch.clamp(example['voxels'] - origin_voxels_v0, min=-eps, max=eps)
        example['voxels'] = origin_voxels_v0 + delta

        cfg.logger.info(
            'Iter {}: total_loss={:.6f}, loss_iou={:.6f}, loss_l2={:.6f}, Lambda={}'.format(n_step, total_loss,
                                                                                            loss_iou,
                                                                                            loss_l2, Lambda))

    # voxel to point
    points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                      example['num_points'].cpu().detach().numpy())
    # point to voxel
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
        points_adv
    )
    save_to_example(example, voxels, coordinates, num_points, device)

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def IOU_CatAdv_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.IOU_CatAdv.eps
    eps_iter = cfg.IOU_CatAdv.eps_iter
    num_steps = cfg.IOU_CatAdv.num_steps
    Lambda = cfg.IOU_CatAdv.Lambda
    iou_thre = cfg.IOU_CatAdv.iou_thre
    score_thre = cfg.IOU_CatAdv.score_thre
    dir_save_adv = cfg.IOU_CatAdv.save_dir
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for i in range(num_steps):
        # voxel to point
        if i == 0:
            points_innocent = points_innocent_v0.copy()
        else:
            points_innocent = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                                   example['num_points'].cpu().detach().numpy())

        example['voxels'].requires_grad = True
        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)

        # 计算loss
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pre_labels = predictions[0]['label_preds']

        rot_anno = torch.atan2(example['gt_boxes_and_cls'][0, :, -2], example['gt_boxes_and_cls'][0, :, -1])  # 500
        boxes_anno = torch.cat([example['gt_boxes_and_cls'][0, :, :6].view(-1, 6), rot_anno.view(-1, 1)],
                               dim=1)  # 500 * 7
        iou3d = boxes_iou3d_gpu(boxes_anno.to(device, non_blocking=False), pred_boxes)  # 500 * N

        loss_iou = 0
        for pre_index in range(iou3d.shape[1]):
            # loss_iou += -iou3d[:, pre_index].sum() * torch.log2(1 - pred_scores[pre_index])
            if not pre_labels[pre_index] in cfg.target_adv_label_list:
                continue
            iou = iou3d[:, pre_index].sum()
            score = pred_scores[pre_index]
            if iou > iou_thre and score > score_thre:
                loss_iou += torch.log2(iou) * (- torch.log2(1 - score))  # log(IOU)

        # loss_l2 = torch.norm(example['voxels'] - origin_voxels, p=2)
        # total_loss = loss_iou + Lambda * loss_l2  # 坏loss，需要最小化
        loss_l2 = 0
        total_loss = loss_iou - sum(losses['loss'])  # 坏loss，需要最小化

        # get gradient
        grad_voxel = \
            torch.autograd.grad(total_loss, example['voxels'], retain_graph=False, create_graph=False)[0]
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())

        permutation = -1.0 * eps_iter * np.sign(grad_points[:, :3])
        points_adv = points_innocent.copy()
        points_adv[:, :3] = points_innocent[:, :3] + permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent[:, :3]), min=-eps,
                            max=eps).numpy()
        points_adv[:, :3] = points_innocent[:, :3] + delta

        cfg.logger.info(
            'Iter {}: total_loss={:.6f}, loss_iou={:.6f}, loss_l2={:.6f}, Lambda={}'.format(i, total_loss, loss_iou,
                                                                                            loss_l2, Lambda))

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions

def DistScore_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps_iter = cfg.DistScore.eps_iter
    num_steps = cfg.DistScore.num_steps
    Lambda = cfg.DistScore.Lambda

    origin_voxels = example['voxels'].clone()
    example['voxels'].requires_grad = True

    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
    for i in range(num_steps):

        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
        boxes_pred = torch.tensor(predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]).to(device)
        scores_pred = predictions[0]['scores']
        pre_labels = predictions[0]['label_preds']

        # rot_anno = torch.atan2(example['gt_boxes_and_cls'][0, :, -2], example['gt_boxes_and_cls'][0, :, -1])
        # boxes_anno = torch.cat([example['gt_boxes_and_cls'][0, :, :6].view(-1, 6), rot_anno.view(-1, 1)], dim=1)
        boxes_anno = torch.tensor(example['gt_boxes_and_cls'][0, :, :6].view(-1, 6)).to(device)

        loss_dist = 0
        dist_gt_pred = torch.zeros(size=[boxes_anno.shape[0], boxes_pred.shape[0]])
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        for pre_index in range(boxes_pred.shape[0]):

            if not pre_labels[pre_index] == cfg.target_adv_label:
                continue

            # location1 = boxes_pred[pre_index, [0, 1]]
            # for j in range(boxes_anno.shape[0]):
            #     if boxes_anno[j, 3:6].sum() < 0.000001:
            #         continue
            #     location2 = boxes_anno[j, [0, 1]]
            #     dist_gt_pred[j, pre_index] = pdist(location1, location2)
            # dist_sum = dist_gt_pred[:, pre_index].sum()
            # if dist_sum > 0:
            #     loss_dist += (- torch.log2(dist_sum)) * (- torch.log2(1 - scores_pred[pre_index]))

            location3 = boxes_pred[pre_index, [0, 1]].repeat(boxes_anno.shape[0], 1)
            location4 = boxes_anno[:, [0, 1]]
            dist_gt_pred[:, pre_index] = pdist(location3, location4).squeeze()
            # dist_sum = dist_gt_pred[:, pre_index].sum()

            # 只攻击距离小于0.5的proposal
            dist_mask = dist_gt_pred[:, pre_index] <= 2
            dist_filtered_sum = (dist_gt_pred[:, pre_index] * dist_mask).sum()

            if dist_filtered_sum > 0:
                loss_dist += (- torch.log2(dist_filtered_sum)) * (- torch.log2(1 - scores_pred[pre_index]))

        # loss_dist = -1.0 * dist_gt_pred.sum()
        loss_l2 = torch.norm(example['voxels'] - origin_voxels, p=2)
        total_loss = loss_dist + Lambda * loss_l2  # 最小化
        grad_voxel = \
            torch.autograd.grad(total_loss, example['voxels'], retain_graph=False, create_graph=False)[0]

        permutation = -1.0 * eps_iter * grad_voxel.sign() * perm_mask
        example['voxels'] = example['voxels'] + permutation
        cfg.logger.info(
            'Iter {}: total_loss={:.6f}, loss_dist={:.6f}, loss_l2={:.6f}, Lambda={}'.format(i, total_loss, loss_dist,
                                                                                             loss_l2, Lambda))

    # save
    token = example['metadata'][0]['token']
    save_dir = os.path.join(cfg.DistScore.save_dir, 'iter_eps_{}-num_steps_{}-Lambda_{}'.format(eps_iter, num_steps, Lambda))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, str(token) + '.npy')
    save_sample(origin_voxels, example, save_path)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def FalsePositive_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps_iter = cfg.FalsePositive.eps_iter
    num_steps = cfg.FalsePositive.num_steps
    Lambda = cfg.FalsePositive.Lambda

    origin_voxels = example['voxels'].clone()
    example['voxels'].requires_grad = True

    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
    for i in range(num_steps):

        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pre_labels = predictions[0]['label_preds']

        rot_anno = torch.atan2(example['gt_boxes_and_cls'][0, :, -2], example['gt_boxes_and_cls'][0, :, -1])  # 500
        boxes_anno = torch.cat([example['gt_boxes_and_cls'][0, :, :6].view(-1, 6), rot_anno.view(-1, 1)],
                               dim=1)  # 500 * 7
        iou3d = boxes_iou3d_gpu(boxes_anno.to(device, non_blocking=False), pred_boxes)  # 500 * N

        loss_fp = 0
        for pre_index in range(iou3d.shape[1]):
            # loss_iou += -iou3d[:, pre_index].sum() * torch.log2(1 - pred_scores[pre_index])
            # if not pre_labels[pre_index] == cfg.target_adv_label:
            #     continue
            iou = iou3d[:, pre_index].sum()
            if iou <= 1e-8:
                loss_fp += torch.log2(1 - pred_scores[pre_index])

        loss_l2 = torch.norm(example['voxels'] - origin_voxels, p=2)
        total_loss = loss_fp + Lambda * loss_l2  # 坏loss，需要最小化
        grad_voxel = \
            torch.autograd.grad(total_loss, example['voxels'], retain_graph=False, create_graph=False)[0]

        permutation = -1.0 * eps_iter * grad_voxel.sign() * perm_mask
        example['voxels'] = example['voxels'] + permutation
        cfg.logger.info(
            'Iter {}: total_loss={:.6f}, loss_iou={:.6f}, loss_l2={:.6f}, Lambda={}'.format(i, total_loss, loss_fp,
                                                                                            loss_l2, Lambda))

        ## coordinate adjustment
        points = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                      example['num_points'].cpu().detach().numpy())
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points
        )
        example['voxels'] = torch.from_numpy(voxels).to(device, non_blocking=False)
        Coors = collate_kitti_v0(coordinates)
        example['coordinates'] = Coors.to(device, non_blocking=False)
        example['num_points'] = torch.from_numpy(num_points).to(device, non_blocking=False)
        example['num_voxels'] = np.array([example['voxels'].shape[0]], dtype=np.int64)

    # save
    token = example['metadata'][0]['token']
    save_dir = os.path.join(cfg.FalsePositive.save_dir,
                            'iter_eps_{}-num_steps_{}-Lambda_{}'.format(eps_iter, num_steps, Lambda))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, str(token) + '.npy')
    save_sample(origin_voxels, example, save_path)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def CA_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps_iter = cfg.CA.eps_iter
    num_steps = cfg.CA.num_steps
    Lambda = cfg.CA.Lambda

    origin_voxels = example['voxels'].clone()
    example['voxels'].requires_grad = True

    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
    for i in range(num_steps):

        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)

        boxes_pred = torch.tensor(predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]).to(device)
        scores_pred = predictions[0]['scores']
        pre_labels = predictions[0]['label_preds']

        # rot_anno = torch.atan2(example['gt_boxes_and_cls'][0, :, -2], example['gt_boxes_and_cls'][0, :, -1])
        # boxes_anno = torch.cat([example['gt_boxes_and_cls'][0, :, :6].view(-1, 6), rot_anno.view(-1, 1)], dim=1)
        boxes_anno = torch.tensor(example['gt_boxes_and_cls'][0, :, :6].view(-1, 6)).to(device)

        loss_dist = 0
        dist_gt_pred = torch.zeros(size=[boxes_anno.shape[0], boxes_pred.shape[0]])
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        for pre_index in range(boxes_pred.shape[0]):

            if not pre_labels[pre_index] == cfg.target_adv_label:
                continue

            # location1 = boxes_pred[pre_index, [0, 1]]
            # for j in range(boxes_anno.shape[0]):
            #     if boxes_anno[j, 3:6].sum() < 0.000001:
            #         continue
            #     location2 = boxes_anno[j, [0, 1]]
            #     dist_gt_pred[j, pre_index] = pdist(location1, location2)
            # dist_sum = dist_gt_pred[:, pre_index].sum()
            # if dist_sum > 0:
            #     loss_dist += (- torch.log2(dist_sum)) * (- torch.log2(1 - scores_pred[pre_index]))

            location3 = boxes_pred[pre_index, [0, 1]].repeat(boxes_anno.shape[0], 1)
            location4 = boxes_anno[:, [0, 1]]
            dist_gt_pred[:, pre_index] = pdist(location3, location4).squeeze()
            # dist_sum = dist_gt_pred[:, pre_index].sum()

            # 只攻击距离小于0.5的proposal
            dist_mask = dist_gt_pred[:, pre_index] <= 2
            dist_filtered_sum = (dist_gt_pred[:, pre_index] * dist_mask).sum()

            if dist_filtered_sum > 0:
                loss_dist += (- torch.log2(dist_filtered_sum)) * (- torch.log2(1 - scores_pred[pre_index]))

        # loss_dist = -1.0 * dist_gt_pred.sum()
        loss_l2 = torch.norm(example['voxels'] - origin_voxels, p=2)
        total_loss = loss_dist + Lambda * loss_l2  # 最小化
        grad_voxel = \
            torch.autograd.grad(total_loss, example['voxels'], retain_graph=False, create_graph=False)[0]

        permutation = -1.0 * eps_iter * grad_voxel.sign() * perm_mask
        example['voxels'] = example['voxels'] + permutation
        cfg.logger.info(
            'Iter {}: total_loss={:.6f}, loss_dist={:.6f}, loss_l2={:.6f}, Lambda={}'.format(i, total_loss, loss_dist,
                                                                                             loss_l2, Lambda))

    # save
    token = example['metadata'][0]['token']
    save_dir = os.path.join(cfg.CA.save_dir, 'iter_eps_{}-num_steps_{}-Lambda_{}'.format(eps_iter, num_steps, Lambda))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, str(token) + '.npy')
    save_sample(origin_voxels, example, save_path)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def generate_random_numbers(n, size):
    return np.random.choice(n, size=size, replace=False)


def PointAttach_Attack_random(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.PointAttach.eps
    eps_iter = cfg.PointAttach.eps_iter
    num_steps = cfg.PointAttach.num_steps
    Lambda = cfg.PointAttach.Lamda
    strategy = cfg.PointAttach.strategy
    attach_rate = cfg.PointAttach.attach_rate
    dir_save_adv = cfg.PointAttach.save_dir
    gt_boxes_and_cls = example['gt_boxes_and_cls']

    # attack
    origin_voxels = example['voxels'].clone()
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            # voxel to point
            points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
        else:
            points_adv = points_innocent_v0.copy()

        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape

        # is_object_point = object_point_filter(points_innocent, gt_boxes_and_cls.numpy()[0])
        # visualize_points(points_innocent[:, :3] * np.repeat(is_object_point.reshape(-1, 1), 3, axis=-1))
        # is_object_point_numba = object_point_filter_numba(points_innocent, gt_boxes_and_cls.numpy()[0])

        # grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        # for j in range(grad_points.shape[0]):
        #     grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        # grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1, descending=True).numpy()  # 梯度排序
        # assert np.max(grad_points_order) < points_adv.shape[0]
        attach_num = int(points_adv.shape[0] * attach_rate)
        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)
        rand_indexs = generate_random_numbers(points_adv.shape[0], attach_num)
        cnt = 0
        for j in range(points_adv.shape[0]):
            # if is_object_point[grad_points_order[i]] is not True:
            #     continue
            if cnt >= int(attach_num / num_steps):
                break
            new_point = points_adv[rand_indexs[j], :]
            grad_for_new_point = grad_points[rand_indexs[j], :]
            new_point[:3] = new_point[:3] + eps_iter * np.sign(grad_for_new_point[:3])
            points_adv[rand_indexs[j], :] = new_point
            cnt += 1

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = points_adv[:, :3] - points_innocent_v0[:, :3]
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg.logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))


def PointAttach_Attack_gradient_sorted(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.PointAttach.eps
    eps_iter = cfg.PointAttach.eps_iter
    num_steps = cfg.PointAttach.num_steps
    Lambda = cfg.PointAttach.Lamda
    strategy = cfg.PointAttach.strategy
    attach_rate = cfg.PointAttach.attach_rate
    dir_save_adv = cfg.PointAttach.save_dir
    os.makedirs(dir_save_adv, exist_ok=True)
    gt_boxes_and_cls = example['gt_boxes_and_cls']

    # attack
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        # voxel to point
        if n_step == 0:
            points_innocent = points_innocent_v0.copy()
        else:
            points_innocent = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                                   example['num_points'].cpu().detach().numpy())

        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_innocent.shape

        # is_object_point = object_point_filter(points_innocent, gt_boxes_and_cls.numpy()[0])
        # visualize_points(points_innocent[:, :3] * np.repeat(is_object_point.reshape(-1, 1), 3, axis=-1))
        # is_object_point_numba = object_point_filter_numba(points_innocent, gt_boxes_and_cls.numpy()[0])

        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1, descending=True).numpy()  # 梯度排序
        assert np.max(grad_points_order) < points_innocent.shape[0]
        attach_num = int(points_innocent.shape[0] * attach_rate)
        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)

        new_points_list = []
        cnt = 0
        for j in range(grad_points_order.shape[0]):
            # if is_object_point[grad_points_order[i]] is not True:
            #     continue
            if cnt >= (attach_num / num_steps):
                break
            new_point = points_innocent[grad_points_order[j], :].copy()
            grad_for_new_point = grad_points[grad_points_order[j], :]
            new_point[:3] = new_point[:3] + eps_iter * np.sign(grad_for_new_point[:3])
            new_points_list.append(new_point)
            cnt += 1

        points_adv = np.concatenate([np.array(new_points_list), points_innocent], axis=0)

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))


def PointAttach_Attack_gradient_sorted_not_add(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.PointAttach.eps
    eps_iter = cfg.PointAttach.eps_iter
    num_steps = cfg.PointAttach.num_steps
    Lambda = cfg.PointAttach.Lamda
    strategy = cfg.PointAttach.strategy
    attach_rate = cfg.PointAttach.attach_rate
    dir_save_adv = cfg.PointAttach.save_dir
    gt_boxes_and_cls = example['gt_boxes_and_cls']

    # attack
    origin_voxels = example['voxels'].clone()
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            # voxel to point
            points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
        else:
            points_adv = points_innocent_v0.copy()

        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape

        # is_object_point = object_point_filter(points_innocent, gt_boxes_and_cls.numpy()[0])
        # visualize_points(points_innocent[:, :3] * np.repeat(is_object_point.reshape(-1, 1), 3, axis=-1))
        # is_object_point_numba = object_point_filter_numba(points_innocent, gt_boxes_and_cls.numpy()[0])

        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        if 'small_grad' not in strategy:
            grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
                                              descending=True).numpy()  # 梯度排序
        else:
            grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
                                              descending=False).numpy()  # 梯度排序
        assert np.max(grad_points_order) < points_adv.shape[0]
        attach_num = int(points_adv.shape[0] * attach_rate)
        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)

        cnt = 0
        for j in range(grad_points_order.shape[0]):
            # if is_object_point[grad_points_order[i]] is not True:
            #     continue
            if cnt >= (attach_num / num_steps):
                break
            new_point = points_adv[grad_points_order[j], :]
            grad_for_new_point = grad_points[grad_points_order[j], :]
            new_point[:3] = new_point[:3] + eps_iter * np.sign(grad_for_new_point[:3])
            points_adv[grad_points_order[j], :] = new_point
            cnt += 1

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = points_adv[:, :3] - points_innocent_v0[:, :3]
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg.logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))


def compute_density(pcd, radius=0.15, knn=3, is_norm=True):
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
        if is_norm:
            l2_dist = np.linalg.norm(np.asarray(pcd.points)[i, :3])
            radius_norm = radius * l2_dist
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius_norm)
            density[i] = k / (4 / 3 * np.pi * radius ** 3)
        else:
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], radius)
            if k < knn:
                density[i] = 0
            else:
                density[i] = k / (4 / 3 * np.pi * radius ** 3)
    return density


def leave_one_out(voxels, marked_index, num_points, permutation_leave_one):
    for i in range(num_points.shape[0]):
        points_mean = voxels[i, :num_points[i], :3].mean(axis=0)
        Index = marked_index[i, 0]  # 可以增加选点策略
        permutation_leave_one[Index] = points_mean


def PointAttach_Attack_gradient_sorted_not_add_voxelize_permutation(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.PointAttach.eps
    eps_iter = cfg.PointAttach.eps_iter
    num_steps = cfg.PointAttach.num_steps
    Lambda = cfg.PointAttach.Lamda
    strategy = cfg.PointAttach.strategy
    attach_rate = cfg.PointAttach.attach_rate
    dir_save_adv = cfg.PointAttach.save_dir

    # attack
    origin_voxels = example['voxels'].clone()
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            # voxel to point
            points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
        else:
            points_adv = points_innocent_v0.copy()

        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape

        # is_object_point = object_point_filter(points_innocent, gt_boxes_and_cls.numpy()[0])
        # visualize_points(points_innocent[:, :3] * np.repeat(is_object_point.reshape(-1, 1), 3, axis=-1))
        # is_object_point_numba = object_point_filter_numba(points_innocent, gt_boxes_and_cls.numpy()[0])

        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1, descending=True).numpy()  # 梯度排序
        assert np.max(grad_points_order) < points_adv.shape[0]
        attach_num = int(points_adv.shape[0] * attach_rate)
        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)

        cnt = 0
        permutation = np.ones((points_adv.shape[0], 3), dtype=np.float32) * 999
        for j in range(grad_points_order.shape[0]):
            # if is_object_point[grad_points_order[i]] is not True:
            #     continue
            if cnt >= (attach_num / num_steps):
                break
            grad_for_new_point = grad_points[grad_points_order[j], :]
            adv_point = points_adv[grad_points_order[j], :3]
            # perm_point = permutation[grad_points_order[j], :]
            permutation[grad_points_order[j], :3] = adv_point + eps_iter * np.sign(grad_for_new_point[:3])
            cnt += 1

        # permutation to voxel
        voxels_permutation, marked_index_permutation, coordinates_permutation, num_points_permutation = cfg.voxelization.voxel_generator.generate_mark_index(
            permutation,
        )
        permutation_leave_one = np.zeros_like(permutation, dtype=np.float32)
        leave_one_out(voxels_permutation, marked_index_permutation, num_points_permutation, permutation_leave_one)
        for j in range(points_adv.shape[0]):
            if np.abs(permutation_leave_one[j, :3]).sum() > 0.00001:
                points_adv[j, :3] = permutation_leave_one[j, :3]  # permutation_leave_one 更新到对抗样本上

        n_modi_perm = int(np.linalg.norm(permutation.reshape(-1) - 999, ord=0) / 3)
        assert n_modi_perm == attach_num
        n_modi_perm_final = int(np.linalg.norm((points_adv - points_innocent_v0).reshape(-1), ord=0) / 3)
        n_modi_perm_leave_one = int(np.linalg.norm(permutation_leave_one.reshape(-1), ord=0) / 3)
        assert n_modi_perm_final == n_modi_perm_leave_one
        # assert np.all(np.abs(np.abs(points_adv - points_innocent_v0) - eps_iter) < 0.000001) # 最大扰动幅度暂时不能控制，可以通过leave one out 时候的选点策略
        # temp = np.abs(points_adv - points_innocent_v0)
        # for k in range(temp.shape[0]):
        #     for l in range(temp.shape[1]):
        #         if np.abs(temp[k, l]) > eps_iter + 0.05:
        #             print("k， l : {}，{}, val: {}".format(k, l, temp[k, l]))
        cfg.logger.info(
            'Iter {}: l0修改量： leave on out 前={}， 后={}(修改量 {:02.1f}%)'.format(n_step, n_modi_perm,
                                                                                    n_modi_perm_leave_one,
                                                                                    n_modi_perm_leave_one /
                                                                                    points_adv.shape[0] * 100))

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv,
        )
        save_to_example(example, voxels, coordinates, num_points, device)

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))

    # permutation_leave_one_pure = np.zeros((n_modi_perm_leave_one, 5), dtype=np.float32)
    # for j in range(permutation_leave_one.shape[0]):
    #     if np.abs(permutation_leave_one[j, :3]).sum() > 0.00001:
    #         permutation_leave_one_pure[j, :3] = permutation_leave_one[j, :3]  # permutation_leave_one 过滤（0， 0， 0）
    # permutation_leave_one_pure.tofile(dir_save_adv, token + '-permutation_leave_one_pure.bin')


def PointAttach_Attack_gradient_sorted_not_add_voxelize_permutation_limitH(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.PointAttach.eps
    eps_iter = cfg.PointAttach.eps_iter
    num_steps = cfg.PointAttach.num_steps
    Lambda = cfg.PointAttach.Lamda
    strategy = cfg.PointAttach.strategy
    attach_rate = cfg.PointAttach.attach_rate
    dir_save_adv = cfg.PointAttach.save_dir
    os.makedirs(dir_save_adv, exist_ok=True)
    gt_boxes_and_cls = example['gt_boxes_and_cls']

    # attack
    origin_voxels = example['voxels'].clone()
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            # voxel to point
            points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
        else:
            points_adv = points_innocent_v0.copy()

        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape

        # is_object_point = object_point_filter(points_innocent, gt_boxes_and_cls.numpy()[0])
        # visualize_points(points_innocent[:, :3] * np.repeat(is_object_point.reshape(-1, 1), 3, axis=-1))
        # is_object_point_numba = object_point_filter_numba(points_innocent, gt_boxes_and_cls.numpy()[0])

        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1, descending=True).numpy()  # 梯度排序
        assert np.max(grad_points_order) < points_adv.shape[0]
        attach_num = int(points_adv.shape[0] * attach_rate)
        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)

        cnt = 0
        permutation = np.ones((points_adv.shape[0], 3), dtype=np.float32) * 999
        h_low = -2.0
        h_up = -1.5
        for j in range(grad_points_order.shape[0]):
            # if is_object_point[grad_points_order[i]] is not True:
            #     continue
            if cnt >= (attach_num / num_steps):
                break
            cnt += 1
            if (h_low <= points_adv[grad_points_order[j], 2] and points_adv[
                grad_points_order[j], 2] <= h_up):  # 如果该点是地面
                continue
            grad_for_new_point = grad_points[grad_points_order[j], :]
            adv_point = points_adv[grad_points_order[j], :3]
            # perm_point = permutation[grad_points_order[j], :]
            permutation[grad_points_order[j], :3] = adv_point + eps_iter * np.sign(grad_for_new_point[:3])

        # permutation to voxel
        voxels_permutation, marked_index_permutation, coordinates_permutation, num_points_permutation = cfg.voxelization.voxel_generator.generate_mark_index(
            permutation,
        )
        permutation_leave_one = np.zeros_like(permutation, dtype=np.float32)
        leave_one_out(voxels_permutation, marked_index_permutation, num_points_permutation, permutation_leave_one)
        for j in range(points_adv.shape[0]):
            if np.abs(permutation_leave_one[j, :3]).sum() > 0.00001:
                points_adv[j, :3] = permutation_leave_one[j, :3]  # permutation_leave_one 更新到对抗样本上

        n_modi_perm = int(np.linalg.norm(permutation.reshape(-1) - 999, ord=0) / 3)
        # assert n_modi_perm == attach_num
        n_modi_perm_final = int(np.linalg.norm((points_adv - points_innocent_v0).reshape(-1), ord=0) / 3)
        n_modi_perm_leave_one = int(np.linalg.norm(permutation_leave_one.reshape(-1), ord=0) / 3)
        assert n_modi_perm_final == n_modi_perm_leave_one
        # assert np.all(np.abs(np.abs(points_adv - points_innocent_v0) - eps_iter) < 0.000001) # 最大扰动幅度暂时不能控制，可以通过leave one out 时候的选点策略
        # temp = np.abs(points_adv - points_innocent_v0)
        # for k in range(temp.shape[0]):
        #     for l in range(temp.shape[1]):
        #         if np.abs(temp[k, l]) > eps_iter + 0.05:
        #             print("k， l : {}，{}, val: {}".format(k, l, temp[k, l]))
        cfg.logger.info(
            'Iter {}: l0修改量： leave on out 前={}， 后={}(修改量 {:02.1f}%)'.format(n_step, n_modi_perm,
                                                                                    n_modi_perm_leave_one,
                                                                                    n_modi_perm_leave_one /
                                                                                    points_adv.shape[0] * 100))

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))


@numba.jit(nopython=True)
def initialize_voxel_center_points(grid_size, coors_range, voxel_size, voxel_center_bias, random_rate):
    rand_Matrix = np.random.rand(int(grid_size[0] * grid_size[1] * grid_size[2]))
    # mask_rand_Matrix = rand_Matrix[rand_Matrix < random_rate]

    voxel_center_points_array = np.zeros((int(grid_size[0] * grid_size[1] * grid_size[2]), 3), dtype=np.float32)
    cnt = 0
    for i in range(int(grid_size[0])):
        for j in range(int(grid_size[1])):
            for k in range(int(grid_size[2])):
                if rand_Matrix[i * j + k] >= random_rate:
                    continue
                voxel_center_points = coors_range[:3] + np.array([i * voxel_size[0], j * voxel_size[1],
                                                                  k * voxel_size[2]]) + voxel_center_bias
                # voxel_center_points_array[i*j+k, :] = voxel_center_points
                voxel_center_points_array[cnt, :] = voxel_center_points
                cnt += 1
    return voxel_center_points_array[:cnt, :]


@numba.jit(nopython=True)
def remove_points(add_num, grad_points_order, points_adv, coors_range, voxel_size, voxel_center_bias):
    remained_points = np.zeros(shape=(add_num, 5), dtype=np.float32)
    cnt = 0
    for i in range(grad_points_order.shape[0]):
        ind = grad_points_order[i]
        if np.abs(points_adv[ind, :3] - (
                ((points_adv[ind, :3] - coors_range[:3]) // voxel_size) * voxel_size + coors_range[
                                                                                       :3] + voxel_center_bias)).sum() < 0.000001:
            remained_points[i, :] = points_adv[ind, :]
            cnt += 1
        if cnt >= add_num:
            break
    return remained_points

def PointAttach_Attack_gradient_sorted_removePoints(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.PointAttach.eps
    eps_iter = cfg.PointAttach.eps_iter
    num_steps = cfg.PointAttach.num_steps
    Lambda = cfg.PointAttach.Lamda
    strategy = cfg.PointAttach.strategy
    attach_rate = cfg.PointAttach.attach_rate
    dir_save_adv = cfg.PointAttach.save_dir
    # attack
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    coors_range = np.array(voxelization_cfg.cfg.range)
    voxel_size = np.array(voxelization_cfg.cfg.voxel_size)
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size  # [1440, 1440, 40]
    voxel_center_bias = voxel_size / 2
    random_rate = np.float32(strategy.split('init_')[-1])  # 0.00005
    voxel_center_points_array = initialize_voxel_center_points(grid_size, coors_range, voxel_size, voxel_center_bias,
                                                               random_rate)
    voxel_center_points_array = np.hstack(
        (voxel_center_points_array, np.zeros((voxel_center_points_array.shape[0], 2)))).astype(np.float32)
    # 填充intensity, timestamp
    randint_array = np.random.randint(low=0, high=points_innocent_v0.shape[0], size=voxel_center_points_array.shape[0])
    intensity_timestamp = points_innocent_v0[randint_array, 3:]
    voxel_center_points_array[:, 3:] = intensity_timestamp
    # for i in range(voxel_center_points_array.shape[0]):
    #     randint = np.random.randint(low=0, high=points_innocent_v0.shape[0])
    #     intensity, timestamp = points_innocent_v0[randint, 3:]
    #     voxel_center_points_array[i, 3:] = [intensity, timestamp]

    points_combined = np.vstack([points_innocent_v0, voxel_center_points_array])
    # visualize_points(points_combined)
    # point to voxel
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
        points_combined
    )
    save_to_example(example, voxels, coordinates, num_points, device)
    add_num = voxel_center_points_array.shape[0]
    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        # voxel to point
        points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                          example['num_points'].cpu().detach().numpy())
        # visualize_points(points_adv)
        example['voxels'] = example['voxels'].to(dtype=torch.float32)
        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape
        # is_object_point = object_point_filter(points_innocent, gt_boxes_and_cls.numpy()[0])
        # visualize_points(points_innocent[:, :3] * np.repeat(is_object_point.reshape(-1, 1), 3, axis=-1))
        # is_object_point_numba = object_point_filter_numba(points_innocent, gt_boxes_and_cls.numpy()[0])
        ## 梯度排序
        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for i in range(grad_points.shape[0]):
            grad_points_abs_sum[i] = np.abs(grad_points[i, :3]).sum()
        grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1, descending=True).numpy()  # 梯度排序
        assert np.max(grad_points_order) < points_adv.shape[0]
        attach_num = int(points_adv.shape[0] * attach_rate)
        add_num = add_num // 2
        ## 留下add_num个点
        remained_points = remove_points(add_num, grad_points_order, points_adv, coors_range, voxel_size,
                                        voxel_center_bias)

        points_combined = np.vstack([points_innocent_v0, np.array(remained_points).reshape((-1, 5))])
        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_combined
        )
        save_to_example(example, voxels, coordinates, num_points, device)

        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)
        # visualize_points(points_combined)

    # save
    points_adv = points_combined
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    if num_steps > 0:
        grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))


def PointAttach_Attack_gradient_sorted_iouLoss(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.PointAttach.eps
    eps_iter = cfg.PointAttach.eps_iter
    num_steps = cfg.PointAttach.num_steps
    Lambda = cfg.PointAttach.Lamda
    strategy = cfg.PointAttach.strategy
    attach_rate = cfg.PointAttach.attach_rate
    dir_save_adv = os.path.join(cfg.PointAttach.save_dir,
                                'eps_{}-iter_eps_{}-num_steps_{}-Lambda_{}-attach_rate_{}-strategy_{}'.format(Epsilon,
                                                                                                              eps_iter,
                                                                                                              num_steps,
                                                                                                              Lambda,
                                                                                                              attach_rate,
                                                                                                              strategy))
    os.makedirs(dir_save_adv, exist_ok=True)
    gt_boxes_and_cls = example['gt_boxes_and_cls']

    # attack
    origin_voxels = example['voxels'].clone()
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    iou_thre = 0.1
    score_thre = 0.1
    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            # voxel to point
            points_innocent = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                                   example['num_points'].cpu().detach().numpy())
        else:
            points_innocent = points_innocent_v0.copy()
        origin_voxels = example['voxels'].clone()

        losses, predictions = model(example, return_loss=True, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pre_labels = predictions[0]['label_preds']

        rot_anno = torch.atan2(example['gt_boxes_and_cls'][0, :, -2], example['gt_boxes_and_cls'][0, :, -1])  # 500
        boxes_anno = torch.cat([example['gt_boxes_and_cls'][0, :, :6].view(-1, 6), rot_anno.view(-1, 1)],
                               dim=1)  # 500 * 7
        iou3d = boxes_iou3d_gpu(boxes_anno.to(device, non_blocking=False), pred_boxes)  # 500 * N

        loss_iou = 0
        for pre_index in range(iou3d.shape[1]):
            # loss_iou += -iou3d[:, pre_index].sum() * torch.log2(1 - pred_scores[pre_index])
            if not pre_labels[pre_index] == cfg.target_adv_label:
                continue
            iou = iou3d[:, pre_index].sum()
            score = pred_scores[pre_index]
            if iou > iou_thre and score > score_thre:
                loss_iou += torch.log2(iou) * (- torch.log2(1 - score))  # log(IOU)

        loss_l2 = torch.norm(example['voxels'] - origin_voxels, p=2)
        total_loss = loss_iou + Lambda * loss_l2 - sum(losses['loss'])  # 坏loss，需要最小化
        grad_voxel = \
            torch.autograd.grad(total_loss, example['voxels'], retain_graph=False, create_graph=False)[0]

        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_innocent.shape

        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for i in range(grad_points.shape[0]):
            grad_points_abs_sum[i] = np.abs(grad_points[i, :3]).sum()
        grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1, descending=True).numpy()  # 梯度排序
        assert np.max(grad_points_order) < points_innocent.shape[0]
        attach_num = int(points_innocent.shape[0] * attach_rate)
        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)

        # 加点
        new_points_list = []
        cnt = 0
        for i in range(grad_points_order.shape[0]):
            # if is_object_point[grad_points_order[i]] is not True:
            #     continue
            if cnt >= (attach_num / num_steps):
                break
            new_point = points_innocent[grad_points_order[i], :].copy()
            grad_for_new_point = grad_points[grad_points_order[i], :]
            new_point[:3] = new_point[:3] + eps_iter * np.sign(grad_for_new_point[:3])
            new_points_list.append(new_point)
            cnt += 1

        points_adv = np.concatenate([np.array(new_points_list), points_innocent], axis=0)

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

        cfg.logger.info(
            'Iter {}: total_loss={:.6f}, loss_iou={:.6f}, loss_l2={:.6f}, Lambda={}'.format(i, total_loss, loss_iou,
                                                                                            loss_l2, Lambda))
    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))


def PointAttach_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.PointAttach.eps
    eps_iter = cfg.PointAttach.eps_iter
    num_steps = cfg.PointAttach.num_steps
    Lambda = cfg.PointAttach.Lamda
    strategy = cfg.PointAttach.strategy
    attach_rate = cfg.PointAttach.attach_rate
    dir_save_adv = os.path.join(cfg.PointAttach.save_dir,
                                'eps_{}-iter_eps_{}-num_steps_{}-Lambda_{}-attach_rate_{}-strategy_{}'.format(Epsilon,
                                                                                                              eps_iter,
                                                                                                              num_steps,
                                                                                                              Lambda,
                                                                                                              attach_rate,
                                                                                                              strategy))
    os.makedirs(dir_save_adv, exist_ok=True)
    gt_boxes_and_cls = example['gt_boxes_and_cls']
    if strategy == 'random':
        PointAttach_Attack_random(model, example, device, **kwargs)
    elif strategy == 'gradient_sorted':
        PointAttach_Attack_gradient_sorted(model, example, device, **kwargs)
    # elif strategy == 'gradient_sorted_iouLoss':
    elif strategy == 'gradient_sorted_not_add':
        PointAttach_Attack_gradient_sorted_not_add(model, example, device, **kwargs)
    elif strategy == 'gradient_sorted_not_add-small_grad':
        PointAttach_Attack_gradient_sorted_not_add(model, example, device, **kwargs)
    elif strategy == 'gradient_sorted_iouLoss+loss' or strategy == 'gradient_sorted_iouLoss':
        PointAttach_Attack_gradient_sorted_iouLoss(model, example, device, **kwargs)
    elif strategy == 'gradient_sorted-not_add-voxelize_permutation':
        PointAttach_Attack_gradient_sorted_not_add_voxelize_permutation(model, example, device, **kwargs)
    elif strategy == 'gradient_sorted-not_add-voxelize_permutation-limitH':
        PointAttach_Attack_gradient_sorted_not_add_voxelize_permutation_limitH(model, example, device, **kwargs)
    elif 'gradient_sorted_removePoints' in strategy:
        PointAttach_Attack_gradient_sorted_removePoints(model, example, device, **kwargs)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


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


def SpareAdv_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.SpareAdv.eps
    eps_iter = cfg.SpareAdv.eps_iter
    num_steps = cfg.SpareAdv.num_steps
    strategy = cfg.SpareAdv.strategy
    attach_rate = cfg.SpareAdv.attach_rate
    dir_save_adv = cfg.SpareAdv.save_dir
    is_radius_norm = ('norm' in strategy)
    # attack
    origin_voxels = example['voxels'].clone()
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            # voxel to point
            points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
        else:
            points_adv = points_innocent_v0.copy()

        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape

        # is_object_point = object_point_filter(points_innocent, gt_boxes_and_cls.numpy()[0])
        # visualize_points(points_innocent[:, :3] * np.repeat(is_object_point.reshape(-1, 1), 3, axis=-1))
        # is_object_point_numba = object_point_filter_numba(points_innocent, gt_boxes_and_cls.numpy()[0])

        #### 计算attention
        if 'base' in strategy:
            attention_normed = np.zeros(points_adv.shape[0])
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_adv[:, :3])
            if 'density_attention' in strategy:
                attention = compute_density(pcd, is_norm=is_radius_norm)
            elif 'normal_attention' in strategy:
                attention = compute_normal(pcd, is_norm=is_radius_norm)
            attention_normed = (attention - np.min(attention)) / (np.max(attention) - np.min(attention))
        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum() * attention_normed[j]
        grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
                                          descending=True).numpy()  # 梯度排序
        # MyHist(grad_points_abs_sum[grad_points_order[:int(points_adv.shape[0] * attach_rate)]], bins=1000, title_name='grad_points_abs_sum')
        assert np.max(grad_points_order) < points_adv.shape[0]
        attach_num = int(points_adv.shape[0] * attach_rate)
        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)

        cnt = 0
        for j in range(grad_points_order.shape[0]):
            # if is_object_point[grad_points_order[i]] is not True:
            #     continue
            if cnt >= (attach_num / num_steps):
                break
            new_point = points_adv[grad_points_order[j], :]
            grad_for_new_point = grad_points[grad_points_order[j], :]
            new_point[:3] = new_point[:3] + eps_iter * np.sign(grad_for_new_point[:3])
            points_adv[grad_points_order[j], :] = new_point
            cnt += 1

        # new_points = points_adv[grad_points_order[:attach_num], :]
        # grad_for_new_point = grad_points[grad_points_order[:attach_num], :]
        # new_points[:3] = new_points[:3] + eps_iter * np.sign(grad_for_new_point[:3])
        # points_adv[grad_points_order[j], :] = new_points

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv,
        )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = points_adv[:, :3] - points_innocent_v0[:, :3]
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg.logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, 5), points_innocent_v0, points_adv,
                       token, dir_save_adv)
    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def SpareAdv_MI_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.SpareAdv.eps
    eps_iter = cfg.SpareAdv.eps_iter
    num_steps = cfg.SpareAdv.num_steps
    strategy = cfg.SpareAdv.strategy
    decay = cfg.SpareAdv.decay
    attach_rate = cfg.SpareAdv.attach_rate
    dir_save_adv = cfg.SpareAdv.save_dir
    is_radius_norm = ('norm' in strategy)
    # attack
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    momentum = np.zeros_like(points_innocent_v0)
    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            #### voxel to point
            points_adv, points_innocent_ori, momentum = voxel_to_point_numba_for_iterAttack_momentum(
                example['voxels'].cpu().detach().numpy(),
                example['num_points'].cpu().detach().numpy(),
                voxel_innocent_ori, voxel_momentum)
        else:
            points_adv = points_innocent_v0.copy()
            points_innocent_ori = points_innocent_v0.copy()

        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape

        if 'MI' in strategy:
            ### momentum gradients
            grad_points_norm = torch.norm(torch.tensor(grad_points).view(grad_points.shape[0], -1), dim=1,
                                          keepdim=True).numpy()
            grad_points = grad_points / (grad_points_norm + 1e-10)
            grad_points = grad_points + momentum * decay
            momentum = grad_points

        # is_object_point = object_point_filter(points_innocent, gt_boxes_and_cls.numpy()[0])
        # visualize_points(points_innocent[:, :3] * np.repeat(is_object_point.reshape(-1, 1), 3, axis=-1))
        # is_object_point_numba = object_point_filter_numba(points_innocent, gt_boxes_and_cls.numpy()[0])

        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        #### 计算attention
        if 'base' in strategy:
            for j in range(grad_points.shape[0]):
                grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_adv[:, :3])
            if 'density_attention' in strategy:
                attention = compute_density(pcd, is_norm=is_radius_norm)
            elif 'normal_attention' in strategy:
                attention = compute_normal(pcd, is_norm=is_radius_norm)
            attention_normed = (attention - np.min(attention)) / (np.max(attention) - np.min(attention))
            for j in range(grad_points.shape[0]):
                grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum() * attention_normed[j]

        grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
                                          descending=True).numpy()  # 梯度排序
        # MyHist(grad_points_abs_sum[grad_points_order[:int(points_adv.shape[0] * attach_rate)]], bins=1000, title_name='grad_points_abs_sum')
        assert np.max(grad_points_order) < points_adv.shape[0]
        attach_num = int(points_adv.shape[0] * attach_rate)
        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)

        # cnt = 0
        # for j in range(grad_points_order.shape[0]):
        #     # if is_object_point[grad_points_order[i]] is not True:
        #     #     continue
        #     if cnt >= attach_num:
        #         break
        #     new_point = points_adv[grad_points_order[j], :]
        #     grad_for_new_point = grad_points[grad_points_order[j], :]
        #     new_point[:3] = new_point[:3] + eps_iter * np.sign(grad_for_new_point[:3])
        #     points_adv[grad_points_order[j], :] = new_point
        #     cnt += 1

        new_points = points_adv[grad_points_order[:attach_num], :]
        grad_for_new_point = grad_points[grad_points_order[:attach_num], :]
        new_points[:, :3] = new_points[:, :3] + eps_iter * np.sign(grad_for_new_point[:, :3])
        points_adv[grad_points_order[:attach_num], :] = new_points

        ##### restrict permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent_ori[:, :3]), min=-Epsilon,
                            max=Epsilon).numpy()
        points_adv[:, :3] = points_innocent_ori[:, :3] + delta

        ##### point to voxel
        voxels, coordinates, num_points, voxel_innocent_ori, voxel_momentum = \
            cfg.voxelization.voxel_generator.generate_for_iterAttack_momentum(
                points_adv, points_innocent_ori=points_innocent_ori, momentum=momentum
            )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = delta
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg.logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    points_innocent_ori.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, 5), points_innocent_ori, points_adv,
                       token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def SpareAdv_PGD_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.SpareAdv.eps
    eps_iter = cfg.SpareAdv.eps_iter
    num_steps = cfg.SpareAdv.num_steps
    strategy = cfg.SpareAdv.strategy
    decay = cfg.SpareAdv.decay
    attach_rate = cfg.SpareAdv.attach_rate
    dir_save_adv = cfg.SpareAdv.save_dir
    is_radius_norm = ('norm' in strategy)
    # attack
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    momentum = np.zeros_like(points_innocent_v0)
    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            #### voxel to point
            points_adv, points_innocent_ori, momentum = voxel_to_point_numba_for_iterAttack_momentum(
                example['voxels'].cpu().detach().numpy(),
                example['num_points'].cpu().detach().numpy(),
                voxel_innocent_ori, voxel_momentum)
        else:
            points_adv = points_innocent_v0.copy()
            points_innocent_ori = points_innocent_v0.copy()

        losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape
        if 'MI' in strategy:
            ### momentum gradients
            grad_points_norm = torch.norm(torch.tensor(grad_points).view(grad_points.shape[0], -1), dim=1,
                                          keepdim=True).numpy()
            grad_points = grad_points / (grad_points_norm + 1e-10)
            grad_points = grad_points + momentum * decay
            momentum = grad_points

        # is_object_point = object_point_filter(points_innocent, gt_boxes_and_cls.numpy()[0])
        # visualize_points(points_innocent[:, :3] * np.repeat(is_object_point.reshape(-1, 1), 3, axis=-1))
        # is_object_point_numba = object_point_filter_numba(points_innocent, gt_boxes_and_cls.numpy()[0])

        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        #### 计算attention
        if 'base' in strategy:
            for j in range(grad_points.shape[0]):
                grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_adv[:, :3])
            if 'density_attention' in strategy:
                attention = compute_density(pcd, is_norm=is_radius_norm)
            elif 'normal_attention' in strategy:
                attention = compute_normal(pcd, is_norm=is_radius_norm)
            attention_normed = (attention - np.min(attention)) / (np.max(attention) - np.min(attention))
            for j in range(grad_points.shape[0]):
                grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum() * attention_normed[j]

        grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
                                          descending=True).numpy()  # 梯度排序
        # MyHist(grad_points_abs_sum[grad_points_order[:int(points_adv.shape[0] * attach_rate)]], bins=1000, title_name='grad_points_abs_sum')
        assert np.max(grad_points_order) < points_adv.shape[0]
        attach_num = int(points_adv.shape[0] * attach_rate)
        # visulize with color
        # colors = np.ones(shape=(points_innocent.shape[0], 3), dtype=np.float) * 0.5
        # colors[grad_points_order[:attach_num], :] = [1, 1, 1]
        # # colors = np.array([[1.0, 1.0, 1.0]]).reshape(-1, 3) * (np.array(grad_points_abs_sum / np.max(grad_points_abs_sum)).reshape(-1, 1).repeat(3, axis=1))
        # visualize_points_colored(points_innocent, colors)

        cnt = 0
        for j in range(grad_points_order.shape[0]):
            # if is_object_point[grad_points_order[i]] is not True:
            #     continue
            if cnt >= (attach_num / num_steps):
                break
            new_point = points_adv[grad_points_order[j], :]
            grad_for_new_point = grad_points[grad_points_order[j], :]
            new_point[:3] = new_point[:3] + eps_iter * np.sign(grad_for_new_point[:3])
            points_adv[grad_points_order[j], :] = new_point
            cnt += 1

        # new_points = points_adv[grad_points_order[:attach_num], :]
        # grad_for_new_point = grad_points[grad_points_order[:attach_num], :]
        # new_points[:3] = new_points[:3] + eps_iter * np.sign(grad_for_new_point[:3])
        # points_adv[grad_points_order[j], :] = new_points

        ##### restrict permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent_ori[:, :3]), min=-Epsilon,
                            max=Epsilon).numpy()
        points_adv[:, :3] = points_innocent_ori[:, :3] + delta

        ##### point to voxel
        voxels, coordinates, num_points, voxel_innocent_ori, voxel_momentum = \
            cfg.voxelization.voxel_generator.generate_for_iterAttack_momentum(
                points_adv, points_innocent_ori=points_innocent_ori, momentum=momentum
            )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = delta
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg.logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    points_innocent_ori.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, 5), points_innocent_ori, points_adv,
                       token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


from tools.Lib.get_points_in_3d_boxes import *


def AdaptiveEPS_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.AdaptiveEPS.eps
    eps_ratio = cfg.AdaptiveEPS.eps_ratio
    num_steps = cfg.AdaptiveEPS.num_steps
    strategy = cfg.AdaptiveEPS.strategy
    attach_rate = cfg.AdaptiveEPS.attach_rate
    dir_save_adv = cfg.AdaptiveEPS.save_dir
    is_radius_norm = ('norm' in strategy)
    # attack
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            # voxel to point
            points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
        else:
            points_adv = points_innocent_v0.copy()

        # 计算loss
        losses, predictions = model(example, return_loss=True, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pred_labels = predictions[0]['label_preds']

        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape
        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        # grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
        #                                   descending=True).numpy()  # 梯度排序
        point_indices = points_in_rbbox(points_adv, pred_boxes.cpu().detach().numpy())
        num_obj = pred_boxes.shape[0]
        obj_indices = []
        for i in range(num_obj):
            points_in_box = points_adv[point_indices[:, i]]
            grad_in_box = grad_points[point_indices[:, i]]
            grad_sum_in_box = grad_points_abs_sum[point_indices[:, i]]

            grad_order_in_box = torch.argsort(torch.tensor(grad_sum_in_box), dim=-1,
                                              descending=True).numpy()  # 梯度排序
            modify_num_in_box = int(points_in_box.shape[0] * attach_rate)
            min_edge = pred_boxes[i, 3:6].min().cpu().detach().numpy()
            eps = min_edge * eps_ratio

            grad_for_new_point = grad_in_box[grad_order_in_box[:modify_num_in_box], :]
            points_in_box[grad_order_in_box[:modify_num_in_box], :3] += eps * np.sign(grad_for_new_point[:, :3])

            points_adv[point_indices[:, i]] = points_in_box

        ##### restrict permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent_v0[:, :3]), min=-Epsilon,
                            max=Epsilon).numpy()
        points_adv[:, :3] = points_innocent_v0[:, :3] + delta

        # MyHist(grad_points_abs_sum[grad_points_order[:int(points_adv.shape[0] * attach_rate)]], bins=1000, title_name='grad_points_abs_sum')
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0)
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0, is_show_adv=False)

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = points_adv[:, :3] - points_innocent_v0[:, :3]
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg.logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, 5), points_innocent_v0, points_adv,
                       token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


# import cupy as cp
import re


def AdaptiveEPS_MI_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.AdaptiveEPS.eps
    eps_ratio = cfg.AdaptiveEPS.eps_ratio
    num_steps = cfg.AdaptiveEPS.num_steps
    strategy = cfg.AdaptiveEPS.strategy
    decay = cfg.AdaptiveEPS.decay
    attach_rate = cfg.AdaptiveEPS.attach_rate
    dir_save_adv = cfg.AdaptiveEPS.save_dir
    is_radius_norm = ('norm' in strategy)
    # attack
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    momentum = np.zeros_like(points_innocent_v0)
    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            #### voxel to point
            points_adv, points_innocent_ori, momentum = voxel_to_point_numba_for_iterAttack_momentum(
                example['voxels'].cpu().detach().numpy(),
                example['num_points'].cpu().detach().numpy(),
                voxel_innocent_ori, voxel_momentum)
        else:
            points_adv = points_innocent_v0.copy()
            points_innocent_ori = points_innocent_v0.copy()

        # 计算loss
        losses, predictions = model(example, return_loss=True, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pred_labels = predictions[0]['label_preds']
        ## visualize_points_without_color(points_adv, {'box3d_lidar':pred_boxes.detach().cpu(), 'scores':pred_scores.detach().cpu(), 'label_preds':pred_labels.detach().cpu()})
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape

        if 'MI' in strategy:
            ### momentum gradients
            grad_points_norm = torch.norm(torch.tensor(grad_points).view(grad_points.shape[0], -1), dim=1,
                                          keepdim=True).numpy()
            grad_points = grad_points / (grad_points_norm + 1e-10)
            grad_points = grad_points + momentum * decay
            momentum = grad_points
        elif 'PGD' in strategy:
            pass

        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        # grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
        #                                   descending=True).numpy()  # 梯度排序
        point_indices = points_in_rbbox(points_adv, pred_boxes.cpu().detach().numpy())
        num_obj = pred_boxes.shape[0]

        if 'filterOnce' in strategy:
            points_selected_by_bbox = np.zeros(points_adv.shape[0], dtype=np.bool_)
            for i in range(num_obj):
                points_selected_by_bbox = np.logical_or(points_selected_by_bbox, point_indices[:, i])
            if 'OppoPSel_v2' in strategy:
                points_selected_by_bbox = ~points_selected_by_bbox
            points_in_box = points_adv[points_selected_by_bbox]
            grad_in_box = grad_points[points_selected_by_bbox]
            grad_sum_in_box = grad_points_abs_sum[points_selected_by_bbox]

            grad_order_in_box = torch.argsort(torch.tensor(grad_sum_in_box), dim=-1,
                                              descending=True).numpy()  # 梯度排序
            modify_num_in_box = int(points_in_box.shape[0] * attach_rate)
            ## 扰动量
            if 'fixedEPS' in strategy:
                eps = float(strategy.split('fixedEPS_')[-1])
            else:
                return
                # adaptive EPS
                maximum_BBedges_for_points = np.zeros(points_adv.shape[0], dtype=np.float)
                for i in range(num_obj):
                    min_BBedge = np.min(pred_boxes[i, 3:6].cpu().detach().numpy())
                    maximum_BBedges_for_points[point_indices[:, i]] = np.where(
                        maximum_BBedges_for_points[point_indices[:, i]] >= min_BBedge,
                        maximum_BBedges_for_points[point_indices[:, i]], min_BBedge)

                eps = (maximum_BBedges_for_points * eps_ratio)[points_selected_by_bbox][
                    grad_order_in_box[:modify_num_in_box]]
                eps = eps.reshape(-1, 1)
                eps = np.concatenate([eps, eps, eps], axis=1)

            if 'OppoPSel_v1' in strategy:
                if 'OppoPSel_v1.1' in strategy:
                    # LowRate = float(re.findall(r"[-+]?\d*\.\d+|\d+", "Current Level: -13.2 db or 14.2 or 3")[0])
                    # LowRate = float(strategy.split('OppoPSel_v1.1_')[-1].split('-')[-2])
                    LowRate = cfg.AdaptiveEPS.LowRate
                    L_modify_num_in_box = int(points_in_box.shape[0] * LowRate)
                    H_modify_num_in_box = int(points_in_box.shape[0] * (LowRate + attach_rate))
                    grad_for_new_point = grad_in_box[grad_order_in_box[L_modify_num_in_box:H_modify_num_in_box], :]
                    points_in_box[grad_order_in_box[L_modify_num_in_box:H_modify_num_in_box], :3] += eps * np.sign(
                        grad_for_new_point[:, :3])
                else:
                    grad_for_new_point = grad_in_box[grad_order_in_box[modify_num_in_box:], :]
                    points_in_box[grad_order_in_box[modify_num_in_box:], :3] += eps * np.sign(grad_for_new_point[:, :3])
            # elif 'OppoPSel_v2' in strategy:
            else:
                grad_for_new_point = grad_in_box[grad_order_in_box[:modify_num_in_box], :]
                points_in_box[grad_order_in_box[:modify_num_in_box], :3] += eps * np.sign(grad_for_new_point[:, :3])

            points_adv[points_selected_by_bbox] = points_in_box

        else:
            return
            for i in range(num_obj):
                points_in_box = points_adv[point_indices[:, i]]
                grad_in_box = grad_points[point_indices[:, i]]
                grad_sum_in_box = grad_points_abs_sum[point_indices[:, i]]

                grad_order_in_box = torch.argsort(torch.tensor(grad_sum_in_box), dim=-1,
                                                  descending=True).numpy()  # 梯度排序
                modify_num_in_box = int(points_in_box.shape[0] * attach_rate)
                min_edge = pred_boxes[i, 3:6].min().cpu().detach().numpy()
                eps = min_edge * eps_ratio

                grad_for_new_point = grad_in_box[grad_order_in_box[:modify_num_in_box], :]
                points_in_box[grad_order_in_box[:modify_num_in_box], :3] += eps * np.sign(grad_for_new_point[:, :3])

                points_adv[point_indices[:, i]] = points_in_box

        ##### restrict permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent_ori[:, :3]), min=-Epsilon,
                            max=Epsilon).numpy()
        points_adv[:, :3] = points_innocent_ori[:, :3] + delta

        # MyHist(grad_points_abs_sum[grad_points_order[:int(points_adv.shape[0] * attach_rate)]], bins=1000, title_name='grad_points_abs_sum')
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0)
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0, is_show_adv=False)

        ##### point to voxel
        voxels, coordinates, num_points, voxel_innocent_ori, voxel_momentum = \
            cfg.voxelization.voxel_generator.generate_for_iterAttack_momentum(
                points_adv, points_innocent_ori=points_innocent_ori, momentum=momentum
            )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = delta
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg.logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    # points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    points_innocent_ori.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))
    # grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    if "NuScenesDataset" in cfg.dataset_type:
        save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, example['points'][0].shape[1]),
                           points_innocent_ori, points_adv,
                           token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


def AdaptiveEPS_MI_Attack_light(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.AdaptiveEPS.eps
    eps_ratio = cfg.AdaptiveEPS.eps_ratio
    num_steps = cfg.AdaptiveEPS.num_steps
    strategy = cfg.AdaptiveEPS.strategy
    decay = cfg.AdaptiveEPS.decay
    attach_rate = cfg.AdaptiveEPS.attach_rate
    dir_save_adv = cfg.AdaptiveEPS.save_dir

    # attack
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    for n_step in range(num_steps):

        # 计算loss
        example['voxels'].requires_grad = True
        losses, predictions = model(example, return_loss=True, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        ## visualize_points_without_color(points_adv, {'box3d_lidar':pred_boxes.detach().cpu(), 'scores':pred_scores.detach().cpu(), 'label_preds':pred_labels.detach().cpu()})
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        # assert grad_points.shape == points_adv.shape

        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()

        ### voxel map to points
        points_adv, voxelMap = voxel_to_point_numba_voxelMap(example['voxels'].cpu().detach().numpy(),
                                                             example['num_points'].cpu().detach().numpy())

        point_indices = points_in_rbbox(points_adv, pred_boxes.cpu().detach().numpy())
        num_obj = pred_boxes.shape[0]

        if 'filterOnce' in strategy:
            points_selected_by_bbox = np.zeros(points_adv.shape[0], dtype=np.bool_)
            for i in range(num_obj):
                points_selected_by_bbox = np.logical_or(points_selected_by_bbox, point_indices[:, i])
            points_in_box = points_adv[points_selected_by_bbox]
            grad_in_box = grad_points[points_selected_by_bbox]
            grad_sum_in_box = grad_points_abs_sum[points_selected_by_bbox]

            grad_order_in_box = torch.argsort(torch.tensor(grad_sum_in_box), dim=-1,
                                              descending=True).numpy()  # 梯度排序
            modify_num_in_box = int(points_in_box.shape[0] * attach_rate)
            ## 扰动量
            if 'fixedEPS' in strategy:
                eps = float(strategy.split('fixedEPS_')[-1])

            grad_for_new_point = grad_in_box[grad_order_in_box[:modify_num_in_box], :]
            points_in_box[grad_order_in_box[:modify_num_in_box], :3] += eps * np.sign(grad_for_new_point[:, :3])

            points_adv[points_selected_by_bbox] = points_in_box

        ##### restrict permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent_v0[:, :3]), min=-Epsilon,
                            max=Epsilon).numpy()
        points_adv[:, :3] = points_innocent_v0[:, :3] + delta

        ### point map to voxel
        voxel_adv = points_to_voxel_numba_voxelMap(points_adv, voxelMap, tuple(list(example['voxels'].shape)))
        example['voxels'] = torch.from_numpy(voxel_adv).to(device)
        assert points_adv.shape[0] == example['num_points'].sum()

        cfg.logger.info('Iter {}: detection_loss={:.6f}'.format(n_step, sum(losses['loss'])))

    ##### point to voxel
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(points_adv)
    save_to_example(example, voxels, coordinates, num_points, device)

    dist_points = delta
    dist_points_l0 = np.abs(dist_points).sum(1)
    dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
    dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
    cfg.logger.info(
        'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    # points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))
    # grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    # if "NuScenesDataset" in cfg.dataset_type:
    #     save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, example['points'][0].shape[1]),
    #                        points_innocent_ori, points_adv,
    #                        token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions

def AdaptiveEPS_MI_Attack_cuda(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.AdaptiveEPS.eps
    eps_ratio = cfg.AdaptiveEPS.eps_ratio
    num_steps = cfg.AdaptiveEPS.num_steps
    strategy = cfg.AdaptiveEPS.strategy
    decay = cfg.AdaptiveEPS.decay
    attach_rate = cfg.AdaptiveEPS.attach_rate
    dir_save_adv = cfg.AdaptiveEPS.save_dir
    is_radius_norm = ('norm' in strategy)
    # attack
    points_innocent_v0 = torch.from_numpy(voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                                               example['num_points'].cpu().detach().numpy())).to(device)
    momentum = torch.zeros_like(points_innocent_v0).to(device)
    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            #### voxel to point
            points_adv, points_innocent_ori, momentum = voxel_to_point_numba_for_iterAttack_momentum(
                example['voxels'].cpu().detach().numpy(),
                example['num_points'].cpu().detach().numpy(),
                voxel_innocent_ori, voxel_momentum)
            points_adv = torch.from_numpy(points_adv).to(device)
            points_innocent_ori = torch.from_numpy(points_innocent_ori).to(device)
            momentum = torch.from_numpy(momentum).to(device)
        else:
            points_adv = points_innocent_v0.clone()
            points_innocent_ori = points_innocent_v0.clone()

        # 计算loss
        losses, predictions = model(example, return_loss=True, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pred_labels = predictions[0]['label_preds']
        ## visualize_points_without_color(points_adv, {'box3d_lidar':pred_boxes.detach().cpu(), 'scores':pred_scores.detach().cpu(), 'label_preds':pred_labels.detach().cpu()})
        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = torch.from_numpy(grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                                      example['num_points'].cpu().detach().numpy())).to(
            device)
        assert grad_points.shape == points_adv.shape

        if 'MI' in strategy:
            ### momentum gradients
            grad_points_norm = torch.norm(grad_points.view(grad_points.shape[0], -1), dim=1,
                                          keepdim=True)
            grad_points = grad_points / (grad_points_norm + 1e-10)
            grad_points = grad_points + momentum * decay
            momentum = grad_points
        elif 'PGD' in strategy:
            pass

        grad_points_abs_sum = torch.zeros(grad_points.shape[0], dtype=torch.float32, device=device)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = torch.abs(grad_points[j, :3]).sum()
        # grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
        #                                   descending=True).numpy()  # 梯度排序
        point_indices = points_in_rbbox(points_adv.cpu().detach().numpy(), pred_boxes.cpu().detach().numpy())
        point_indices = torch.from_numpy(point_indices).to(device)
        num_obj = pred_boxes.shape[0]

        if 'filterOnce' in strategy:

            ## 选点
            points_selected_by_bbox = torch.zeros(points_adv.shape[0], dtype=torch.bool, device=device)
            for i in range(num_obj):
                points_selected_by_bbox = points_selected_by_bbox | point_indices[:, i]
            points_in_box = points_adv[points_selected_by_bbox]
            grad_in_box = grad_points[points_selected_by_bbox]
            grad_sum_in_box = grad_points_abs_sum[points_selected_by_bbox]

            grad_order_in_box = torch.argsort(grad_sum_in_box, dim=-1,
                                              descending=True)  # 梯度排序
            modify_num_in_box = int(points_in_box.shape[0] * attach_rate)

            ## 扰动量
            if 'fixedEPS' in strategy:
                eps = float(strategy.split('fixedEPS_')[-1])
            else:
                # adaptive EPS
                maximum_BBedges_for_points = torch.zeros(points_adv.shape[0], dtype=torch.float32, device=device)
                for i in range(num_obj):
                    min_BBedge = torch.min(pred_boxes[i, 3:6])
                    maximum_BBedges_for_points[point_indices[:, i]] = torch.where(
                        maximum_BBedges_for_points[point_indices[:, i]] >= min_BBedge,
                        maximum_BBedges_for_points[point_indices[:, i]], min_BBedge)

                eps = (maximum_BBedges_for_points * eps_ratio)[points_selected_by_bbox][
                    grad_order_in_box[:modify_num_in_box]]
                eps = eps.reshape(-1, 1)
                eps = torch.cat([eps, eps, eps], dim=1)

            if 'OppoPSel_v1' in strategy:
                grad_for_new_point = grad_in_box[grad_order_in_box[modify_num_in_box:], :]
                points_in_box[grad_order_in_box[modify_num_in_box:], :3] += eps * np.sign(grad_for_new_point[:, :3])
            # elif 'OppoPSel_v2' in strategy:
            else:
                grad_for_new_point = grad_in_box[grad_order_in_box[:modify_num_in_box], :]
                points_in_box[grad_order_in_box[:modify_num_in_box], :3] += eps * torch.sign(grad_for_new_point[:, :3])

            points_adv[points_selected_by_bbox] = points_in_box

        else:
            return 'Mistakes'
            for i in range(num_obj):
                points_in_box = points_adv[point_indices[:, i]]
                grad_in_box = grad_points[point_indices[:, i]]
                grad_sum_in_box = grad_points_abs_sum[point_indices[:, i]]

                grad_order_in_box = torch.argsort(torch.tensor(grad_sum_in_box), dim=-1,
                                                  descending=True).numpy()  # 梯度排序
                modify_num_in_box = int(points_in_box.shape[0] * attach_rate)
                min_edge = pred_boxes[i, 3:6].min().cpu().detach().numpy()
                eps = min_edge * eps_ratio

                grad_for_new_point = grad_in_box[grad_order_in_box[:modify_num_in_box], :]
                points_in_box[grad_order_in_box[:modify_num_in_box], :3] += eps * np.sign(grad_for_new_point[:, :3])

                points_adv[point_indices[:, i]] = points_in_box

        ##### restrict permutation
        delta = torch.clamp(points_adv[:, :3] - points_innocent_ori[:, :3], min=-Epsilon,
                            max=Epsilon)
        points_adv[:, :3] = points_innocent_ori[:, :3] + delta

        # MyHist(grad_points_abs_sum[grad_points_order[:int(points_adv.shape[0] * attach_rate)]], bins=1000, title_name='grad_points_abs_sum')
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0)
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0, is_show_adv=False)

        ##### point to voxel
        voxels, coordinates, num_points, voxel_innocent_ori, voxel_momentum = \
            cfg.voxelization.voxel_generator.generate_for_iterAttack_momentum(
                points_adv.cpu().detach().numpy(), points_innocent_ori=points_innocent_ori.cpu().detach().numpy(),
                momentum=momentum.cpu().detach().numpy()
            )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = delta
        dist_points_l0 = torch.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0.cpu().detach().numpy(), ord=0, axis=None)
        dist_l2 = torch.norm(dist_points, p=2)
        cfg.logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.cpu().detach().numpy().tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.cpu().detach().numpy().tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    points_innocent_ori.cpu().detach().numpy().tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))
    grad_points.cpu().detach().numpy().tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    save_origin_points_cuda(cfg, example['points'][0].reshape(-1, 5), points_innocent_ori, points_adv,
                            token, dir_save_adv, device)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions



from tools.SlowLiDAR.dist_lib.dist_metric import L2Dist, ChamferDist, HausdorffDist
from tools.SlowLiDAR.attack_lib.attack_utils import total_loss
from tools.SlowLiDAR.processing.postprocessing import filter_pred


def SlowLiDAR_voxel_light(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    eps = cfg.SlowLiDAR.eps
    eps_iter = cfg.SlowLiDAR.eps_iter
    num_steps = cfg.SlowLiDAR.num_steps
    strategy = cfg.SlowLiDAR.strategy
    dir_save_adv = cfg.SlowLiDAR.save_dir
    cls_threshold = 0.05
    nms_iou_threshold = 0.1
    nms_top = 3000

    # voxel to point
    voxel_innocent_ori = example['voxels'].clone()
    example['voxels'].requires_grad = True
    perm_mask = permutation_mask(example['voxels'].shape, example['num_points'])
    # ori_data = torch.masked_select(voxel_innocent_ori, perm_mask).view(1, -1, 3)

    for n_step in range(num_steps):
        if n_step == 0:
            if cfg.SlowLiDAR.random_start:
                # Starting at a uniformly random point
                permutation = torch.empty_like(example['voxels']).uniform_(-eps, eps) * perm_mask
                example['voxels'] = example['voxels'] + permutation

        # Calculate loss
        # losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
        # before_corners, after_corners = filter_pred(pred, cls_threshold, nms_iou_threshold, nms_top)

        # compute loss and backward
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        # pred_scores = predictions[0]['scores']  # N
        # pre_labels = predictions[0]['label_preds']
        # adv_data = torch.masked_select(example['voxels'], perm_mask).view(1, -1, 3)
        # loss = total_loss(predictions, pred_boxes, cls_threshold, ori_data, adv_data)
        t1 = time.time()
        loss = total_loss(predictions, pred_boxes, cls_threshold)
        t_loss = time.time()
        # Update adversarial images
        grad = torch.autograd.grad(loss, example['voxels'],
                                   retain_graph=False, create_graph=False)[0]
        # mask voxel内空白的点、intensity、timestamp
        permutation = eps_iter * grad.sign() * perm_mask
        example['voxels'] = example['voxels'] + permutation * (-1.0)
        delta = torch.clamp(example['voxels'] - voxel_innocent_ori, min=-eps, max=eps)
        example['voxels'] = voxel_innocent_ori + delta
        cfg.logger.info('Iter {}: detection_loss={:.6f}'.format(n_step, loss))

        # light version
        ## coordinate adjustment
    points_adv, points_innocent_ori = voxel_to_point_numba_for_iterAttack(
        example['voxels'].cpu().detach().numpy(),
        example['num_points'].cpu().detach().numpy(),
        voxel_innocent_ori.cpu().detach().numpy())
    # point to voxel
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(points_adv)
    save_to_example(example, voxels, coordinates, num_points, device)

    # save
    assert points_adv.shape == points_innocent_ori.shape
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_ori.tofile(os.path.join(dir_save_adv, token + '-innocent_ori.bin'))
    # points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    # grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    if "NuScenesDataset" in cfg.dataset_type:
        save_origin_points(cfg,
                           example['points'][0].detach().cpu().numpy().reshape(-1, example['points'][0].shape[1]),
                           points_innocent_ori, points_adv,
                           token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions

from tools.GSDA_Attacker.loss_utils import _get_kappa_ori
from tools.GSDA_Attacker.utility import *
import torch_dct
# import tools.GSDA_Attacker.modules.functional as F
from tools.GSDA_Attacker import spectral_attack


def select_points_by_GFT(cfg, points, K=10):
    x = torch.tensor(points[:, :3]).unsqueeze(0)  # (b,n,3)
    if cfg.test_time:
        import time
        time_begin = time.time()
    v, laplacian, u = spectral_attack.eig_vector(x, K)
    if cfg.test_time:
        print("gft time: {}, point.shape: {}".format(time.time() - time_begin, points.shape))
    u = u.unsqueeze(-1)
    u_ = torch.cat((torch.ones_like(u).to(u.device), u, u * u, u * u * u, u * u * u * u),
                   dim=-1)  # (b, n, 5)
    # print('v', v.shape, 'x', x.shape, 'u', u_.shape)
    x_ = torch.einsum('bij,bjk->bik', v.transpose(1, 2), x)  # (b,n,3)

    low_fred_bound = int(x_.shape[1] * 32 / 1024 + 0.5)
    mid_fred_bound = int(x_.shape[1] * 256 / 1024 + 0.5)

    eps_GFT = cfg.eps_GFT
    x_perturb_low = x_.clone()
    x_perturb_low[0, :low_fred_bound, :] = x_perturb_low[0, :low_fred_bound, :] * (1 + eps_GFT)
    GFT_pc_low = torch.einsum('bij,bjk->bik', v, x_perturb_low).squeeze().cpu().detach().numpy()

    '''    
    plt_GFT(x_.cpu().detach().numpy()[0])
    
    eps_GFT = 0.001
    x_perturb_low = x_.clone()
    x_perturb_mid = x_.clone()
    x_perturb_high = x_.clone()
    x_perturb_low[0, :low_fred_bound, :] = x_perturb_low[0, :low_fred_bound, :] * (1 + eps_GFT)
    x_perturb_mid[0, low_fred_bound:mid_fred_bound, :] = x_perturb_mid[0, low_fred_bound:mid_fred_bound, :] * (1 + eps_GFT)
    x_perturb_high[0, mid_fred_bound:, :] = x_perturb_high[0, mid_fred_bound:, :] * (1 + eps_GFT)
    GFT_pc_low = torch.einsum('bij,bjk->bik', v, x_perturb_low).squeeze().cpu().detach().numpy()
    GFT_pc_mid = torch.einsum('bij,bjk->bik', v, x_perturb_mid).squeeze().cpu().detach().numpy()
    GFT_pc_high = torch.einsum('bij,bjk->bik', v, x_perturb_high).squeeze().cpu().detach().numpy()
    
    visualize_points(points)
    visualize_points(GFT_pc_low)
    visualize_points(GFT_pc_mid)
    visualize_points(GFT_pc_high)
    '''
    return GFT_pc_low


def GSDA_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg'].GSDA
    cfg_logger = kwargs['cfg'].logger
    if cfg.is_debug:
        token = example['metadata'][0]['token']
        if 'a860c02c06e54d9dbe83ce7b694f6c17' not in token:
            with torch.no_grad():
                predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
            return predictions

    Epsilon = cfg.eps
    eps_ratio = cfg.eps_ratio
    num_steps = cfg.num_steps
    strategy = cfg.strategy
    attach_rate = cfg.attach_rate
    dir_save_adv = cfg.save_dir
    max_points = cfg.max_points
    KNN = cfg.KNN
    is_radius_norm = ('norm' in strategy)
    # attack
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    example['voxels'].requires_grad = True
    for n_step in range(num_steps):
        if not n_step == 0:
            # voxel to point
            points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
        else:
            points_adv = points_innocent_v0.copy()

        # 计算loss
        losses, predictions = model(example, return_loss=True, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pred_labels = predictions[0]['label_preds']

        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape
        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        # grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
        #                                   descending=True).numpy()  # 梯度排序
        point_indices = points_in_rbbox(points_adv, pred_boxes.cpu().detach().numpy())
        num_obj = pred_boxes.shape[0]
        if cfg.test_time:
            time_begin = time.time()
        if 'base' in strategy:
            for i in range(num_obj):
                points_in_box = points_adv[point_indices[:, i]]

                if points_in_box.shape[0] > max_points or points_in_box.shape[0] <= KNN:
                    continue
                iGFT_points = select_points_by_GFT(cfg, points_in_box)
                points_adv[point_indices[:, i], :3] = iGFT_points

        if cfg.test_time:
            cfg_logger.info(
                "gft all instance time: {}, points_adv.shape: {}".format(time.time() - time_begin, points_adv.shape))

        ##### restrict permutation
        delta = torch.clamp(torch.tensor(points_adv[:, :3]) - torch.tensor(points_innocent_v0[:, :3]), min=-Epsilon,
                            max=Epsilon).numpy()
        points_adv[:, :3] = points_innocent_v0[:, :3] + delta

        # MyHist(grad_points_abs_sum[grad_points_order[:int(points_adv.shape[0] * attach_rate)]], bins=1000, title_name='grad_points_abs_sum')
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0)
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0, is_show_adv=False)

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = points_adv[:, :3] - points_innocent_v0[:, :3]
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg_logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


from tools.Lib.kruskal import *


def edgeAdv_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.AdaptiveEPS.eps
    eps_ratio = cfg.AdaptiveEPS.eps_ratio
    num_steps = cfg.AdaptiveEPS.num_steps
    strategy = cfg.AdaptiveEPS.strategy
    attach_rate = cfg.AdaptiveEPS.attach_rate
    dir_save_adv = cfg.AdaptiveEPS.save_dir
    is_radius_norm = ('norm' in strategy)
    # attack
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())

    for n_step in range(num_steps):
        example['voxels'].requires_grad = True
        if not n_step == 0:
            # voxel to point
            points_adv = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
        else:
            points_adv = points_innocent_v0.copy()

        # 计算loss
        losses, predictions = model(example, return_loss=True, **kwargs, is_eval_after_attack=True)
        pred_boxes = predictions[0]['box3d_lidar'][:, [0, 1, 2, 3, 4, 5, -1]]  # N * 7
        pred_scores = predictions[0]['scores']  # N
        pred_labels = predictions[0]['label_preds']

        model.zero_grad()
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
        assert grad_voxel.shape == example['voxels'].shape
        grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                     example['num_points'].cpu().detach().numpy())
        assert grad_points.shape == points_adv.shape
        grad_points_abs_sum = np.zeros(grad_points.shape[0], dtype=np.float)
        for j in range(grad_points.shape[0]):
            grad_points_abs_sum[j] = np.abs(grad_points[j, :3]).sum()
        grad_points_order = torch.argsort(torch.tensor(grad_points_abs_sum), dim=-1,
                                          descending=True).numpy()  # 梯度排序
        point_indices = points_in_rbbox(points_adv, pred_boxes.cpu().detach().numpy())
        num_obj = pred_boxes.shape[0]
        graph_kruskal = Graph_Kruskal(num_obj)
        for i in range(num_obj):
            for j in range(i + 1, num_obj):
                dist = np.linalg.norm(x=(pred_boxes[i, :3] - pred_boxes[j, :3]), ord=2, axis=None)
                graph_kruskal.add_edge(i, j, dist)

        obj_indices = []
        for i in range(num_obj):
            points_in_box = points_adv[point_indices[:, i]]
            grad_in_box = grad_points[point_indices[:, i]]
            grad_sum_in_box = grad_points_abs_sum[point_indices[:, i]]

            grad_order_in_box = torch.argsort(torch.tensor(grad_sum_in_box), dim=-1,
                                              descending=True).numpy()  # 梯度排序
            modify_num_in_box = int(points_in_box.shape[0] * attach_rate)
            min_edge = pred_boxes[i, 3:6].min().cpu().detach().numpy()
            eps = min_edge * eps_ratio
            if not Epsilon == 'Nan':
                eps = min(eps, Epsilon)

            grad_for_new_point = grad_in_box[grad_order_in_box[:modify_num_in_box], :]
            points_in_box[grad_order_in_box[:modify_num_in_box], :3] += eps * np.sign(grad_for_new_point[:, :3])

            points_adv[point_indices[:, i]] = points_in_box

        # MyHist(grad_points_abs_sum[grad_points_order[:int(points_adv.shape[0] * attach_rate)]], bins=1000, title_name='grad_points_abs_sum')
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0)
        # visualize_points_colorModification_debug(points_adv, points_innocent_v0, is_show_adv=False)

        # point to voxel
        voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(
            points_adv
        )
        save_to_example(example, voxels, coordinates, num_points, device)

        dist_points = points_adv[:, :3] - points_innocent_v0[:, :3]
        dist_points_l0 = np.abs(dist_points).sum(1)
        dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
        dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
        cfg.logger.info(
            'Iter {}: l0={:02.1f}%, l2={:.1f})'.format(n_step, dist_l0 / dist_points.shape[0] * 100, dist_l2))

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions


# from modules.voxelization import Voxelization as PCVNN_Voxelization
# def pvcnn_Attack(model, example, device, **kwargs):
#     cfg = kwargs['cfg'].pvcnn
#     cfg_logger = kwargs['cfg'].logger
#     Epsilon = cfg.Epsilon
#     dir_save_adv = cfg.save_dir
#
#     resolution = [1440, 1440, 40]
#     pvcnn_voxelization = PCVNN_Voxelization(resolution)
#     points_innocent_v0 = example['points']
#     points_adv = points_innocent_v0.clone()
#     example['points'].requires_grad = True
#
#     features = example['points'].permute(0, 2, 1)
#     coords = features[:, :3, :]
#     voxel_features, voxel_coords = pvcnn_voxelization(features, coords)
#
#     points_mean_centerp, voxel_coords_centerp = voxels_to_pointsmean_by_mask(voxel_features, voxel_coords)
#     example['points_mean_centerp'] = points_mean_centerp
#     example['voxel_coords_centerp'] = voxel_coords_centerp
#     losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
#     # Get gradient
#     model.zero_grad()
#     with torch.autograd.set_detect_anomaly(True):
#         grad_points = \
#             torch.autograd.grad(sum(losses['loss']), example['points'], retain_graph=False, create_graph=False)[0]
#
#     # Add permutation
#     permutation = Epsilon * np.sign(grad_points[:, :3])
#     points_adv[:, :3] = points_adv[:, :3] + permutation
#
#     # point to voxel
#     voxels, coordinates, num_points = Voxelize.voxel_generator.generate(
#         points_adv, max_voxels=max_voxels
#     )
#     save_to_example(example, voxels, coordinates, num_points, device)
#
#     dist_points = points_adv[:, :3] - points_innocent_v0[:, :3]
#     dist_points_l0 = np.abs(dist_points).sum(1)
#     dist_l0 = np.linalg.norm(dist_points_l0, ord=0, axis=None)
#     dist_l2 = np.linalg.norm(dist_points, ord=2, axis=None)
#     cfg_logger.info(
#         'l0={:02.1f}%, l2={:.1f})'.format(dist_l0 / dist_points.shape[0] * 100, dist_l2))
#     # save
#     token = example['metadata'][0]['token']
#     points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
#     points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
#     grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
#
#     with torch.no_grad():
#         predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
#     return predictions

def MFS_Appr_Attack(model, example, device, **kwargs):
    cfg = kwargs['cfg']
    Epsilon = cfg.MFS_Appr.Epsilon
    dir_save_adv = cfg.MFS_Appr.save_dir

    ## MFS
    points_scence = example['points']

    # voxel to point
    points_innocent_v0 = voxel_to_point_numba(example['voxels'].cpu().detach().numpy(),
                                              example['num_points'].cpu().detach().numpy())
    points_adv = points_innocent_v0.copy()

    # Get gradient
    example['voxels'].requires_grad = True
    losses = model(example, return_loss=True, **kwargs, is_eval_after_attack=False)
    model.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        grad_voxel = \
            torch.autograd.grad(sum(losses['loss']), example['voxels'], retain_graph=False, create_graph=False)[0]
    grad_points = grad_voxel_to_grad_point_numba(grad_voxel.cpu().detach().numpy(),
                                                 example['num_points'].cpu().detach().numpy())
    # Add permutation
    permutation = Epsilon * np.sign(grad_points[:, :3])
    points_adv[:, :3] = points_adv[:, :3] + permutation

    # point to voxel
    voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(points_adv)
    save_to_example(example, voxels, coordinates, num_points, device)

    # save
    token = example['metadata'][0]['token']
    points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    # grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))
    # save combined pointclouds
    if "NuScenesDataset" in cfg.dataset_type:
        save_origin_points(cfg, example['points'][0].detach().cpu().numpy().reshape(-1, example['points'][0].shape[1]),
                           points_innocent_v0, points_adv,
                           token, dir_save_adv)

    with torch.no_grad():
        predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
    return predictions

def batch_processor(model, data, train_mode, **kwargs):
    if "local_rank" in kwargs:
        device = torch.device(kwargs["local_rank"])
    else:
        device = None

    # if not kwargs['cfg'].is_adv_eval:  # 如果是评测对抗样本，则在后续流程将data迁移到cuda上
    # data = example_convert_to_torch(data, device=device)
    example = example_to_device(data, device, non_blocking=False)
    token = data['metadata'][0]['token']
    del data

    if train_mode:
        losses = model(example, return_loss=True)
        loss, log_vars = parse_second_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(example["anchors"][0])
        )
        return outputs

    # adv -------------------------------------------
    cfg = kwargs['cfg']
    token = example['metadata'][0]['token']
    dir_save_adv = cfg.outputs_dir
    ##
    if cfg.dataset_type == "WaymoDataset":
        if 'collected_token' in cfg:
            if token not in cfg.collected_token:
                return None
            else:
                pass
                # 意外中断，恢复生成对抗样本

    if cfg.get('resume_generate_adv', False):
        points_adv_path = os.path.join(dir_save_adv, token + '.bin')
        if os.path.exists(points_adv_path):
            sz = os.path.getsize(points_adv_path)
            if not sz:
                ## 文件为空，重新生成
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                print(points_adv_path, " is empty! Re-generating!")
                print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
            else:
                cfg = kwargs['cfg']
                args = kwargs['args']
                # token = data['metadata'][0]['token']
                # token = kwargs['token']
                adv_eval_dir = cfg.outputs_dir
                is_adv_eval_entire_pc = cfg.get('is_adv_eval_entire_pc', False)
                if not is_adv_eval_entire_pc:
                    adv_eval_path = os.path.join(adv_eval_dir, str(token) + '.bin')
                else:
                    adv_eval_path = os.path.join(args.outputs_dir, str(token) + '-conbined_adv.bin')
                assert os.path.exists(adv_eval_path)
                # if not os.path.exists(adv_eval_path):
                #     print('=== None model output ===')
                #     return None
                points_adv = np.fromfile(adv_eval_path, dtype=np.float32).reshape(-1,
                                                                                  cfg.model.reader.num_input_features)
                voxels, coordinates, num_points = cfg.voxelization.voxel_generator.generate(points_adv)
                save_to_example(example, voxels, coordinates, num_points, device)

                # data['voxels'] = adv_sample.reshape(data['voxels'].shape)
                # data 迁移到cuda上
                # example = example_to_device(data, device, non_blocking=False)
                # del data
                with torch.no_grad():
                    predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
                return predictions
                # ## 文件存在而且有效，直接返回prediction结果
                # with torch.no_grad():
                #     predictions = model(example, return_loss=False, **kwargs, is_eval_after_attack=True)
                # return predictions
                # pass

    # 既使对抗样本存在，也重新生成
    else:
        pass

    # 测试
    if cfg.is_innocent_eval or cfg.is_adv_eval_waymo or cfg.is_adv_pred_by_innocent_pipeline:
        assert cfg.is_innocent_eval ^ cfg.is_adv_eval_waymo ^ cfg.is_adv_pred_by_innocent_pipeline

        # 测试re-Voxelize
        if cfg.is_test_reVoxelize:
            cfg.logger.info('=== test reVoxelize ===')
            predictions = test_reVoxelize(model, example, device, **kwargs)
            return predictions

        if cfg.is_innocent_eval or cfg.is_adv_pred_by_innocent_pipeline:
            with torch.no_grad():
                predictions = model(example, return_loss=False, **kwargs)
            return predictions
        elif cfg.is_adv_eval_waymo:
            print('=== Evaluating ===')
            predictions = adv_evaluation(model, example, device, token, **kwargs)
            # predictions = adv_evaluation(model=model, data=data, device=device, cfg=cfg, **kwargs)
            return predictions

    # 生成对抗样本
    if cfg.adv_flag:
        # assert sum([cfg.is_adv.is_FGSM, cfg.is_adv.is_PGD, cfg.is_adv.is_IOU]) == 1
        # FGSM
        if 'is_FGSM' in cfg.is_adv and cfg.is_adv.is_FGSM:
            cfg.logger.info('=== FGSM ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.FGSM)
            # predictions = FGSM_Attack_test_reVoxelize(model, example, device, **kwargs)
            predictions = FGSM_Attack(model, example, device, **kwargs)
            # predictions = FGSM_Attack_debug(model, example, device, **kwargs)
            return predictions
        # PGD
        elif 'is_PGD' in cfg.is_adv and cfg.is_adv.is_PGD:
            cfg.logger.info('=== PGD ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.PGD)
            predictions = PGD_Attack_v0(model, example, device, **kwargs)
            return predictions
        # IOU + score
        elif 'is_IOU' in cfg.is_adv and cfg.is_adv.is_IOU:
            cfg.logger.info('=== IOU Score ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.IOU)
            if 'light' in cfg.IOU.strategy:
                predictions = IOU_Attack_iter_light(model, example, device, **kwargs)
            else:
                predictions = IOU_Attack_iter(model, example, device, **kwargs)
            return predictions
        # distance + score
        elif 'is_DistScore' in cfg.is_adv and cfg.is_adv.is_DistScore:
            cfg.logger.info('=== Distance Score ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.DistScore)
            predictions = DistScore_Attack(model, example, device, **kwargs)
            return predictions
        elif 'is_FalsePositive' in cfg.is_adv and cfg.is_adv.is_FalsePositive:
            cfg.logger.info('=== False Positive ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.FalsePositive)
            predictions = FalsePositive_Attack(model, example, device, **kwargs)
            return predictions
        # PGD CoorAdjust
        elif 'is_PGD_CoorAdjust' in cfg.is_adv and cfg.is_adv.is_PGD_CoorAdjust:
            cfg.logger.info('=== PGD CoorAdjust ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.PGD_CoorAdjust)
            if 'light' in cfg.PGD_CoorAdjust.get('strategy', ''):
                cfg.logger.info("=== Basic Framework ===")
                predictions = PGD_CoorAdjust_Attack_light(model, example, device, **kwargs)
            else:
                predictions = PGD_CoorAdjust_Attack(model, example, device, **kwargs)
            return predictions
        # MI-FGSM
        elif 'is_MI_FGSM' in cfg.is_adv and cfg.is_adv.is_MI_FGSM:
            cfg.logger.info('=== MI_FGSM ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.MI_FGSM)
            if 'light' in cfg.MI_FGSM.get('strategy', ''):
                cfg.logger.info("=== Basic Framework ===")
                predictions = MI_FGSM_Attack_light(model, example, device, **kwargs)
            else:
                predictions = MI_FGSM_Attack(model, example, device, **kwargs)
            return predictions
            # PGD CoorAdjust
        elif 'is_PGD_CoorAdjust_light' in cfg.is_adv and cfg.is_adv.is_PGD_CoorAdjust_light:
            cfg.logger.info('=== PGD CoorAdjust light ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.PGD_CoorAdjust_light)
            predictions = PGD_CoorAdjust_Attack_light(model, example, device, **kwargs)
            return predictions
        elif 'is_CA' in cfg.is_adv and cfg.is_adv.is_CA:
            cfg.logger.info('=== CA ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.CA)
            predictions = CA_Attack(model, example, device, **kwargs)
            return predictions
        elif 'is_PointAttach' in cfg.is_adv and cfg.is_adv.is_PointAttach:
            cfg.logger.info('=== PointAttach ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.PointAttach)
            predictions = PointAttach_Attack(model, example, device, **kwargs)
            return predictions
        elif 'is_SpareAdv' in cfg.is_adv and cfg.is_adv.is_SpareAdv:
            cfg.logger.info('=== SpareAdv ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.SpareAdv)
            if 'MI' in cfg.SpareAdv.strategy or 'PGD' in cfg.SpareAdv.strategy:
                predictions = SpareAdv_MI_Attack(model, example, device, **kwargs)
            # elif 'PGD' in cfg.SpareAdv.strategy:
            #     predictions = SpareAdv_PGD_Attack(model, example, device, **kwargs)
            else:
                predictions = SpareAdv_Attack(model, example, device, **kwargs)
            return predictions
        elif 'is_AdaptiveEPS' in cfg.is_adv and cfg.is_adv.is_AdaptiveEPS:
            cfg.logger.info('=== AdaptiveEPS ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.AdaptiveEPS)
            if 'MI' in cfg.AdaptiveEPS.strategy or 'PGD' in cfg.AdaptiveEPS.strategy:
                if 'cuda' in cfg.AdaptiveEPS.strategy:
                    predictions = AdaptiveEPS_MI_Attack_cuda(model, example, device, **kwargs)
                else:
                    if 'light' in cfg.AdaptiveEPS.strategy:
                        cfg.logger.info("=== Basic Framework ===")
                        predictions = AdaptiveEPS_MI_Attack_light(model, example, device, **kwargs)
                    else:
                        predictions = AdaptiveEPS_MI_Attack(model, example, device, **kwargs)
            else:
                predictions = AdaptiveEPS_Attack(model, example, device, **kwargs)
            return predictions
        elif 'is_GSDA' in cfg.is_adv and cfg.is_adv.is_GSDA:
            cfg.logger.info('=== AdaptiveEPS ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.GSDA)
            predictions = GSDA_Attack(model, example, device, **kwargs)
            return predictions
        elif 'is_RoadAdv' in cfg.is_adv and cfg.is_adv.is_RoadAdv:
            cfg.logger.info('=== RoadAdv ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.RoadAdv)
            predictions = RoadAdv_Attack(model, example, device, **kwargs)
            return predictions
        elif 'is_IOU_CatAdv' in cfg.is_adv and cfg.is_adv.is_IOU_CatAdv:
            cfg.logger.info('=== IOU_CatAdv ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.IOU_CatAdv)
            predictions = IOU_CatAdv_Attack(model, example, device, **kwargs)
            return predictions
        elif 'is_pvcnn' in cfg.is_adv and cfg.is_adv.is_pvcnn:
            cfg.logger.info('=== pvcnn adv ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.pvcnn)
            predictions = pvcnn_Attack(model, example, device, **kwargs)
            return predictions
        elif 'is_MFS_Appr' in cfg.is_adv and cfg.is_adv.is_MFS_Appr:
            cfg.logger.info('=== MFS_Appr ===')
            cfg.logger.info('attack parameters:')
            cfg.logger.info(cfg.MFS_Appr)
            predictions = MFS_Appr_Attack(model, example, device, **kwargs)
            return predictions
    # adv -------------------------------------------
    else:
        return model(example, return_loss=False)  # adv


def batch_processor_ensemble(model1, model2, data, train_mode, **kwargs):
    assert 0, 'deprecated'
    if "local_rank" in kwargs:
        device = torch.device(kwargs["local_rank"])
    else:
        device = None

    assert train_mode is False

    example = example_to_device(data, device, non_blocking=False)
    del data

    preds_dicts1 = model1.pred_hm(example)
    preds_dicts2 = model2.pred_hm(example)

    num_task = len(preds_dicts1)

    merge_list = []

    # take the average
    for task_id in range(num_task):
        preds_dict1 = preds_dicts1[task_id]
        preds_dict2 = preds_dicts2[task_id]

        for key in preds_dict1.keys():
            preds_dict1[key] = (preds_dict1[key] + preds_dict2[key]) / 2

        merge_list.append(preds_dict1)

    # now get the final prediciton 
    return model1.pred_result(example, merge_list)


def flatten_model(m):
    return sum(map(flatten_model, m.children()), []) if len(list(m.children())) else [m]


def get_layer_groups(m):
    return [nn.Sequential(*flatten_model(m))]


def build_one_cycle_optimizer(model, optimizer_config):
    if optimizer_config.fixed_wd:
        optimizer_func = partial(
            torch.optim.Adam, betas=(0.9, 0.99), amsgrad=optimizer_config.amsgrad
        )
    else:
        optimizer_func = partial(torch.optim.Adam, amsgrad=optimizer_cfg.amsgrad)

    optimizer = OptimWrapper.create(
        optimizer_func,
        3e-3,  # TODO: CHECKING LR HERE !!!
        get_layer_groups(model),
        wd=optimizer_config.wd,
        true_wd=optimizer_config.fixed_wd,
        bn_wd=True,
    )

    return optimizer


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.
    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.
    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, "module"):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop("paramwise_options", None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        return obj_from_dict(
            optimizer_cfg, torch.optim, dict(params=model.parameters())
        )
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg["lr"]
        base_wd = optimizer_cfg.get("weight_decay", None)
        # weight_decay must be explicitly specified if mult is specified
        if (
                "bias_decay_mult" in paramwise_options
                or "norm_decay_mult" in paramwise_options
        ):
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get("bias_lr_mult", 1.0)
        bias_decay_mult = paramwise_options.get("bias_decay_mult", 1.0)
        norm_decay_mult = paramwise_options.get("norm_decay_mult", 1.0)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            param_group = {"params": [param]}
            if not param.requires_grad:
                # FP16 training needs to copy gradient/weight between master
                # weight copy and model weight, it is convenient to keep all
                # parameters here to align with model.parameters()
                params.append(param_group)
                continue

            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r"(bn|gn)(\d+)?.(weight|bias)", name):
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith(".bias"):
                param_group["lr"] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * bias_decay_mult
            # otherwise use the global settings

            params.append(param_group)

        optimizer_cls = getattr(torch.optim, optimizer_cfg.pop("type"))
        return optimizer_cls(params, **optimizer_cfg)


def train_detector(model, dataset, cfg, distributed=False, validate=False, logger=None, **kwargs):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, dist=distributed
        )
        for ds in dataset
    ]

    total_steps = cfg.total_epochs * len(data_loaders[0])
    # print(f"total_steps: {total_steps}")
    if distributed:
        model = apex.parallel.convert_syncbn_model(model)
    if cfg.lr_config.type == "one_cycle":
        # build trainer
        optimizer = build_one_cycle_optimizer(model, cfg.optimizer)
        lr_scheduler = _create_learning_rate_scheduler(
            optimizer, cfg.lr_config, total_steps
        )
        cfg.lr_config = None
    else:
        optimizer = build_optimizer(model, cfg.optimizer)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.drop_step, gamma=.1)
        # lr_scheduler = None
        cfg.lr_config = None

        # put model on gpus
    if distributed:
        model = DistributedDataParallel(
            model.cuda(cfg.local_rank),
            device_ids=[cfg.local_rank],
            output_device=cfg.local_rank,
            # broadcast_buffers=False,
            find_unused_parameters=True,
        )
    else:
        model = model.cuda()

    logger.info(f"model structure: {model}")

    trainer = Trainer(
        model, batch_processor, optimizer, lr_scheduler, cfg.work_dir, cfg.log_level
    )

    if distributed:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    trainer.register_training_hooks(
        cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config
    )

    if distributed:
        trainer.register_hook(DistSamplerSeedHook())

    # # register eval hooks
    # if validate:
    #     val_dataset_cfg = cfg.data.val
    #     eval_cfg = cfg.get('evaluation', {})
    #     dataset_type = DATASETS.get(val_dataset_cfg.type)
    #     trainer.register_hook(
    #         KittiEvalmAPHookV2(val_dataset_cfg, **eval_cfg))

    if cfg.resume_from:
        trainer.resume(cfg.resume_from)
    elif cfg.load_from:
        trainer.load_checkpoint(cfg.load_from)

    trainer.run(data_loaders, cfg.workflow, cfg.total_epochs, local_rank=cfg.local_rank, dataset=dataset[0], **kwargs)
