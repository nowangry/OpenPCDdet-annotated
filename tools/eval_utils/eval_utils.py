import pickle
import time

import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import os


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def save_origin_points(cfg, points, points_innocent_ori, points_adv, token, dir_save_adv):
    voxel_range = cfg.voxel_generator.range
    points_out_of_range = []
    for i in range(points.shape[0]):
        if point_in_range(points[i, :], voxel_range):
            continue
        else:
            points_out_of_range.append(points[i, :])
    points_conbined_innocent = np.concatenate([points_out_of_range, points_innocent_ori], axis=0)
    points_conbined_adv = np.concatenate([points_out_of_range, points_adv], axis=0)
    points_conbined_innocent.tofile(os.path.join(dir_save_adv, token + '-conbined_innocent_ori.bin'))
    points_conbined_adv.tofile(os.path.join(dir_save_adv, token + '-conbined_adv.bin'))

def FGSM_Attack(model, batch_dict):
    cfg = batch_dict['cfg']
    Epsilon = 0.2
    # dir_save_adv = cfg.FGSM.save_dir

    if 'is_point_adv' in cfg:
        # voxel to point
        points_innocent_v0 = batch_dict['points']
        points_adv = points_innocent_v0.clone()

        # Get gradient
        batch_dict['points'].requires_grad = True
        loss = model(batch_dict, return_loss=True, is_eval_after_attack=False)

        model.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            grad_points = \
                torch.autograd.grad(loss, batch_dict['points'], retain_graph=False, create_graph=False)[0]
            print('grad_points.shape: {}'.format(grad_points.shape))

        # Add permutation
        permutation = Epsilon * torch.sign(grad_points[:, 1:4])
        points_adv[:, 1:4] = points_adv[:, 1:4] + permutation

    else:
        # voxel to point
        voxels_innocent_v0 = batch_dict['voxels']
        voxels_adv = voxels_innocent_v0.clone()

        # Get gradient
        batch_dict['voxels'].requires_grad = True
        loss = model(batch_dict, return_loss=True, is_eval_after_attack=False)

        model.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            # grad_points = \
            #     torch.autograd.grad(loss, batch_dict['points'], retain_graph=False, create_graph=False)[0]
            grad_voxels = \
                torch.autograd.grad(loss, batch_dict['voxels'], retain_graph=False, create_graph=False)[0]
            print('grad_voxels.shape: {}'.format(grad_voxels.shape))

        # Add permutation
        permutation = Epsilon * torch.sign(grad_voxels[:, 1:4])
        voxels_adv[:, 1:4] = grad_voxels[:, 1:4] + permutation

    # point to voxel
    # save
    # token = batch_dict['frame_id']
    # points_adv.tofile(os.path.join(dir_save_adv, token + '.bin'))
    # points_innocent_v0.tofile(os.path.join(dir_save_adv, token + '-innocent.bin'))
    # grad_points.tofile(os.path.join(dir_save_adv, token + '-innocent_gradient.bin'))

    # return pred_dicts, batch_dict

from det3d.torchie.utils.config import Config
def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, **kwargs):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    # model.eval()
    model.train()  # adv
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        # Adv -----------------------------------------------------------------------
        # config_file = r'./cfgs/adv/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z-adv-FGSM_CoorAdjust.py'
        # cfg = Config.fromfile(config_file)
        # cfg = kwargs['cfg']
        batch_dict['cfg'] = cfg
        FGSM_Attack(model, batch_dict)

        # Adv -----------------------------------------------------------------------
        # torch.autograd.set_detect_anomaly(True)
        # with torch.set_grad_enabled(cfg.IS_ADV):
        #     load_data_to_gpu(batch_dict)
        #     batch_dict['points'].requires_grad = True
        #     pred_dicts, ret_dict = model(batch_dict)
        #     model.zero_grad()

            # # RPN loss for PointRCNN
            # loss_rpn_cls = batch_dict['batch_cls_preds'].sum()
            # loss_rpn_cls.backward()
            # print(batch_dict['points'].grad)
            # grad = torch.autograd.grad(loss_rpn_cls, batch_dict['points'], retain_graph=False, create_graph=False)[0]
            # print(grad)

            # grad = torch.autograd.grad(loss_rpn_cls, batch_dict['points'], retain_graph=False, create_graph=False)[0]

            # # RCNN loss for PointRCNN
            # loss_rcnn_cls = batch_dict['rcnn_cls'].sum()
            # loss_rcnn_cls.backward()
            # loss_adv = loss_rpn_cls + loss_rcnn_cls
            # loss_adv.backward()

            # # loss for PointPillar
            # loss_adv_pred= pred_dicts[0]['pred_scores'].sum()
            # loss_adv_pred.backward()


            # loss for centerPoint



            # inputs = batch_dict['points'].detach()
            #
            # Epsilon = 0.1
            # adv_inputs = inputs + Epsilon * inputs.grad.sign()
            #
            # benign_pc_dir = r'/data/dataset_wujunqi/KITTI/object/training/velodyne'
            # adv_pc_dir = r'/data/dataset_wujunqi/KITTI/object/training/velodyne-adv'
            #
            # for root, _, file_names in os.walk(benign_pc_dir):
            #     pass
            # adv_pc_path = os.path.join(adv_pc_dir, file_names[i] + '.bin')
            # adv_inputs.detach().cpu().numpy().tofile(adv_pc_path)  # 保存对抗点云

        # Adv -----------------------------------------------------------------------


        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        disp_dict = {}

        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
