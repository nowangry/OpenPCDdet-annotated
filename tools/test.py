import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    ## adv
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--cfg_pyfile', type=str, default=None, help='specify the .py config for training')
    parser.add_argument('--is_torch_adv', action='store_true', default=False)

    # adv eval
    parser.add_argument('--evaluate_adv', action='store_true', default=False, help='whether to evaluate ADV examples')
    parser.add_argument('--transfer_adv', action='store_true', default=False, help='whether to perform transfer attack')

    ## AdaptiveEPS
    parser.add_argument('--subsample_num', type=int, default=None, help='specify subsample number for AdaptiveEPS')
    parser.add_argument('--fixedEPS', type=float, default=None, help='specify fixedEPS for AdaptiveEPS')
    parser.add_argument('--attach_rate', type=float, default=None, help='specify attach_rate for AdaptiveEPS')
    parser.add_argument('--strategy', type=str, default=None, help='specify strategy for AdaptiveEPS')

    ## IOU ADV
    parser.add_argument('--num_steps', type=int, default=None, help='specify num_steps for IOU-ADV')
    parser.add_argument('--eps', type=float, default=None, help='specify eps for IOU-ADV')
    parser.add_argument('--eps_iter', type=float, default=None, help='specify eps_iter for IOU-ADV')

    ## MI-FGSM
    parser.add_argument('--L_norm', type=str, default=None, help='specify L_norm for MI-FGSM')


    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    ## adv
    if args.cfg_pyfile is not None:
        # 结合.py和.yaml的配置信息更新config
        from det3d.torchie.utils.config import Config
        pycfg = Config.fromfile(args.cfg_pyfile)
        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.update(pycfg)

        if 'FGSM' in cfg:
            if cfg.FGSM.get('strategy', ''):
                folder = 'strategy_{}-Epsilon_{}'.format(cfg.FGSM.strategy, cfg.FGSM.Epsilon)
            else:
                folder = 'Epsilon_{}'.format(cfg.FGSM.Epsilon)
            cfg.exp_name = 'FGSM-' + folder
            cfg.save_dir = os.path.join(cfg.save_dir, folder)
            cfg.adv_alg = 'FGSM'

        elif 'PGD' in cfg:
            folder = '{}eps_{}-eps_iter_{}-num_steps_{}{}{}'.format(
                'stragety_{}-'.format(cfg.PGD.get('strategy', False)) if cfg.PGD.get('strategy', False) else '',
                cfg.PGD.eps, cfg.PGD.eps_iter,
                cfg.PGD.num_steps,
                '-randStart' if cfg.PGD.random_start else '',
                '-n_{}-'.format(cfg.PGD.get('subsample_num', False)) if cfg.PGD.get('subsample_num', False) else '',
            )
            cfg.exp_name = 'PGD-' + folder
            cfg.save_dir = os.path.join(cfg.save_dir, folder)
            cfg.adv_alg = 'PGD'

        # IOU + score
        elif 'IOU' in cfg:
            if args.num_steps is not None:
                cfg.IOU.num_steps = args.num_steps
            if args.eps is not None:
                cfg.IOU.eps = args.eps
            if args.eps_iter is not None:
                cfg.IOU.eps_iter = args.eps_iter
            if args.subsample_num is not None:
                cfg.IOU.subsample_num = args.subsample_num
            folder = '{}eps_{}-eps_iter_{}-num_steps_{}-Lambda_{}-iou_{}-score_{}{}'.format(
                'stragety_{}-'.format(cfg.IOU.get('strategy', False)) if cfg.IOU.get('strategy', False) else '',
                cfg.IOU.eps, cfg.IOU.eps_iter,
                cfg.IOU.num_steps,
                cfg.IOU.Lambda, cfg.IOU.iou_thre,
                cfg.IOU.score_thre,
                '-n_{}-'.format(cfg.IOU.get('subsample_num', False)) if cfg.IOU.get('subsample_num', False) else '',
                )
            cfg.exp_name = 'IOU-' + folder
            cfg.save_dir = os.path.join(cfg.save_dir, folder)
            cfg.adv_alg = 'IOU'

        elif 'MI_FGSM' in cfg:
            if args.subsample_num is not None:
                cfg.MI_FGSM.subsample_num = args.subsample_num
            if args.L_norm is not None:
                cfg.MI_FGSM.L_norm = args.L_norm
            cfg_adv = cfg.MI_FGSM
            folder = '{}eps_{}-eps_iter_{}-num_steps_{}-decay_{}-L_norm_{}{}'.format(
                'stragety_{}-'.format(cfg_adv.get('strategy', False)) if cfg_adv.get('strategy', False) else '',
                cfg_adv.eps, cfg_adv.eps_iter,
                cfg_adv.num_steps,
                cfg_adv.decay,
                cfg_adv.L_norm,
                '-n_{}-'.format(cfg_adv.get('subsample_num', False)) if cfg_adv.get('subsample_num', False) else '',
            )
            cfg.exp_name = 'MI_FGSM-' + folder
            cfg.save_dir = os.path.join(cfg.save_dir, folder)
            cfg.adv_alg = 'MI_FGSM'

        elif 'VMI_FGSM' in cfg:
            if args.subsample_num is not None:
                cfg.VMI_FGSM.subsample_num = args.subsample_num
            cfg_adv = cfg.VMI_FGSM
            folder = '{}eps_{}-eps_iter_{}-num_steps_{}-decay_{}-beta_{}-N_{}{}'.format(
                'stragety_{}-'.format(cfg_adv.get('strategy', False)) if cfg_adv.get('strategy', False) else '',
                cfg_adv.eps,
                cfg_adv.eps_iter,
                cfg_adv.num_steps,
                cfg_adv.decay,
                cfg_adv.beta,
                cfg_adv.N,
                '-n_{}'.format(cfg_adv.get('subsample_num', False)) if cfg_adv.get('subsample_num', False) else '',
            )
            cfg.exp_name = 'VMI_FGSM-' + folder
            cfg.save_dir = os.path.join(cfg.save_dir, folder)
            cfg.adv_alg = 'VMI_FGSM'

        elif 'AdaptiveEPS' in cfg:
            if args.fixedEPS is not None:
                cfg.AdaptiveEPS.fixedEPS = args.fixedEPS
            if args.attach_rate is not None:
                cfg.AdaptiveEPS.attach_rate = args.attach_rate
            if args.strategy is not None:
                cfg.AdaptiveEPS.strategy = args.strategy
            if args.subsample_num is not None:
                cfg.AdaptiveEPS.subsample_num = args.subsample_num
            folder = 'strategy_{}-eps_{}-num_steps_{}-fixedEPS_{}-attach_rate_{}{}'.format(
                cfg.AdaptiveEPS.strategy,
                cfg.AdaptiveEPS.eps,
                cfg.AdaptiveEPS.num_steps,
                cfg.AdaptiveEPS.fixedEPS,
                cfg.AdaptiveEPS.attach_rate,
                '-n_{}-'.format(cfg.AdaptiveEPS.get('subsample_num', False)) if cfg.AdaptiveEPS.get('subsample_num', False) else '',
              )
            cfg.exp_name = 'AdaptiveEPS-' + folder
            cfg.save_dir = os.path.join(cfg.save_dir, folder)
            cfg.adv_alg = 'AdaptiveEPS'
        elif 'is_reVoxelization' in cfg:
            folder = 'reVoxelization'
            cfg.exp_name = 'reVoxelization'
            cfg.save_dir = os.path.join(cfg.save_dir, folder)
            cfg.adv_alg = 'reVoxelization'


        args.eval_tag = Path(cfg.adv_alg) / cfg.exp_name
        cfg.dataset_type = 'KITTI'
        ## -------------------------------------------------------------------------------------------------------------
        # adv Voxelization
        from det3d.datasets.pipelines.preprocess import Voxelization
        from det3d.torchie.utils.config import ConfigDict
        voxelization_cfg = ConfigDict()
        voxelization_cfg.cfg = cfg.voxel_generator
        cfg.voxelization = Voxelization(**voxelization_cfg)

    else:
        cfg_from_yaml_file(args.cfg_file, cfg)

    # 更新
    cfg.is_adv_eval = args.evaluate_adv
    cfg.transfer_adv = args.transfer_adv

    if args.subsample_num is not None:
        cfg.subsample_num = args.subsample_num

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file,
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def transfer_main(args, cfg):
    if 'pv_rcnn' in args.cfg_file:
        cfg.transfer_attack_dirs = {
            'FGSM': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/FGSM/Epsilon_0.2',
            'PGD': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/PGD/eps_0.2-eps_iter_0.03-num_steps_10-randStart',
            'PGD_light': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/PGD/stragety_light-eps_0.2-eps_iter_0.03-num_steps_10-randStart',
            'MI_FGSM': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/MI_FGSM/eps_0.2-eps_iter_0.03-num_steps_10-decay_1.0-L_norm_L2',
            'MI_FGSM_light': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/MI_FGSM/stragety_light-eps_0.2-eps_iter_0.03-num_steps_10-decay_1.0-L_norm_L2',
            'IOU': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/IOU/eps_0.2-eps_iter_0.2-num_steps_1-Lambda_0.1-iou_0.1-score_0.1',
            'AdaptiveEPS': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/AdaptiveEPS/strategy_PGD-filterOnce-eps_0.5-num_steps_10-fixedEPS_0.5-attach_rate_0.3',
            'AdaptiveEPS_light': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/AdaptiveEPS/strategy_light-PGD-filterOnce-eps_0.5-num_steps_10-fixedEPS_0.5-attach_rate_0.5',
        }
    elif 'pointpillar' in args.cfg_file:
        cfg.transfer_attack_dirs = {
            'FGSM': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/FGSM/Epsilon_0.2',
            'PGD': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/PGD/eps_0.2-eps_iter_0.03-num_steps_10-randStart',
            'PGD_light': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/PGD/stragety_light-eps_0.2-eps_iter_0.03-num_steps_10-randStart',
            'MI_FGSM': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/MI_FGSM/eps_0.2-eps_iter_0.03-num_steps_10-decay_1.0-L_norm_L2',
            'MI_FGSM_light': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/MI_FGSM/stragety_light-eps_0.2-eps_iter_0.03-num_steps_10-decay_1.0-L_norm_L2',
            'IOU': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/IOU/eps_0.2-eps_iter_0.2-num_steps_1-Lambda_0.1-iou_0.1-score_0.1',
            'AdaptiveEPS': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/AdaptiveEPS/strategy_PGD-filterOnce-eps_0.5-num_steps_10-fixedEPS_0.3-attach_rate_0.2',
            'AdaptiveEPS_light': '/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/AdaptiveEPS/strategy_light-PGD-filterOnce-eps_0.5-num_steps_10-fixedEPS_0.3-attach_rate_0.2',
        }

    for key_adv in cfg.transfer_attack_dirs:
        cfg.transfer_attack_dir = cfg.transfer_attack_dirs[key_adv]

        if args.launcher == 'none':
            dist_test = False
            total_gpus = 1
        else:
            total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
                args.tcp_port, args.local_rank, backend='nccl'
            )
            dist_test = True

        if args.batch_size is None:
            args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        else:
            assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
            args.batch_size = args.batch_size // total_gpus

        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
        output_dir.mkdir(parents=True, exist_ok=True)

        eval_output_dir = output_dir / 'eval'

        if not args.eval_all:
            num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
            epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
            eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        else:
            eval_output_dir = eval_output_dir / 'eval_all_default'

        if args.eval_tag is not None:
            eval_output_dir = eval_output_dir / args.eval_tag

        # log to file
        eval_output_dir.mkdir(parents=True, exist_ok=True)
        if cfg.transfer_attack_dir:
            log_file = eval_output_dir / ('evaluation_transfer_attack_{}.txt'.format(key_adv))
            logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
            cfg.logger = logger
            cfg.logger.info("cfg.transfer_attack_dir: {}".format(cfg.transfer_attack_dir))

        ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False,
            cfg=cfg ##
        )

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
        # with torch.no_grad():
        with torch.set_grad_enabled('IS_ADV' in cfg and cfg.IS_ADV and not cfg.is_adv_eval and not cfg.transfer_adv):
            if args.eval_all:
                repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
            else:
                eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)



def main():
    args, cfg = parse_config()
    if cfg.transfer_adv:
        transfer_main(args, cfg)
        return

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    # log to file
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.is_adv_eval:
        log_file = eval_output_dir / ('evaluation_results_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
        cfg.logger = logger
    else:
        log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
        cfg.logger = logger
        logger.info('**********************Start logging**********************')

        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        if dist_test:
            logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))
        log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False,
        cfg=cfg ##
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    # with torch.no_grad():
    with torch.set_grad_enabled('IS_ADV' in cfg and cfg.IS_ADV and not cfg.is_adv_eval):
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)

    if cfg.get('IS_ADV', False) and cfg.get('is_adv_eval', False) and cfg.get('is_reVoxelization', False):
        if cfg.save_dir:
            cfg.dataset_type = 'KITTI'
            from tools.analysis.stat_permutation import stat_L_permutations_all_logger
            stat_L_permutations_all_logger(pc_dir_list=[cfg.save_dir + os.sep], cfg=cfg)
        print('######################################################')


if __name__ == '__main__':
    main()
