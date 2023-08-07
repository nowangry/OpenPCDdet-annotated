import itertools
import logging
import os
# from det3d.utils.config_tool import get_downsample_factor

# adv ---------------------------------------
# python dist_test.py ../configs/adv/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z_adv_eval.py --work_dir ../work_dirs/ADV/PGD_CoorAdjust/eps_0.10-eps_iter_0.015-num_steps_20  --checkpoint  ../work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/centerpoint_voxel_1440-epoch_20.pth --speed_test
# python dist_test.py ../configs/adv/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z-adv-PGD_CoorAdjust.py --work_dir ../work_dirs/ADV/PGD_CoorAdjust/eps_0.80-eps_iter_0.06-num_steps_20  --checkpoint  ../work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/centerpoint_voxel_1440-epoch_20.pth --speed_test
# python dist_test.py ../configs/adv/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z-adv-PGD_CoorAdjust.py --work_dir ../work_dirs/ADV/PGD_CoorAdjust/eps_1.60-eps_iter_0.12-num_steps_20  --checkpoint  ../work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/centerpoint_voxel_1440-epoch_20.pth --speed_test
# python dist_test.py ../configs/adv/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z-adv-FGSM_CoorAdjust.py  --checkpoint  ../work_dirs/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z/centerpoint_voxel_1440-epoch_20.pth --speed_test

# 测试
is_innocent_eval = False
is_adv_eval = False
adv_eval_dir = r''
is_point_interpolate = False
is_evaluate_ASR = False

# 生成对抗样本
is_adv = dict(
    is_FGSM=True,
    is_PGD=False,
    is_PGD_CoorAdjust=False,
    is_IOU=False,
    is_DistScore=False,
    is_FalsePositive=False,
)
adv_flag = True
# FGSM_CoorAdjust
FGSM = dict(
    Epsilon=0.2,
    save_dir=r'/data/dataset_wujunqi/Outputs/PointRCNN/FGSM'
)
# catAdv_dict = dict(
#     swap_pred_cat=False,
#     swap_gt_cat=True,
# )
# adv ---------------------------------------
