'''
测试
'''
is_innocent_eval = False  # 测试干净样本
is_eval_after_attack = False
is_adv_eval = False  # 测试对抗样本
is_adv_eval_entire_pc = False
adv_eval_dir = r''
# is_point_interpolate = False
is_evaluate_ASR = False

# 生成对抗样本
IS_ADV = True
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
)
save_dir = r'/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/FGSM'

voxel_generator = dict(
    range=[0, -39.68, -3, 69.12, 39.68, 1],
    voxel_size=[0.16, 0.16, 4],
    max_points_in_voxel=32,
    max_voxel_num=[40000, 40000],
)