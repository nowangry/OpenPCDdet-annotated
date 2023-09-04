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
adv_flag = True


AdaptiveEPS = dict(
    eps=0.5,
    # eps_ratio=0,  # 动态变化的扰动量；最多修改eps的一半
    num_steps=10,
    # strategy='base',
    # strategy='MI',
    # strategy='MI-filterOnce',
    # strategy='PGD',
    # strategy='PGD-filterOnce',
    # strategy='PGD-filterOnce',  # 固定扰动量
    strategy='light-PGD-filterOnce',  # 固定扰动量
    decay=1.0,
    fixedEPS=0.3,
    attach_rate=0.2,  # 每次迭代修改多少比例的点云
)
save_dir = r'/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/AdaptiveEPS'
# adv ------------------------------------------------------------------------------------------------------------------

voxel_generator = dict(
    range=[0, -39.68, -3, 69.12, 39.68, 1],
    voxel_size=[0.16, 0.16, 4],
    max_points_in_voxel=32,
    max_voxel_num=[40000, 40000],
)