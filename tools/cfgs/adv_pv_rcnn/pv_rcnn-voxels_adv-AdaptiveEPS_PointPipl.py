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
    strategy='PointPipl-PGD-filterOnce',  # 固定扰动量
    # strategy='light-PGD-filterOnce',  # 固定扰动量
    decay=1.0,
    fixedEPS=0.3,
    attach_rate=0.2,  # 每次迭代修改多少比例的点云
)
save_dir = r'/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/AdaptiveEPS'
# adv ------------------------------------------------------------------------------------------------------------------


voxel_generator = dict(
    range=[0, -40, -3, 70.4, 40, 1],
    voxel_size=[0.05, 0.05, 0.1],
    max_points_in_voxel=5,
    max_voxel_num=[40000, 40000],
)
# POINT_CLOUD_RANGE: [0, -40, -3, 70.4, 40, 1]
# voxel_generator = dict(
#     range=[-54, -54, -5.0, 54, 54, 3.0],
#     voxel_size=[0.075, 0.075, 0.2],
#     max_points_in_voxel=10,
#     max_voxel_num=[120000, 160000],
# )