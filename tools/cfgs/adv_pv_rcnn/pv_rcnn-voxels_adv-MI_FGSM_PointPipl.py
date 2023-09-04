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


MI_FGSM = dict(
    eps=0.2,
    eps_iter=0.03,
    num_steps=10,
    decay=1.0,
    # L_norm='L1',
    L_norm='L2',
    # strategy='',
    strategy='PointPipl',
)
save_dir = r'/data/dataset_wujunqi/Outputs/GSVA/KITTI/PV_RCNN-adv/MI_FGSM'
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