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


IOU = dict(
    eps=0.20,
    eps_iter=0.2,
    num_steps=1,
    Lambda=0.1,
    iou_thre=0.1,
    score_thre=0.1,
    strategy='',
    # strategy='light',
)

save_dir = r'/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/IOU'
get_refined_box_when_training = True
# adv ------------------------------------------------------------------------------------------------------------------


voxel_generator = dict(
    range=[0, -39.68, -3, 69.12, 39.68, 1],
    voxel_size=[0.16, 0.16, 4],
    max_points_in_voxel=32,
    max_voxel_num=[40000, 40000],
)