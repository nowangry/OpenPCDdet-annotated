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


PGD = dict(
    random_start=True,
    # random_start=True,
    eps=0.20,
    eps_iter=0.03,
    num_steps=10,
    # strategy='light',
)
save_dir = r'/data/dataset_wujunqi/Outputs/GSVA/KITTI/PointPillar-adv/PGD'
# adv ------------------------------------------------------------------------------------------------------------------


voxel_generator = dict(
    range=[0, -39.68, -3, 69.12, 39.68, 1],
    voxel_size=[0.16, 0.16, 4],
    max_points_in_voxel=32,
    max_voxel_num=[40000, 40000],
)
#
# DATA_CONFIG:
#     _BASE_CONFIG_: cfgs/dataset_configs/kitti_dataset.yaml
#     POINT_CLOUD_RANGE: [0, -39.68, -3, 69.12, 39.68, 1]
#     DATA_PROCESSOR:
#         - NAME: mask_points_and_boxes_outside_range
#           REMOVE_OUTSIDE_BOXES: True
#
#         - NAME: shuffle_points
#           SHUFFLE_ENABLED: {
#             'train': True,
#             'test': False
#           }
#
#         - NAME: transform_points_to_voxels
#           VOXEL_SIZE: [0.16, 0.16, 4]
#           MAX_POINTS_PER_VOXEL: 32
#           MAX_NUMBER_OF_VOXELS: {
#             'train': 16000,
#             'test': 40000
#           }
#     DATA_AUGMENTOR:
#         DISABLE_AUG_LIST: ['placeholder']
#         AUG_CONFIG_LIST:
#             - NAME: gt_sampling
#               USE_ROAD_PLANE: False
#               DB_INFO_PATH:
#                   - kitti_dbinfos_train.pkl
#               PREPARE: {
#                  filter_by_min_points: ['Car:5', 'Pedestrian:5', 'Cyclist:5'],
#                  filter_by_difficulty: [-1],
#               }
#
#               SAMPLE_GROUPS: ['Car:15','Pedestrian:15', 'Cyclist:15']
#               NUM_POINT_FEATURES: 4
#               DATABASE_WITH_FAKELIDAR: False
#               REMOVE_EXTRA_WIDTH: [0.0, 0.0, 0.0]
#               LIMIT_WHOLE_SCENE: False
#
#             - NAME: random_world_flip
#               ALONG_AXIS_LIST: ['x']
#
#             - NAME: random_world_rotation
#               WORLD_ROT_ANGLE: [-0.78539816, 0.78539816]
#
#             - NAME: random_world_scaling
#               WORLD_SCALE_RANGE: [0.95, 1.05]
