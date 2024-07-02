## transfer attack
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/kitti_models/voxel_rcnn_car.yaml --ckpt ../checkpoints/voxel_rcnn_car_84.54.pth --batch_size 1 --transfer_adv
