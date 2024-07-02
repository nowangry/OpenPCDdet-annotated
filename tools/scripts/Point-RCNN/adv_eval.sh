## transfer attack
CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/kitti_models/pointrcnn.yaml --ckpt ../checkpoints/pointrcnn_7870.pth --batch_size 1 --transfer_adv
