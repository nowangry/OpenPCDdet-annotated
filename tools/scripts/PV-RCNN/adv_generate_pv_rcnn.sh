#
#CUDA_VISIBLE_DEVICES=2 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM.py
#
#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM_light.py
#
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-PGD_light.py


#CUDA_VISIBLE_DEVICES=0 python test.py --strategy PGD-filterOnce --fixedEPS 0.3 --attach_rate 0.2 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-AdaptiveEPS.py
#CUDA_VISIBLE_DEVICES=0 python test.py --strategy PGD-filterOnce --fixedEPS 0.3 --attach_rate 0.2 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-AdaptiveEPS.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=0 python test.py --strategy light-PGD-filterOnce --fixedEPS 0.3 --attach_rate 0.2 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-AdaptiveEPS_light.py
#CUDA_VISIBLE_DEVICES=0 python test.py --strategy light-PGD-filterOnce --fixedEPS 0.3 --attach_rate 0.2 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-AdaptiveEPS_light.py --evaluate_adv


## IOU-ADV
#CUDA_VISIBLE_DEVICES=2 python test.py --subsample_num 323 --eps 0.20 --eps_iter 0.2 --num_steps 1 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU.py
#CUDA_VISIBLE_DEVICES=2 python test.py --subsample_num 323 --eps 0.20 --eps_iter 0.2 --num_steps 1 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 2 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU.py --evaluate_adv

#CUDA_VISIBLE_DEVICES=1 python test.py --subsample_num 323 --eps 0.20 --eps_iter 0.03 --num_steps 10 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU.py
#CUDA_VISIBLE_DEVICES=1 python test.py --subsample_num 323 --eps 0.20 --eps_iter 0.03 --num_steps 10 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU.py --evaluate_adv

CUDA_VISIBLE_DEVICES=2 python test.py --eps 0.20 --eps_iter 0.2 --num_steps 1 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU.py
CUDA_VISIBLE_DEVICES=2 python test.py --eps 0.20 --eps_iter 0.2 --num_steps 1 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 2 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU.py --evaluate_adv
