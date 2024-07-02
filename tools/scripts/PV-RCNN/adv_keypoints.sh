### keypoints

## FGSM
#CUDA_VISIBLE_DEVICES=0 python test.py --strategy keypoints --subsample_num 323 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-FGSM.py
#CUDA_VISIBLE_DEVICES=0 python test.py --strategy keypoints --subsample_num 323 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-FGSM.py  --evaluate_adv
#CUDA_VISIBLE_DEVICES=0 python test.py --strategy keypoints --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-FGSM.py
#CUDA_VISIBLE_DEVICES=0 python test.py --strategy keypoints --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-FGSM.py  --evaluate_adv

## PGD
#CUDA_VISIBLE_DEVICES=0 python test.py --strategy keypoints --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-PGD.py
#CUDA_VISIBLE_DEVICES=0 python test.py --strategy keypoints --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-PGD.py --evaluate_adv

# MI_FGSM
#CUDA_VISIBLE_DEVICES=1 python test.py --strategy keypoints --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM.py
CUDA_VISIBLE_DEVICES=1 python test.py --strategy keypoints --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM.py --evaluate_adv

## IOU
#CUDA_VISIBLE_DEVICES=3 python test.py --strategy keypoints --eps 0.20 --eps_iter 0.2 --num_steps 1 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU.py
CUDA_VISIBLE_DEVICES=3 python test.py --strategy keypoints --eps 0.20 --eps_iter 0.2 --num_steps 1 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU.py --evaluate_adv

## GSVA
#CUDA_VISIBLE_DEVICES=2 python test.py --strategy keypoints-PGD-filterOnce --fixedEPS 0.3 --attach_rate 0.2 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-AdaptiveEPS.py
CUDA_VISIBLE_DEVICES=2 python test.py --strategy keypoints-PGD-filterOnce --fixedEPS 0.3 --attach_rate 0.2 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-AdaptiveEPS.py --evaluate_adv


#CUDA_VISIBLE_DEVICES=2 python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-PGD_light.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=2 python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=2 python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM_light.py --evaluate_adv
