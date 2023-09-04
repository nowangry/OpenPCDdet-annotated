#
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-FGSM.py
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-PGD.py
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-PGD_light.py

#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-MI_FGSM.py
#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-MI_FGSM_light.py
#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-IOU.py
#

# 评测
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-FGSM.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-PGD.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-PGD_light.py --evaluate_adv

#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-MI_FGSM.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-MI_FGSM_light.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-IOU.py --evaluate_adv



# GSVA
#CUDA_VISIBLE_DEVICES=2 python ./test.py --fixedEPS 0.5 --attach_rate 0.3 --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-AdaptiveEPS.py
#CUDA_VISIBLE_DEVICES=2 python ./test.py --fixedEPS 0.5 --attach_rate 0.3 --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-AdaptiveEPS.py --evaluate_adv


#CUDA_VISIBLE_DEVICES=3 python ./test.py --fixedEPS 0.5 --attach_rate 0.5 --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-AdaptiveEPS_light.py
#CUDA_VISIBLE_DEVICES=3 python ./test.py --fixedEPS 0.5 --attach_rate 0.5 --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-AdaptiveEPS_light.py --evaluate_adv


## IOU ADV
CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-IOU.py --num_steps 10 --eps 0.2 --eps_iter 0.03
CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pointpillar.yaml --ckpt ../checkpoints/pointpillar_7728.pth --batch_size 1 --cfg_pyfile cfgs/adv_pointpillar/pointpillar-adv-IOU.py --num_steps 10 --eps 0.2 --eps_iter 0.03 --evaluate_adv
