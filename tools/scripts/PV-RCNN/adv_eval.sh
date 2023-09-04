
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 2 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-FGSM.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=2 python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 2 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-PGD.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=2 python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 2 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-PGD_light.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=2 python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 2 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM.py --evaluate_adv
#CUDA_VISIBLE_DEVICES=2 python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 2 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM_light.py --evaluate_adv

## transfer attack
CUDA_VISIBLE_DEVICES=0 python test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --transfer_adv
