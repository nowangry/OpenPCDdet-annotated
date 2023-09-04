#
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-FGSM_PointPipl.py
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-FGSM_PointPipl.py --evaluate_adv

#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-PGD_PointPipl.py
#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-PGD_PointPipl.py --evaluate_adv

#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM_PointPipl.py
#CUDA_VISIBLE_DEVICES=0 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-MI_FGSM_PointPipl.py  --evaluate_adv

#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU_PointPipl.py
#CUDA_VISIBLE_DEVICES=1 python ./test.py --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-IOU_PointPipl.py --evaluate_adv

CUDA_VISIBLE_DEVICES=0 python test.py --strategy PointPipl-PGD-filterOnce --fixedEPS 0.3 --attach_rate 0.2 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-AdaptiveEPS_PointPipl.py
CUDA_VISIBLE_DEVICES=0 python test.py --strategy PointPipl-PGD-filterOnce --fixedEPS 0.3 --attach_rate 0.2 --cfg_file cfgs/kitti_models/pv_rcnn.yaml --ckpt ../checkpoints/pv_rcnn_8369.pth --batch_size 1 --cfg_pyfile cfgs/adv_pv_rcnn/pv_rcnn-voxels_adv-AdaptiveEPS_PointPipl.py --evaluate_adv
