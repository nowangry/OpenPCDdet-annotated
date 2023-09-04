#!/usr/bin/env bash

set -x
#NGPUS=$1
#PY_ARGS=${@:2}
#
#python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

python -m torch.distributed.launch --nproc_per_node=2 train.py --launcher pytorch --cfg_file ./cfgs/waymo_models/pv_rcnn.yam
