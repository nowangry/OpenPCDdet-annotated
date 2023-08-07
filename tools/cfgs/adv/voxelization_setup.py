from det3d.datasets.pipelines.preprocess import Voxelization
from det3d.torchie.utils.config import ConfigDict

# voxelization_cfg = ConfigDict(
#     {'cfg': {'range': [-54, -54, -5.0, 54, 54, 3.0], 'voxel_size': [0.075, 0.075, 0.2], 'max_points_in_voxel': 10,
#              'max_voxel_num': [120000, 160000]}})
voxelization_cfg = ConfigDict(
    {'cfg': {'range': [-54, -54, -5.0, 54, 54, 3.0], 'voxel_size': [0.075, 0.075, 0.2], 'max_points_in_voxel': 10,
             'max_voxel_num': [120000, 160000]}})

Voxelize = Voxelization(**voxelization_cfg)
max_voxels = 120000

out_size_factor = 8
