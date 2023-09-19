import numpy as np
from det3d.ops.point_cloud.point_cloud_ops import points_to_voxel, points_to_voxel_for_iterAttack, \
    points_to_voxel_mark_index, points_to_voxel_for_iterAttack_momentum, \
    points_to_voxel_for_iterAttack_momentum_GV, points_to_voxel_for_iterAttack_momentum_V

class VoxelGenerator:
    def __init__(self, voxel_size, point_cloud_range, max_num_points, max_voxels=20000):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels=-1):
        if max_voxels == -1:
            max_voxels = self._max_voxels

        return points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            max_voxels,
        )

    def generate_mark_index(self, points, max_voxels=-1):
        if max_voxels == -1:
            max_voxels = self._max_voxels

        return points_to_voxel_mark_index(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            max_voxels,
        )

    def generate_for_iterAttack(self, points, max_voxels=-1, points_innocent_ori=np.array([])):
        if max_voxels == -1:
            max_voxels = self._max_voxels

        return points_to_voxel_for_iterAttack(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            max_voxels,
            points_innocent_ori,
        )

    def generate_for_iterAttack_momentum(self, points, max_voxels=-1, points_innocent_ori=np.array([]),
                                         momentum=np.array([])):
        if max_voxels == -1:
            max_voxels = self._max_voxels

        return points_to_voxel_for_iterAttack_momentum(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            max_voxels,
            points_innocent_ori,
            momentum,
        )

    def generate_for_iterAttack_momentum_GV(self, points, max_voxels=-1, points_innocent_ori=np.array([]),
                                         momentum=np.array([]), GV_grad=np.array([]), grad_points=np.array([]),
                                         points_adv=np.array([]), v=np.array([]),
        ):
        if max_voxels == -1:
            max_voxels = self._max_voxels

        return points_to_voxel_for_iterAttack_momentum_GV(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            max_voxels,
            points_innocent_ori,
            momentum,
            GV_grad,
            grad_points,
            points_adv,
            v,
        )

    def generate_for_iterAttack_momentum_V(self, points, max_voxels=-1, points_innocent_ori=np.array([]),
                                         momentum=np.array([]), v=np.array([]),
        ):
        if max_voxels == -1:
            max_voxels = self._max_voxels

        return points_to_voxel_for_iterAttack_momentum_V(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            max_voxels,
            points_innocent_ori,
            momentum,
            v,
        )

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
