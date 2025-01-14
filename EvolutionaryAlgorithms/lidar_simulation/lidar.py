import numpy as np
from numba import cuda
from math import ceil

from .utilities.visualization import visualize_3d
from .utilities.ray_casting import ray_intersection, ray_intersection_gpu, ray_intersection_uv_gpu

# try:
#     from .utilities.visualization import visualize_3d
#     from .utilities.ray_casting import ray_intersection, ray_intersection_gpu, ray_intersection_uv_gpu
# except ImportError:
#     from utilities.visualization import visualize_3d
#     from utilities.ray_casting import ray_intersection, ray_intersection_gpu, ray_intersection_uv_gpu


class Lidar:
    def __init__(self, delta_azimuth=2 * np.pi / 4000, delta_elevation=np.pi / 128, position=(0, 0, 0)):
        self.delta_azimuth = delta_azimuth
        self.delta_elevation = delta_elevation
        self.position = np.array(position)

    def _tf_into_spherical_sensor_coordinates(self, points):
        pc_tmp = points - self.position
        distance = np.sqrt(np.sum(pc_tmp ** 2, axis=1))
        azimuth = np.arctan2(pc_tmp[:, 1], pc_tmp[:, 0])
        elevation = np.arcsin(pc_tmp[:, 2] / distance)
        points_spherical = np.dstack((distance, azimuth, elevation))[0]
        return points_spherical

    def _tf_into_cartesian_coordinates(self, points_spherical):
        r = points_spherical[:, 0]
        az = points_spherical[:, 1]
        el = points_spherical[:, 2]
        cos_el = np.cos(el)
        x = r * cos_el * np.cos(az)
        y = r * cos_el * np.sin(az)
        z = r * np.sin(el)
        points = np.dstack((x, y, z))[0]
        return points

    def _model_view_dimensions(self, pcloud_spherical):
        azimuth_min, elevation_min = np.min(pcloud_spherical[:, 1:], axis=0)
        azimuth_max, elevation_max = np.max(pcloud_spherical[:, 1:], axis=0)
        return azimuth_min, azimuth_max, elevation_min, elevation_max

    ####################################################################################################################
    def create_rays(self, vertices):
        vertices_spherical = self._tf_into_spherical_sensor_coordinates(vertices)
        azimuth_min, azimuth_max, elevation_min, elevation_max = self._model_view_dimensions(vertices_spherical)
        n1 = int((azimuth_max - azimuth_min) / self.delta_azimuth)
        n2 = int((elevation_max - elevation_min) / self.delta_elevation)
        az = np.linspace(azimuth_min, azimuth_max, n1)
        el = np.linspace(elevation_min, elevation_max, n2)
        rays_spherical = np.dstack((np.ones(n1 * n2), *[m.ravel() for m in np.meshgrid(az, el)]))[0]
        ray_directions = self._tf_into_cartesian_coordinates(rays_spherical)
        return self.position, ray_directions

    def sample_3d_model(self, vertices, polygons, rays_per_cycle=None, return_valid_ray_mask=False):
        '''
        Simulate lidar sensor measurement on a 3d model
        :param  vertices: np.array with x,y,z as columns (shape= n x 3)
                polygons: np.array with vertex indices for each polygon (shape= p x 3),
                          assumes 3-point-polygons
                rays_per_cycle: Setting this parameter limits the amount of rays that are computed at once
                                for limiting memory usage
                return_valid_ray_mask: if True the function returns a boolean mask for which ray hit a polygon
        :return: Measured vertices (shape= m x 3)
                 Mask of rays that hit
        '''
        ray_origin, ray_directions = self.create_rays(vertices)
        if rays_per_cycle is None:
            sampled_points, all_valid_rays = ray_intersection(ray_origin,
                                                              ray_directions,
                                                              vertices,
                                                              polygons)
        else:
            cycles = int(np.ceil(len(ray_directions) / rays_per_cycle))
            sampled_points = []
            all_valid_rays = []
            for c in range(cycles):
                idx = c * rays_per_cycle
                lidar_points, valid_ray = ray_intersection(ray_origin,
                                                           ray_directions[idx:idx + rays_per_cycle],
                                                           vertices,
                                                           polygons)
                sampled_points.append(lidar_points)
                all_valid_rays.append(valid_ray)
            sampled_points = np.vstack(sampled_points)
            all_valid_rays = np.hstack(all_valid_rays)

        if return_valid_ray_mask:
            return sampled_points, all_valid_rays
        else:
            return sampled_points

    def sample_3d_model_gpu(self, vertices, polygons):
        '''
        Simulate lidar sensor measurement on a 3d model
        :param  vertices: np.array with x,y,z as columns (shape= n x 3)
                polygons: np.array with vertex indices for each polygon (shape= p x 3),
                          assumes 3-point-polygons
        :return: Measured vertices (shape= m x 3). returns (0,0,0) in place of each invalid ray.
        '''
        ray_origin, ray_directions = self.create_rays(vertices)
        sampled_points = np.zeros((len(ray_directions), 3))
        ray_intersection_gpu[ceil(len(ray_directions) / 256), 256](ray_origin,
                                                                   ray_directions,
                                                                   vertices,
                                                                   polygons,
                                                                   sampled_points)
        cuda.synchronize()
        return sampled_points

    def sample_3d_model_with_texture_gpu(self, vertices, polygons, uv_coordinates, uv_coordinate_indices):
        '''
        Simulate lidar sensor measurement on a 3d model
        :param  vertices: np.array with x,y,z as columns (shape= n x 3)
                polygons: np.array with vertex indices for each polygon (shape= p x 3),
                          assumes 3-point-polygons
        :return: Measured vertices (shape= m x 3). returns (0,0,0) in place of each invalid ray.
        '''
        ray_origin, ray_directions = self.create_rays(vertices)
        sampled_points = np.zeros((len(ray_directions), 3))
        ray_hit_uv = np.zeros((len(ray_directions), 2))
        cuda.synchronize()  # this seems to prevent some random cuda context corruptions
        ray_intersection_uv_gpu[ceil(len(ray_directions) / 256), 256](ray_origin,
                                                                      ray_directions,
                                                                      vertices,
                                                                      polygons,
                                                                      uv_coordinates,
                                                                      uv_coordinate_indices,
                                                                      sampled_points,
                                                                      ray_hit_uv)
        cuda.synchronize()
        return sampled_points, ray_hit_uv


########################################################################################################################

def sample_usage():
    from data_loaders.load_3d_models import load_Porsche911
    # from utilities.visualization import visualize_3d

    point_cloud = Lidar(delta_azimuth=2 * np.pi / 2000,
                        delta_elevation=np.pi / 200,
                        position=(0, -10, 0)).sample_3d_model(*load_Porsche911(), rays_per_cycle=400)
    print(point_cloud)
    # visualize_3d(point_cloud)
    import pptk
    v = pptk.viewer(point_cloud)


def sample_usage_gpu():
    from data_loaders.load_3d_models import load_obj_file
    from utilities.geometry_calculations import rotate_point_cloud
    import os

    vertices, polygons = load_obj_file(os.path.expanduser("~/Downloads/3d models/Porsche_911_GT2.obj"))
    vertices = rotate_point_cloud(vertices, -.5)
    point_cloud = Lidar(delta_azimuth=2 * np.pi / 3000,
                        delta_elevation=np.pi / 800,
                        position=(0, -10, 1)).sample_3d_model_gpu(vertices, polygons)
    print(point_cloud)
    print(len(point_cloud))
    print(np.sum(point_cloud[:, 2]))

    import pptk
    v = pptk.viewer(point_cloud[np.any(point_cloud != 0)])
    v.set(point_size=.003)


def sample_usage_with_texture_gpu():
    from data_loaders.load_3d_models import load_obj_file
    from utilities.geometry_calculations import rotate_point_cloud
    import os

    obj_file = os.path.expanduser("~/Downloads/3d models/Porsche_911_GT2.obj")
    vertices, polygons, uv_coordinates, uv_coordinate_indices = load_obj_file(obj_file, texture=True)
    vertices = rotate_point_cloud(vertices, -.5)
    point_cloud, ray_hit_uv = \
        Lidar(delta_azimuth=2 * np.pi / 3000,
              delta_elevation=np.pi / 800,
              position=(0, -10, 1)).sample_3d_model_with_texture_gpu(vertices,
                                                                     polygons,
                                                                     uv_coordinates,
                                                                     uv_coordinate_indices)
    print(len(ray_hit_uv[np.any(ray_hit_uv != 0, axis=1)]))
    print(len(point_cloud[np.any(point_cloud != 0, axis=1)]))

    import pptk
    v = pptk.viewer(point_cloud[np.any(point_cloud != 0, axis=1)])
    v.set(point_size=.003)


if __name__ == "__main__":
    sample_usage_with_texture_gpu()
