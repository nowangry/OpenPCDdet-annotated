import collections

from det3d.utils import build_from_cfg
from ..registry import PIPELINES


@PIPELINES.register_module
class Compose(object):
    def __init__(self, transforms, **kwargs):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                if transform['type'] == 'Empty':
                    continue
                transform = build_from_cfg(transform, PIPELINES, **kwargs)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError("transform must be callable or a dict")

    def __call__(self, res, info):
        '''
        pipeline:
        - LoadPointCloudFromFile
        - LoadPointCloudAnnotations
        - Preprocess
        - Voxelization: points_to_voxel()
        - AssignLabel
        - Reformat
        '''
        for t in self.transforms:  # 预处理pipeline
            res, info = t(res, info)
            if res is None:
                return None
        return res, info

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

