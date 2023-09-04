import sys
import pickle
import json
import random
import operator
from numba.cuda.simulator.api import detect
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

from det3d.datasets.custom import PointCloudDataset

from det3d.datasets.registry import DATASETS

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
except:
    print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)


@DATASETS.register_module
class WaymoDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, elongation

    def __init__(
            self,
            info_path,
            root_path,
            cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        sample=False,
        nsweeps=1,
        load_interval=1,
        **kwargs,
    ):
        self.load_interval = load_interval 
        self.sample = sample
        self.nsweeps = nsweeps
        print("Using {} sweeps".format(nsweeps))
        self.collected_token = kwargs.get('collected_token', None)  # adv
        super(WaymoDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names, **kwargs
        )

        self._info_path = info_path
        self._class_names = class_names
        self._num_point_features = WaymoDataset.NumPointFeatures if nsweeps == 1 else WaymoDataset.NumPointFeatures+1

    def reset(self):
        assert False 

    def load_infos(self, info_path):

        with open(self._info_path, "rb") as f:
            _waymo_infos_all = pickle.load(f)

        if self.collected_token == None:
            self._waymo_infos = _waymo_infos_all[::self.load_interval]
        else:
            self._waymo_infos = []
            for waymo_infos in _waymo_infos_all:
                token = waymo_infos['token']
                if token in self.collected_token:
                    self._waymo_infos.append(waymo_infos)

        print("Using {} Frames".format(len(self._waymo_infos)))

    def __len__(self):

        if not hasattr(self, "_waymo_infos"):
            self.load_infos(self._info_path)

        return len(self._waymo_infos)

    def get_sensor_data(self, idx):
        info = self._waymo_infos[idx]

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "annotations": None,
                "nsweeps": self.nsweeps, 
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "type": "WaymoDataset",
        }

        data, _ = self.pipeline(res, info)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    def evaluation(self, detections, output_dir=None, testset=False, **kwargs):
        from .waymo_common import _create_pd_detection, reorganize_info

        infos = self._waymo_infos
        infos = reorganize_info(infos)

        _create_pd_detection(detections, infos, output_dir)

        print("use waymo devkit tool for evaluation")

        return None, None

    ## eval from nuScenesStyle
    def evaluation_nuScenesStyle(self, detections, output_dir=None, testset=False, **kwargs):
        self.eval_version = "detection_cvpr_2019"
        # adv
        cfg = kwargs['cfg']
        version = 'v1.0-mini'

        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-mini-train": "mini_train",  # adv 改为evaluate训练集
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        if not testset:
            dets = []
            gt_annos = self.ground_truth_annotations_waymo()
            assert gt_annos is not None

            miss = 0
            for gt in gt_annos:
                try:
                    dets.append(detections[gt["token"]])
                except Exception:
                    miss += 1

            assert miss == 0  # debug
        else:
            dets = [v for _, v in detections.items()]
            assert len(detections) == 6008

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        # adv 创建NuScenes数据类
        if 'v1.0-mini' in version:
            # nusc = NuScenes(version='v1.0-mini', dataroot=str(self._root_path), verbose=True)
            nusc = NuScenes(version='v1.0-mini', dataroot=str(r'/home/jqwu/Datasets/nuScenes/v1.0-mini/'), verbose=True)
        else:
            nusc = NuScenes(version=version, dataroot=str(self._root_path), verbose=True)

        # 处理class名称
        mapped_class_names = []  # ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        for n in self._class_names:
            # if n in self._name_mapping:
            #     mapped_class_names.append(self._name_mapping[n])
            # else:
            mapped_class_names.append(n)
        # 生成attr结果等
        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)  # 计算quat velocity属性
            boxes = _lidar_nusc_box_to_global(nusc, boxes, det["metadata"][
                "token"])  # Move box to ego vehicle coord system， to global coord system
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]  # label映射到class name
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                        "VEHICLE",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle", "CYCLIST"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian", "PEDESTRIAN"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None

                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),  # 3D 中心
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }
                annos.append(nusc_anno)
            nusc_annos["results"].update({det["metadata"]["token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1].split(".")[0]
        res_path = str(Path(output_dir) / Path(name + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,
                self.eval_version,
                res_path,
                eval_set_map[self.version],
                output_dir,
                **kwargs
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"], },
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"], },
            }
        else:
            res = None

        return res, None

    def ground_truth_annotations_waymo(self):
        if "gt_boxes" not in self._waymo_infos[0]:
            return None
        cls_range_map = config_factory(self.eval_version).serialize()['class_range']
        cls_range_map['SIGN'] = cls_range_map['traffic_cone']
        cls_range_map['VEHICLE'] = cls_range_map['car']
        cls_range_map['PEDESTRIAN'] = cls_range_map['pedestrian']
        cls_range_map['CYCLIST'] = cls_range_map['bicycle']
        gt_annos = []
        for info in self._waymo_infos:
            gt_names = np.array(info["gt_names"])
            gt_boxes = info["gt_boxes"]
            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            # det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append(
                {
                    "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                    "alpha": np.full(N, -10),
                    "occluded": np.zeros(N),
                    "truncated": np.zeros(N),
                    "name": gt_names[mask],
                    "location": gt_boxes[mask][:, :3],
                    "dimensions": gt_boxes[mask][:, 3:6],
                    "rotation_y": gt_boxes[mask][:, 6],
                    "token": info["token"],
                }
            )
        return gt_annos
