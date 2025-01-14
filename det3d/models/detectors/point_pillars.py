from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy 

@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        # PillarFeatureNet
        # P*C(30000*64)
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        # PointPillarsScatter
        # H*W*C(1, 64, 512, 512)
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            # RPN
            x = self.neck(x)
        # x: (bs, 384, 128, 128)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        # CenterHead
        preds, _ = self.bbox_head(x)

        if kwargs['cfg'].is_adv_eval or kwargs['cfg'].is_adv_eval_waymo or kwargs['cfg'].is_innocent_eval \
                or kwargs['cfg'].is_adv_pred_by_innocent_pipeline:
            predictions = self.bbox_head.predict(example, preds, self.test_cfg)
            return predictions
        elif kwargs['cfg'].adv_flag:  # adv
            if return_loss and (not kwargs['is_eval_after_attack']):
                loss = self.bbox_head.loss(example, preds, self.test_cfg, **kwargs)
                return loss
            elif (not return_loss) and (kwargs['is_eval_after_attack']):
                predictions = self.bbox_head.predict(example, preds, self.test_cfg)
                return predictions
            else:
                loss = self.bbox_head.loss(example, preds, self.test_cfg, **kwargs)
                predictions = self.bbox_head.predict(example, preds, self.test_cfg)
                return loss, predictions

        # 原流程
        elif return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x 
        preds, _ = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, None 
