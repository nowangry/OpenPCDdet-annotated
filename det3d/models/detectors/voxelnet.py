from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
import torch 
from copy import deepcopy 

@DETECTORS.register_module
class VoxelNet(SingleStageDetector):
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
        super(VoxelNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        
    def extract_feat(self, data):
        if 'voxels' not in data:
            output = self.reader(data['points'])    
            voxels, coors, shape = output 

            data = dict(
                features=voxels,
                coors=coors,
                batch_size=len(data['points']),
                input_shape=shape,
                voxels=voxels
            )
            input_features = voxels
        else:
            data = dict(
                features=data['voxels'],
                num_voxels=data["num_points"],
                coors=data["coordinates"],
                batch_size=len(data['points']),
                input_shape=data["shape"][0],
            )
        # VoxelFeatureExtractorV3(): locally aggregated feature, 对voxel内的点信息取平均
            input_features = self.reader(data["features"], data['num_voxels'])
        # SpMiddleResNetFHD()
        x, voxel_feature = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )  # SpMiddleResNetFHD
        # RPN
        if self.with_neck:
            x = self.neck(x)

        return x, voxel_feature

    def forward(self, example, return_loss=True, **kwargs):
        '''
        extract_feat:
        - VoxelFeatureExtractorV3
        - SpMiddleResNetFHD
        - RPN
        '''
        x, _ = self.extract_feat(example)
        '''
        bbox_head:
        - CenterHead
        '''
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
        x, voxel_feature = self.extract_feat(example)
        bev_feature = x 
        preds, final_feat = self.bbox_head(x)

        if return_loss:
            # manual deepcopy ...
            new_preds = []
            for pred in preds:
                new_pred = {} 
                for k, v in pred.items():
                    new_pred[k] = v.detach()
                new_preds.append(new_pred)

            boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

            return boxes, bev_feature, voxel_feature, final_feat, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            boxes = self.bbox_head.predict(example, preds, self.test_cfg)
            return boxes, bev_feature, voxel_feature, final_feat, None 