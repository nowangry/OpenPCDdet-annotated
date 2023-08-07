from .detector3d_template import Detector3DTemplate


class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    """
    1、MeanVFE
    2、VoxelBackBone8x
    3、HeightCompression  高度方向堆叠
    4、VoxelSetAbstraction
    5、BaseBEVBackbone
    6、AnchorHeadSingle  第一阶段预测结果
    7、PointHeadSimple   Predicted Keypoint Weighting
    8、PVRCNNHead
    """
    def forward(self, batch_dict, return_loss=False, **kwargs):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        ## adv =========================================================================================================
        cfg = batch_dict['cfg']
        if cfg.is_adv_eval or cfg.is_innocent_eval:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, batch_dict
        elif cfg.adv_flag:  # adv
            if return_loss and (not cfg['is_eval_after_attack']):
                disp_dict = {}
                disp_dict['voxels'] = batch_dict['voxels']
                loss, tb_dict, disp_dict = self.get_training_loss(disp_dict)
                return loss
            elif (not return_loss) and (cfg['is_eval_after_attack']):
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, batch_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts
        ## adv =========================================================================================================

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, disp_dict=None):
        # disp_dict = {}
        disp_dict = {} if disp_dict is None else disp_dict
        # anchor head损失
        loss_rpn, tb_dict = self.dense_head.get_loss()  # AnchorHeadTemplate, 第一阶段损失
        # point head损失(PKW 前背景分割)
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)  # PointHeadSimple
        # roi头损失
        loss_rcnn, tb_dict, disp_dict = self.roi_head.get_loss(tb_dict, disp_dict)  # RoIHeadTemplate

        loss = loss_rpn + loss_point + loss_rcnn

        disp_dict['loss_rpn'] = loss_rpn
        disp_dict['loss_point'] = loss_point
        disp_dict['loss_rcnn'] = loss_rcnn
        disp_dict['loss'] = loss

        return loss, tb_dict, disp_dict
