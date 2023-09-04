from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        '''
        PillarVFE
        PointPillarScatter
        BaseBEVBackbone
        AnchorHeadSingle
        '''

    def forward(self, batch_dict, return_loss=False, **kwargs):
        for cur_module in self.module_list:
            if 'cfg' in kwargs and kwargs['cfg'].get('get_refined_box_when_training', False):
                batch_dict = cur_module(batch_dict, **kwargs)
            else:
                batch_dict = cur_module(batch_dict)

        ## adv =========================================================================================================
        cfg = batch_dict['cfg']
        if 'IS_ADV' in cfg and len(kwargs):
            # if cfg.is_adv_eval or cfg.is_innocent_eval:
            #     pred_dicts, recall_dicts = self.post_processing(batch_dict)
            #     return pred_dicts, batch_dict
            if cfg.adv_flag:  # adv
                if return_loss and (not kwargs['is_eval_after_attack']):
                    # disp_dict = {}
                    # disp_dict['voxels'] = batch_dict['voxels']
                    # loss, tb_dict, disp_dict = self.get_training_loss(disp_dict)
                    loss, tb_dict, disp_dict = self.get_training_loss()
                    return loss
                elif (not return_loss) and (kwargs['is_eval_after_attack']):
                    pred_dicts, recall_dicts = self.post_processing(batch_dict)
                    return pred_dicts, batch_dict
                elif return_loss and kwargs['is_eval_after_attack']:
                    disp_dict = {}
                    disp_dict['voxels'] = batch_dict['voxels']
                    loss, tb_dict, disp_dict = self.get_training_loss(disp_dict)
                    pred_dicts, recall_dicts = self.post_processing(batch_dict)
                    return loss, pred_dicts
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

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
