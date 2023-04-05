from .detector3d_template import Detector3DTemplate


class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        """
        PointNet2MSG
        PointHeadBox
        PointRCNNHead
        """
        cnt = 0
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
            cnt += 1
            # if (cnt >= 2):
            #     break

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # return pred_dicts, recall_dicts
            # print('test pointrcnn return')
            # print("打印键值pointrcnn")
            # for key in batch_dict:
            #     print(key)  # 打印key
            return pred_dicts, batch_dict # adv

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()  # 第一阶段的loss
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)  # 第二阶段的loss

        loss = loss_point + loss_rcnn
        for key in disp_dict:
            print(key) # 打印key
        return loss, tb_dict, disp_dict
