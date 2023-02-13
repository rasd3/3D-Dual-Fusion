from .detector3d_template import Detector3DTemplate
from pcdet.models import dense_heads


class PVRCNN(Detector3DTemplate):

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg,
                         num_class=num_class,
                         dataset=dataset)
        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe', 'backbone_2d',
            'dense_head', 'point_head', 'roi_head', 'dense_head_3d'
        ]
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            if 'loss_box_of_pts' in batch_dict:
                loss += batch_dict['loss_box_of_pts']
                tb_dict['loss_box_of_pts'] = batch_dict['loss_box_of_pts']

            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def build_dense_head_3d(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD_3D', None) is None:
            return None, model_info_dict
        if self.model_cfg.DENSE_HEAD_3D.NAME == 'MMDet3DHead':
            dense_head_module = dense_heads.__all__[
                self.model_cfg.DENSE_HEAD_3D.NAME](
                    model_cfg=self.model_cfg.DENSE_HEAD_3D,
                    feat_lvl=self.model_cfg.BACKBONE_3D.FEATURE_LEVELS)
            model_info_dict['module_list'].append(dense_head_module)
            return dense_head_module, model_info_dict
        else:
            dense_head_module = dense_heads.__all__[
                self.model_cfg.DENSE_HEAD_3D.NAME](
                    model_cfg=self.model_cfg.DENSE_HEAD_3D,
                    input_channels=32,
                    num_class=self.num_class,
                    class_names=self.class_names,
                    grid_size=model_info_dict['grid_size'],
                    point_cloud_range=model_info_dict['point_cloud_range'],
                    predict_boxes_when_training=self.model_cfg.get(
                        'ROI_HEAD', False))
            model_info_dict['module_list'].append(dense_head_module)
            return dense_head_module, model_info_dict

    def get_training_loss(self, batch_dict):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rpn + loss_point + loss_rcnn
        if getattr(self, 'dense_head_3d', None):
            loss_rpn_3d, tb_dict = self.dense_head_3d.get_loss(
                batch_dict, tb_dict)
            tb_dict['loss_rpn3d'] = loss_rpn_3d.item()
            loss += loss_rpn_3d
        if self.model_cfg.BACKBONE_3D.get('SEG_LOSS', None):
            loss_aux_img_seg, tb_dict = self.backbone_3d.semseg.get_loss(
                batch_dict, tb_dict)
            tb_dict['loss_aux_img_seg'] = loss_aux_img_seg.item()
            loss += loss_aux_img_seg
        if self.model_cfg.BACKBONE_3D.get('AUX_PTS_LOSS', None):
            loss_aux_pts_seg, tb_dict = self.backbone_3d.aux_pts_head.get_loss(
                tb_dict)
            tb_dict['loss_aux_pts_seg'] = loss_aux_pts_seg.item()
            loss += loss_aux_pts_seg
        if self.model_cfg.BACKBONE_3D.get('AUX_CNS_LOSS', None):
            aux_pts_dict = self.backbone_3d.aux_pts_head.forward_ret_dict
            loss_aux_cons, tb_dict = self.backbone_3d.aux_cns_head(
                batch_dict, aux_pts_dict, tb_dict)
            tb_dict['loss_aux_cons'] = loss_aux_cons.item()
            loss += loss_aux_cons

        return loss, tb_dict, disp_dict
