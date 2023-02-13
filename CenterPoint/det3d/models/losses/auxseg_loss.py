import torch
import torch.nn as nn

# from pcdet.utils import transform_utils, loss_utils
import numpy as np
from .focalloss_segmentation import FocalLoss
from det3d.core.utils import center_utils
from det3d.core.utils import circle_nms_jit
import cv2


class SEGLOSS(nn.Module):

    def __init__(self,
                 weight,
                 alpha,
                 gamma,
                 fg_weight,
                 bg_weight,
                 downsample_factor):
        """
        Initializes DDNLoss module
        Args:
            weight: float, Loss function weight
            alpha: float, Alpha value for Focal Loss
            gamma: float, Gamma value for Focal Loss
            disc_cfg: dict, Depth discretiziation configuration
            fg_weight: float, Foreground loss weight
            bg_weight: float, Background loss weight
            downsample_factor: int, Depth map downsample factor
        """
        super().__init__()
        self.device = torch.cuda.current_device()
        # self.balancer = Balancer(downsample_factor=downsample_factor,
        #                          fg_weight=fg_weight,
        #                          bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        # self.loss_func = FocalLoss()
        self.loss_func = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.weight = weight
        use_conv_for_no_stride = None
        upsample_strides=[0.5, 1, 2, 4]
        deblocks=[]
        in_channels = 256
        out_channel = 256
        self.num_classes = 1 # foreground : 1 background : 0 
        self.downsample_factor = downsample_factor
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight

        
        self.aux_img_cls = nn.Conv2d(out_channel, self.num_classes+1, 1)
        self.init_weights()
    
    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.aux_img_cls.bias, -np.log((1 - pi) / pi))
        
    
    # to visualize features
    def normalize(self, image_features): 
        image_features = image_features.squeeze(0).cpu().detach().numpy() 
        min = image_features.min() 
        max = image_features.max() 
        image_features = (image_features-min)/(max-min) 
        image_features = (image_features*255) 
        return image_features
    
    def compute_fg_mask(self, gt_boxes2d, shape, downsample_factor=1, device=torch.device("cuda")):
        fg_mask = torch.zeros(shape).to(torch.bool)
        
        N, C = gt_boxes2d.shape[:2]
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[n]
            radius = max(abs((u1-u2)/2), abs((v1-v2)/2))
            center = (u1+u2)/2, (v1+v2)/2
            fg_mask = center_utils.draw_umich_gaussian(fg_mask, center, radius)
            # fg_mask[v1:v2, u1:u2] = True 

        # gt_boxes2d /= downsample_factor
        # gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
        # gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
        # gt_boxes2d = gt_boxes2d.long()
        # B, N = gt_boxes2d.shape[:2]
        # for b in range(B):
        #     for n in range(N):
        #         u1, v1, u2, v2 = gt_boxes2d[b, n]
        #         fg_mask[b, v1:v2, u1:u2] = True 
        return fg_mask
    


    def gaussian2D(self,shape, sigma=1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1,-n:n+1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h


    def draw_umich_gaussian(self,heatmap, center, radius, ind,k=1):
        # def normalize(image_features): 
        #     image_features = image_features.squeeze(0).cpu().detach().numpy() 
        #     min = image_features.min() 
        #     max = image_features.max() 
        #     image_features = (image_features-min)/(max-min) 
        #     image_features = (image_features*255) 
        #     return image_features
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[2:4]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        heatmap[0][0][y - top:y + bottom, x - left:x + right] = torch.Tensor(gaussian)[0:bottom+top, 0 : right+left]
        if ind == 1 :
            heatmap[0][0][y - top:y + bottom, x - left:x - int(left/2)] = 0
            heatmap[0][0][y - top:y + bottom, x + int(right/2):x + right] = 0
        
        else :
            heatmap[0][0][y - top : int(y -top/2), x - left:x + right] = 0
            heatmap[0][0][y + int(bottom/2):y + bottom, x - left:x + right] = 0

        # masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        # if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        #     np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap


    def forward(self, batch_dict, cam_key, _idx):
        """
        Gets DDN loss
        Args:
            depth_logits: (B, D+1, H, W), Predicted depth logits
            depth_maps: (B, H, W), Depth map [m]
            gt_boxes2d: torch.Tensor (B, N, 4), 2D box labels for foreground/background balancing
        Returns:
            loss: (1), Depth distribution network loss
            tb_dict: dict[float], All losses to log in tensorboard
        """
        tb_dict = {}
        gt_boxes2d = batch_dict['gt_boxes2d']
        images = batch_dict['images'][cam_key][_idx]
        fg_pred = batch_dict['fg_pred']
        target_size = fg_pred.shape
        # import cv2
        # B,N,C = gt_boxes2d.shape
        # for i in range(N):
        #     batch_dict['images']=self.normalize(batch_dict['images'])
        #     a = cv2.rectangle(batch_dict['images'][0], (int(gt_boxes2d[0][i][0]), int(gt_boxes2d[0][i][1])), (int(gt_boxes2d[0][i][2]),int(gt_boxes2d[0][i][3])) ,(255,0,0),2)
        # cv2.imwrite(str(batch_dict['frame_id'])+'.png', a[0])

        images_rei = torch.zeros(1, 1, images.shape[0],images.shape[1],device='cuda')
        
        target_size = torch.zeros(target_size[0], target_size[2], target_size[3])
        # Compute loss
        # loss = self.loss_func(depth_logits, depth_target)


        # fg_mask = self.compute_fg_mask(gt_boxes2d=gt_boxes2d,
        #                                             shape=images_rei.shape,
        #                                             downsample_factor=self.downsample_factor,
        #                                             device=images_rei.device)
        b = torch.zeros(images_rei.shape)
        fg_mask = b
        N, C = gt_boxes2d.shape[:2]
        for n in range(N):
            b = torch.zeros(images_rei.shape)
            u1, v1, u2, v2 = gt_boxes2d[n]
            # fg_mask[0][0][v1:v2, u1:u2] = True 
            radius = int(max(abs((u1-u2)/2), abs((v1-v2)/2)))
            if abs((u1-u2)/2)  > abs((v1-v2)/2) : ind = 0
            else : ind = 1
            center = (u1+u2)/2, (v1+v2)/2
            fg_mask += self.draw_umich_gaussian(b, center, radius, ind)

        # if len(fg_mask.shape) == 3 :
        #     fg_mask = fg_mask.unsqueeze(dim=0)
        # elif len(fg_mask.shape) == 5:
        #     fg_mask = fg_mask.squeeze(dim=0)
        fg_mask = nn.functional.interpolate(fg_mask.to(torch.float), size=target_size.shape[1:]).squeeze(dim=1)
        fg_mask = torch.clip(fg_mask,0,1)
        fg_mask_bool = (fg_mask>0.5).to(torch.bool)
        
        bg_mask_bool = ~fg_mask_bool
        bg_mask = bg_mask_bool.to(torch.long)

        # # Compute balancing weights
        weights = (self.fg_weight * fg_mask_bool.to(torch.long) + self.bg_weight * bg_mask_bool.to(torch.long)).cuda()
        num_pixels = fg_mask.sum() + bg_mask.sum()

        def normalize(image_features): 
            image_features = image_features.squeeze(0).cpu().detach().numpy() 
            min = image_features.min() 
            max = image_features.max() 
            image_features = (image_features-min)/(max-min) 
            image_features = (image_features*255) 
            return image_features

        # Compute losses
        loss = self.loss_func(fg_pred, fg_mask.cuda().squeeze(dim=1).to(torch.int64))
        loss *=weights
        fg_loss = torch.mul(loss,fg_mask.cuda()).sum() / num_pixels
        bg_loss = torch.mul(loss,bg_mask.cuda()).sum() / num_pixels

        # Get total loss
        loss = fg_loss + bg_loss
        tb_dict = {"balancer_loss": loss.item(), "fg_loss": fg_loss.item(), "bg_loss": bg_loss.item()}

        # Final loss
        loss *= self.weight
        tb_dict.update({"aux_seg_loss": loss.item()})

        return loss
