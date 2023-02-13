
"""
>>> python ./pth_merger.py ./transvoxel_L_1_7_20.pth ../model_zoo/mask_rcnn_r50_fpn_1x_nuim_20201008_195238-e99f5182_img_backbone.pth
- merge sys.argv[2] state_dict to sys.argv[1] state_dict
- out transvoxel_L_1_7_20+img_backbone.pth
"""

import sys
import torch

mod1 = torch.load(sys.argv[1])
mod2 = torch.load(sys.argv[2])

for key in mod2['state_dict']:
    mod1['state_dict'][key] = mod2['state_dict'][key]

torch.save(mod1, sys.argv[1][:-4] + '+img_backbone.pth')
