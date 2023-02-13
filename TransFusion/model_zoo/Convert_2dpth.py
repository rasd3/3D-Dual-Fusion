
import sys
import copy

import torch

"""
>>> python Conver_2dpth.py ./mask_rcnn_r50_fpn_1x_nuim_20201008_195238-e99f5182.pth
- make mask_rcnn_r50_fpn_1x_nuim_20201008_195238-e99f5182_img_backbone.pth
- with converted state_dict keys (backbone -> img_backbone, neck -> img_neck)
"""

mod = torch.load(sys.argv[1])
mod_st = copy.deepcopy(mod['state_dict'])
for key in mod['state_dict']:
    if 'backbone' in key or 'neck' in key:
        mod_st['img_' + key] = mod_st[key]
        mod_st.pop(key)
    else:
        mod_st.pop(key)
mod['state_dict'] = mod_st

torch.save(mod, sys.argv[1][:-4] + '_img_backbone.pth')
