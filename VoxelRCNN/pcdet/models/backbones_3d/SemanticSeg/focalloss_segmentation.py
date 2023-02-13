import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

def label_to_one_hot_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    ignore_index=255,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([
                [[0, 1], 
                [2, 0]]
            ])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    shape = labels.shape
    # one hot : (B, C=ignore_index+1, H, W)
    one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
    
    # labels : (B, H, W)
    # labels.unsqueeze(1) : (B, C=1, H, W)
    # one_hot : (B, C=ignore_index+1, H, W)
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    
    # ret : (B, C=num_classes, H, W)
    ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]
    
    return ret


def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):
    
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    # input : (B, C, H, W)
    n = input.size(0) # B
    
    # out_sie : (B, H, W)
    out_size = (n,) + input.size()[2:]
    
    # input : (B, C, H, W)
    # target : (B, H, W)
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
    
    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):
        # alpha : (B, C, H, W)
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)       
        

    # compute softmax over the classes axis
    # input_soft : (B, C, H, W)
    # input_soft = F.softmax(input, dim=1) + eps
    threshold = 0.6
    input_soft = F.sigmoid(input) + eps 

    # create the labels one hot tensor
    # target_one_hot : (B, C, H, W)
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device, dtype=input.dtype, ignore_index=ignore_index)

    # compute the actual focal loss
    weight = torch.pow(1.0 - input_soft, gamma)
    
    # alpha, weight, input_soft : (B, C, H, W)
    # focal : (B, C, H, W)
    focal = -alpha * weight * torch.log(input_soft)
    
    # loss_tmp : (B, H, W)
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':
        # loss : (B, H, W)
        loss = loss_tmp
    elif reduction == 'mean':
        # loss : scalar
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        # loss : scalar
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math:

        FL(p_t) = -alpha_t(1 - p_t)^{gamma}, log(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Scalar to enforce numerical stabiliy.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, alpha, gamma = 2.0, reduction = 'mean', eps = 1e-8, ignore_index=30):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)



# import numpy as np

# import torch

# import torch.nn as nn

# import torch.nn.functional as F

 

 

# #          

# class FocalLoss(nn.Module):

#   """

#   This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in

#   'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'

#     Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)

#   :param num_class:

#   :param alpha: (tensor) 3D or 4D the scalar factor for this criterion

#   :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more

#           focus on hard misclassified example

#   :param smooth: (float,double) smooth value when cross entropy

#   :param balance_index: (int) balance class index, should be specific when alpha is float

#   :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.

#   """

 

#   def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):

#     super(FocalLoss, self).__init__()

#     self.num_class = num_class

#     self.alpha = alpha

#     self.gamma = gamma

#     self.smooth = smooth

#     self.size_average = size_average

 

#     if self.alpha is None:

#       self.alpha = torch.ones(self.num_class, 1)

#     elif isinstance(self.alpha, (list, np.ndarray)):

#       assert len(self.alpha) == self.num_class

#       self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)

#       self.alpha = self.alpha / self.alpha.sum()

#     elif isinstance(self.alpha, float):

#       alpha = torch.ones(self.num_class, 1)

#       alpha = alpha * (1 - self.alpha)

#       alpha[balance_index] = self.alpha

#       self.alpha = alpha

#     else:

#       raise TypeError('Not support alpha type')

 

#     if self.smooth is not None:

#       if self.smooth < 0 or self.smooth > 1.0:

#         raise ValueError('smooth value should be in [0,1]')

 

#   def forward(self, input, target):

#     logit = F.softmax(input, dim=1)

 

#     if logit.dim() > 2:

#       # N,C,d1,d2 -> N,C,m (m=d1*d2*...)

#       logit = logit.view(logit.size(0), logit.size(1), -1)

#       logit = logit.permute(0, 2, 1).contiguous()

#       logit = logit.view(-1, logit.size(-1))

#     target = target.view(-1, 1)

 

#     # N = input.size(0)

#     # alpha = torch.ones(N, self.num_class)

#     # alpha = alpha * (1 - self.alpha)

#     # alpha = alpha.scatter_(1, target.long(), self.alpha)

#     epsilon = 1e-10

#     alpha = self.alpha

#     if alpha.device != input.device:

#       alpha = alpha.to(input.device)

 

#     idx = target.cpu().long()

#     one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()

#     one_hot_key = one_hot_key.scatter_(1, idx, 1)

#     if one_hot_key.device != logit.device:

#       one_hot_key = one_hot_key.to(logit.device)

 

#     if self.smooth:

#       one_hot_key = torch.clamp(

#         one_hot_key, self.smooth, 1.0 - self.smooth)

#     pt = (one_hot_key * logit).sum(1) + epsilon

#     logpt = pt.log()

 

#     gamma = self.gamma

 

#     alpha = alpha[idx]

#     loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

 

#     if self.size_average:

#       loss = loss.mean()

#     else:

#       loss = loss.sum()

#     return loss

 

 

 

# class BCEFocalLoss(torch.nn.Module):

#   """

#       Focalloss alpha   

#   """

#   def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):

#     super().__init__()

#     self.gamma = gamma

#     self.alpha = alpha

#     self.reduction = reduction

 

#   def forward(self, _input, target):

#     pt = torch.sigmoid(_input)

#     alpha = self.alpha

#     loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

#     if self.reduction == 'elementwise_mean':

#       loss = torch.mean(loss)

#     elif self.reduction == 'sum':

#       loss = torch.sum(loss)

#     return loss