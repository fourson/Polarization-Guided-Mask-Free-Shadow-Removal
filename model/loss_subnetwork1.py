import torch
import torch.nn.functional as F

from utils.util import torch_laplacian
from .networks import CONV3_3_IN_VGG_19


def m_loss(m_pred, m, **kwargs):
    l1_loss_lambda = kwargs.get('l1_loss_lambda', 1)
    l1_loss = F.l1_loss(m_pred, m) * l1_loss_lambda
    print('m_loss l1:', l1_loss.item())

    l2_loss_lambda = kwargs.get('l2_loss_lambda', 1)
    l2_loss = F.mse_loss(m_pred, m) * l2_loss_lambda
    print('m_loss l2:', l2_loss.item())

    model = CONV3_3_IN_VGG_19
    pred_feature_map = model(m_pred)
    gt_feature_map = model(m).detach()  # we do not need the gradient of it
    perceptual_loss_lambda = kwargs.get('perceptual_loss_lambda', 1)
    perceptual_loss = F.mse_loss(pred_feature_map, gt_feature_map) * perceptual_loss_lambda
    print('m_loss: perceptual:', perceptual_loss.item())

    gradient_loss_lambda = kwargs.get('gradient_loss_lambda', 1)
    gradient_loss = F.mse_loss(torch.abs(torch_laplacian(m_pred)),
                               torch.abs(torch_laplacian(m))) * gradient_loss_lambda
    print('m_loss: gradient:', gradient_loss.item())

    return l1_loss + l2_loss + perceptual_loss + gradient_loss
