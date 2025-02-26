from .loss_subnetwork1 import m_loss
from .loss_subnetwork2 import I_loss


def full_loss(m_pred, m, I_pred, I, **kwargs):
    m_loss_lambda = kwargs.get('m_loss_lambda', 1)
    L_m = m_loss(m_pred, m, **kwargs['m_loss']) * m_loss_lambda
    print('L_m:', L_m.item())

    I_loss_lambda = kwargs.get('I_loss_lambda', 1)
    L_I = I_loss(I_pred, I, **kwargs['I_loss']) * I_loss_lambda
    print('L_I:', L_I.item())

    return L_m + L_I
