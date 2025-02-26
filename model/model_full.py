from base.base_model import BaseModel

from .model_subnetwork1 import DefaultModel as Subnetwork1
from .model_subnetwork2 import DefaultModel as Subnetwork2


class DefaultModel(BaseModel):
    def __init__(self, init_dim=32, n_blocks=6, norm_type='instance', use_dropout=False):
        super(DefaultModel, self).__init__()

        self.Subnetwork1 = Subnetwork1(init_dim, norm_type, use_dropout)
        self.Subnetwork2 = Subnetwork2(init_dim, n_blocks, norm_type)

    def forward(self, prior_i, prior_p, m_shadow, I_shadow):
        m = self.Subnetwork1(prior_i, prior_p, m_shadow)
        I = self.Subnetwork2(prior_i, prior_p, m, I_shadow)
        return m, I
