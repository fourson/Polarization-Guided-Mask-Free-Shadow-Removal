import os
import fnmatch

import numpy as np
from torch.utils.data import Dataset

from utils.util import read_img, compute_Si_from_Ii, calc_prior_i, calc_prior_p


class TrainDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.I1_shadow_dir = os.path.join(data_dir, 'I1_shadow')
        self.I2_shadow_dir = os.path.join(data_dir, 'I2_shadow')
        self.I3_shadow_dir = os.path.join(data_dir, 'I3_shadow')
        self.I4_shadow_dir = os.path.join(data_dir, 'I4_shadow')

        self.I1_dir = os.path.join(data_dir, 'I1')
        self.I2_dir = os.path.join(data_dir, 'I2')
        self.I3_dir = os.path.join(data_dir, 'I3')
        self.I4_dir = os.path.join(data_dir, 'I4')

        self.names = [file_name[:-4] for file_name in fnmatch.filter(os.listdir(self.I1_shadow_dir), '*.png')]

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        I1_shadow = read_img(os.path.join(self.I1_shadow_dir, name + '.png'), rgb=True)
        I2_shadow = read_img(os.path.join(self.I2_shadow_dir, name + '.png'), rgb=True)
        I3_shadow = read_img(os.path.join(self.I3_shadow_dir, name + '.png'), rgb=True)
        I4_shadow = read_img(os.path.join(self.I4_shadow_dir, name + '.png'), rgb=True)
        I_shadow, S1_shadow, S2_shadow = compute_Si_from_Ii(I1_shadow, I2_shadow, I3_shadow, I4_shadow)
        m_shadow = np.sqrt(S1_shadow ** 2 + S2_shadow ** 2)
        p_shadow = np.clip(m_shadow / (I_shadow + 1e-7), a_min=0, a_max=1)
        prior_i = calc_prior_i(I_shadow / 2)
        prior_p = calc_prior_p(p_shadow)

        I1 = read_img(os.path.join(self.I1_dir, name + '.png'), rgb=True)
        I2 = read_img(os.path.join(self.I2_dir, name + '.png'), rgb=True)
        I3 = read_img(os.path.join(self.I3_dir, name + '.png'), rgb=True)
        I4 = read_img(os.path.join(self.I4_dir, name + '.png'), rgb=True)
        I, S1, S2 = compute_Si_from_Ii(I1, I2, I3, I4)
        m = np.sqrt(S1 ** 2 + S2 ** 2)

        if self.transform:
            prior_i = self.transform(prior_i)
            prior_p = self.transform(prior_p)
            m_shadow = self.transform(m_shadow)

            m = self.transform(m)

        return {'prior_i': prior_i, 'prior_p': prior_p, 'm_shadow': m_shadow, 'm': m, 'name': name}


class InferDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.I1_shadow_dir = os.path.join(data_dir, 'I1_shadow')
        self.I2_shadow_dir = os.path.join(data_dir, 'I2_shadow')
        self.I3_shadow_dir = os.path.join(data_dir, 'I3_shadow')
        self.I4_shadow_dir = os.path.join(data_dir, 'I4_shadow')

        self.names = [file_name[:-4] for file_name in fnmatch.filter(os.listdir(self.I1_shadow_dir), '*.png')]

        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        I1_shadow = read_img(os.path.join(self.I1_shadow_dir, name + '.png'), rgb=True)
        I2_shadow = read_img(os.path.join(self.I2_shadow_dir, name + '.png'), rgb=True)
        I3_shadow = read_img(os.path.join(self.I3_shadow_dir, name + '.png'), rgb=True)
        I4_shadow = read_img(os.path.join(self.I4_shadow_dir, name + '.png'), rgb=True)
        I_shadow, S1_shadow, S2_shadow = compute_Si_from_Ii(I1_shadow, I2_shadow, I3_shadow, I4_shadow)
        m_shadow = np.sqrt(S1_shadow ** 2 + S2_shadow ** 2)
        p_shadow = np.clip(m_shadow / (I_shadow + 1e-7), a_min=0, a_max=1)
        prior_i = calc_prior_i(I_shadow / 2)
        prior_p = calc_prior_p(p_shadow)

        if self.transform:
            prior_i = self.transform(prior_i)
            prior_p = self.transform(prior_p)
            m_shadow = self.transform(m_shadow)

        return {'prior_i': prior_i, 'prior_p': prior_p, 'm_shadow': m_shadow, 'name': name}
