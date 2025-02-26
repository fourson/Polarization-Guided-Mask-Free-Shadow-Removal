import random

import numpy as np


def compute_stokes(I1, I2, I3, I4, compute_DoP_AoP=False):
    S0 = (I1 + I2 + I3 + I4) / 2  # I
    S1 = I3 - I1  # I*p*cos(2*theta)
    S2 = I4 - I2  # I*p*sin(2*theta)
    if compute_DoP_AoP:
        DoP = np.clip(np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), a_min=0, a_max=1)  # in [0, 1]
        AoP = np.arctan2(S2, S1) / 2  # in [-pi/2, pi/2]
        AoP = (AoP < 0) * np.pi + AoP  # convert to [0, pi] by adding pi to negative values
        AoP = AoP.astype(np.float32)
        return S0, S1, S2, DoP, AoP
    else:
        return S0, S1, S2


def compute_pol_imgs(S0, S1, S2, compute_DoP_AoP=False):
    I1 = (S0 - S1) / 2
    I2 = (S0 - S2) / 2
    I3 = (S0 + S1) / 2
    I4 = (S0 + S2) / 2
    if compute_DoP_AoP:
        DoP = np.clip(np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), a_min=0, a_max=1)  # in [0, 1]
        AoP = np.arctan2(S2, S1) / 2  # in [-pi/2, pi/2]
        AoP = (AoP < 0) * np.pi + AoP  # convert to [0, pi] by adding pi to negative values
        AoP = AoP.astype(np.float32)
        return I1, I2, I3, I4, DoP, AoP
    else:
        return I1, I2, I3, I4


def generate_darken_args(turb=True):
    """IEEE TCSVT 2021: learning from synthetic shadows for shadow detection and removal"""
    while True:
        x1 = random.uniform(0.0, 0.25)  # l_i in the paper
        y1 = 0.0
        x2 = 1.0
        y2 = random.uniform(0.1, 0.9)  # s_1 in the paper
        a = (y2 - y1) / (x2 - x1)  # Assume slope = const. for all channels
        if 0.15 < a < 0.85:
            break
    if not turb:
        b = y1 - a * x1  # no shift in x_intercept in RGB channel
        b = np.array([b] * 3, dtype=np.float32)
    else:
        # x1_R > x1_G > x1_B
        # y_g = a * x + b
        x1_G = x1
        mu, sigma = 0.05, 0.025
        rd1 = np.clip(np.random.normal(mu, sigma), a_min=0.01, a_max=1.0)
        rd2 = np.clip(np.random.normal(mu, sigma), a_min=0.01, a_max=1.0)
        # print('rd1:', rd1)
        # print('rd2:', rd2)
        x1_R = x1_G + rd1
        x1_B = x1_G - rd2
        # print('x1_R:', x1_R)
        # print('x1_G:', x1_G)
        # print('x1_B:', x1_B)
        b = np.array([y1 - a * x1_R, y1 - a * x1_G, y1 - a * x1_B], dtype=np.float32)
        # print('a:', a)
        # print('b:', b)
    return a, b


def darken_unpol(img, a, b):
    # img: (H, W, 3)
    # a: float
    # b: (3,) array
    img_dark = a * img + b
    img_dark = np.clip(img_dark, a_min=0.0, a_max=1.0)
    return img_dark


class PolShadowEngine:
    # To make the proposed method generalize well to diverse real scenes,
    # one should adjust the hyper-parameters according to the distribution of real scenes in different conditions.
    def __init__(self, I1, I2, I3, I4, matte):
        self.I1 = I1
        self.I2 = I2
        self.I3 = I3
        self.I4 = I4
        self.matte = matte

        self.I, _, _, self.DoP, self.AoP = compute_stokes(self.I1, self.I2, self.I3, self.I4, compute_DoP_AoP=True)
        self.a, self.b = generate_darken_args()

    def simulate(self):
        A = darken_unpol(self.I / 2, self.a, self.b) * 2
        amplification_coef = random.uniform(2, 3.5)
        DoP_A_base = np.mean(self.DoP, axis=2, keepdims=True) * amplification_coef
        noise_coef = random.uniform(0.05, 0.1)
        DoP_A_base = np.float32(np.random.normal(DoP_A_base, DoP_A_base * noise_coef))
        DoP_A_base = np.clip(DoP_A_base, a_min=0, a_max=1)
        mix_coef = random.uniform(0.5, 0.75)
        DoP_A_base = DoP_A_base * mix_coef + self.DoP * (1 - mix_coef)
        fluctuation_coef = random.uniform(0.1, 0.3)
        DoP_A_fluctuation = DoP_A_base * fluctuation_coef
        DoP_A = np.clip(
            np.float32(np.random.uniform(-DoP_A_fluctuation, DoP_A_fluctuation, DoP_A_base.shape) + DoP_A_base),
            a_min=0, a_max=1
        )

        AoP_A_base = np.ones_like(self.AoP) * np.pi / 2
        noise_coef = random.uniform(0.05, 0.1)
        AoP_A_base = np.float32(np.random.normal(AoP_A_base, AoP_A_base * noise_coef))
        AoP_A_base = np.clip(AoP_A_base, a_min=0, a_max=np.pi)
        reduction_coef = np.random.uniform(0.8, 0.95, size=AoP_A_base.shape)
        AoP_A = np.float32(AoP_A_base * reduction_coef)
        mix_coef = random.uniform(0.75, 0.95)
        AoP_A = AoP_A * mix_coef + self.AoP * (1 - mix_coef)

        S1_A = np.float32(A * DoP_A * np.cos(2 * AoP_A))
        S2_A = np.float32(A * DoP_A * np.sin(2 * AoP_A))
        self.A1, self.A2, self.A3, self.A4 = compute_pol_imgs(A, S1_A, S2_A, compute_DoP_AoP=False)
        self.D1, self.D2, self.D3, self.D4 = self.I1 - self.A1, self.I2 - self.A2, self.I3 - self.A3, self.I4 - self.A4

        self.I1_shadow = np.float32(self.A1 * self.matte + (1 - self.matte) * self.I1)
        self.I2_shadow = np.float32(self.A2 * self.matte + (1 - self.matte) * self.I2)
        self.I3_shadow = np.float32(self.A3 * self.matte + (1 - self.matte) * self.I3)
        self.I4_shadow = np.float32(self.A4 * self.matte + (1 - self.matte) * self.I4)

        return {'I1234': [self.I1, self.I2, self.I3, self.I4],
                'I1234_shadow': [self.I1_shadow, self.I2_shadow, self.I3_shadow, self.I4_shadow],
                'A1234': [self.A1, self.A2, self.A3, self.A4],
                'D1234': [self.D1, self.D2, self.D3, self.D4],
                }
