import os

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_lr_lambda(lr_lambda_tag):
    if lr_lambda_tag == 'default':
        # keep the same
        return lambda epoch: 1
    elif lr_lambda_tag == 'subnetwork1':
        return lambda epoch: 1
    elif lr_lambda_tag == 'subnetwork2':
        return lambda epoch: 1
    elif lr_lambda_tag == 'full':
        # 100ep
        return lambda epoch: 1
    else:
        raise NotImplementedError('lr_lambda_tag [%s] is not found' % lr_lambda_tag)


@torch.jit.script
def torch_laplacian(img_tensor):
    # (N, C, H, W) image tensor -> (N, C, H, W) edge tensor, the same as cv2.Laplacian
    padded = F.pad(img_tensor, pad=[1, 1, 1, 1], mode='reflect')
    return padded[:, :, 2:, 1:-1] + padded[:, :, 0:-2, 1:-1] + padded[:, :, 1:-1, 2:] + padded[:, :, 1:-1, 0:-2] - \
           4 * img_tensor


def read_img(path, rgb=True):
    img = cv2.imread(path, -1)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    return img


def write_img(path, img, rgb=True):
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img * 255
    cv2.imwrite(path, img)


def compute_Si_from_Ii(I1, I2, I3, I4):
    I = (I1 + I2 + I3 + I4) / 2  # I
    S1 = I3 - I1  # I*p*cos(2*theta)
    S2 = I4 - I2  # I*p*sin(2*theta)
    return I, S1, S2


def mapping_func(v, a=-50, b=0.08):
    return 1 / (1 + np.exp(a * (v - b)))


def calc_prior_i(img):
    r, g, b = cv2.split(img)
    rgb_avg = np.mean(img, axis=2)
    # common sense1: shadow tends to be dark in the intensity domain
    dark_prior = (1 - r) * (1 - g) * (1 - b)
    # common sense2: shadow tends to be bluish in the intensity domain
    bluish_prior = (2 * b - r - g) / (2 * rgb_avg + 1e-7)
    # some stupid things to make the training process more stable on the synthetic data:
    # use mapping_func only on bluish_prior
    # simply averaging them instead of learning two weights
    prior_i = (dark_prior + mapping_func(bluish_prior)) / 2
    return prior_i


def calc_prior_p(dop):
    dc_dop = np.min(dop, axis=2)
    # the dark channel of shadow tends to be large in the polarization domain
    prior_p = mapping_func(dc_dop)
    return prior_p
