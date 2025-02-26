import os
import fnmatch

import numpy as np
import cv2

from scripts.script_utils.PolShadowEngine import PolShadowEngine


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


def shrink(I_list, mode):
    if mode == 're':
        # directly resize to 400 * 400
        func = lambda x: cv2.resize(x, (400, 400), interpolation=cv2.INTER_LINEAR)
    elif mode == 'md':
        # crop in the middle to 400 * 400
        func = lambda x: x[200:600, 200:600, :]
    elif mode == 'tl':
        # crop in the top left to 400 * 400
        func = lambda x: x[0:400, 0:400, :]
    elif mode == 'tr':
        # crop in the top right to 400 * 400
        func = lambda x: x[0:400, 400:800, :]
    elif mode == 'bl':
        # crop in the bottom left to 400 * 400
        func = lambda x: x[400:800, 0:400, :]
    elif mode == 'br':
        # crop in the bottom right to 400 * 400
        func = lambda x: x[400:800, 400:800, :]
    else:
        raise Exception('??????????')

    return [func(I) for I in I_list]


class Maker:
    def __init__(self, base_dir, matte_dir, out_base_dir):
        self.base_dir = base_dir
        self.matte_dir = matte_dir
        self.out_base_dir = out_base_dir

        self.names = [file_name[:-4] for file_name in
                      sorted(
                          fnmatch.filter(os.listdir(os.path.join(self.base_dir, 'I1')), '*.png'),
                          key=lambda x: int(x[:-4])
                      )]
        print('Pre-processing the following files: {}'.format(self.names))

        self.matte_paths = [os.path.join(matte_dir, matte_name) for matte_name in sorted(os.listdir(matte_dir))]
        print('Matte paths: {}'.format(self.matte_paths))

        self.shrink_modes = ['re', 'md', 'tl', 'tr', 'bl', 'br']
        self.matte_num = 15  # the number of mattes per scene

    def __len__(self):
        return len(self.names) * len(self.shrink_modes) * self.matte_num

    def make(self):
        cnt = 1
        for name in self.names:
            print('Fetching {}'.format(name))
            path_str = os.path.join(self.base_dir, 'I{}', name + '.png')
            # all images should be resized to 800 * 800 first, then processed
            I_list = [cv2.resize(read_img(path_str.format(i)), (800, 800), interpolation=cv2.INTER_LINEAR) for i in
                      range(1, 5)]
            for mode in self.shrink_modes:
                print('shrink: {}'.format(mode))
                name1 = '{}_{}'.format(name, mode)
                # shrink to 400 * 400
                I_list1 = shrink(I_list, mode)

                for _ in range(self.matte_num):
                    # matte should be resized to 400 * 400
                    matte = cv2.resize(read_img(self.matte_paths[cnt - 1]), (400, 400), interpolation=cv2.INTER_LINEAR)
                    engine = PolShadowEngine(*I_list1, matte)
                    data_item = engine.simulate()
                    name_ = '{}_{:0>4d}'.format(name1, cnt)

                    # write to files
                    print('Write matte for {}'.format(name_))
                    out_subdir = os.path.join(self.out_base_dir, 'matte')
                    ensure_dir(out_subdir)
                    write_img(os.path.join(out_subdir, name_ + '.png'), matte, rgb=True)

                    print('Write I1234 for {}'.format(name_))
                    for i in range(4):
                        out_subdir = os.path.join(self.out_base_dir, 'I{}'.format(i + 1))
                        ensure_dir(out_subdir)
                        write_img(os.path.join(out_subdir, name_ + '.png'), data_item['I1234'][i], rgb=True)

                    print('Write I1234_shadow for {}'.format(name_))
                    for i in range(4):
                        out_subdir = os.path.join(self.out_base_dir, 'I{}_shadow'.format(i + 1))
                        ensure_dir(out_subdir)
                        write_img(os.path.join(out_subdir, name_ + '.png'), data_item['I1234_shadow'][i], rgb=True)

                    print('==========Finishing {}/{}=========='.format(cnt, len(self)))
                    cnt += 1


if __name__ == '__main__':
    maker_args = {
        'base_dir': '../DatasetSourceFiles/for_train',
        'matte_dir': '../SynShadow/matte',
        'out_base_dir': '../data/train'
    }

    maker = Maker(**maker_args)
    maker.make()
