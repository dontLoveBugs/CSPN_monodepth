#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2019-05-18 14:17
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : kitti_dataloader.py
"""

import glob
import os
import numpy as np
from torch.utils.data.dataset import Dataset
from dataloaders import transforms


def img_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    from PIL import Image
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8')  # in the range [0,255]
    img_file.close()
    return rgb_png


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt
    assert os.path.exists(filename), "file not found: {}".format(filename)
    from PIL import Image
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    depth = np.expand_dims(depth, -1)
    return depth


def get_img_depth_path(root_dir, mode='train'):
    root_d = os.path.join(root_dir, 'depth')
    root_rgb = os.path.join(root_dir, 'rgb')

    if mode.lower() == 'train':
        glob_gt = "train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png"

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([root_rgb] + ps[-6:-4] + ps[-2:-1] + ['data'] + ps[-1:])
            return pnew

        glob_gt = os.path.join(root_d, glob_gt)
    elif mode.lower() == 'val':
        raise NotImplementedError
    elif mode.lower() == 'val_selection':
        glob_gt = "val_selection_cropped/groundtruth_depth/*.png"

        def get_rgb_paths(p):
            return p.replace("groundtruth_depth", "image")

        glob_gt = os.path.join(root_dir, glob_gt)
    elif mode.lower() == 'test':
        glob_gt = None
        base = "/test_depth_prediction_anonymous/"
        glob_rgb = root_dir + base + "/image/*.png"
    else:
        raise NotImplementedError

    if glob_gt is not None:
        paths_gt = sorted(glob.glob(glob_gt))
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
    else:  # test only has rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)

    return paths_rgb, paths_gt


to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


class KittiDataSet(Dataset):

    def __init__(self, root, mode='train', transform=None, target_transform=None):
        self.img_path, self.depth_path = get_img_depth_path(root, mode)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode.lower()

    def __getitem__(self, index):
        if self.mode.lower() == 'test':
            img = img_read(self.img_path[index])
            return to_float_tensor(img)
        else:
            img = img_read(self.img_path[index])
            depth = depth_read(self.depth_path[index])

            if self.transform:
                img = self.transform(img)

            if self.target_transform:
                depth = self.target_transform(depth)

            return to_float_tensor(img), to_float_tensor(depth)

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    path_rgb, path_gt = get_img_depth_path('/Users/wangxin/Projects/data', mode='test')
    print('path rgb:', len(path_rgb), path_rgb)
    print('path gt:', len(path_gt), path_gt)

    for i in range(len(path_rgb)):
        from PIL import Image

        img = Image.open(path_rgb[i]).convert('RGB')

        img.show()
        break

    dataset = KittiDataSet(root='/Users/wangxin/Projects/data', mode='train')
    from torch.utils.data.dataloader import DataLoader

    loader = DataLoader(dataset, batch_size=1)

    for i, data in enumerate(loader):
        img, depth = data
        print(img.shape)
        print(depth.shape)
        break
