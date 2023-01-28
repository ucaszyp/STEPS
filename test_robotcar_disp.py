import argparse
import os
import os.path as osp

import cv2
import numpy as np
import pytorch_lightning
import torch
from mmcv import Config
from torchvision.transforms import ToTensor
from tqdm import tqdm

from datasets import ROBOTCAR_ROOT
from models import MODELS
from models.utils import disp_to_depth
from transforms import CenterCrop
from utils import read_list_from_file, save_color_disp, save_disp

# crop size
_CROP_SIZE = (1152, 640)
# output dir
_OUT_DIR = 'evaluation/rc_result/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str, help='Tested dataset.')
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--test', type=int, default=1)
    parser.add_argument('--vis', default=0)
    return parser.parse_args()


def load_weights_from_option(option):
    pretrained_cfg = Config.fromfile(option.model.day_disp_config_path)
    pretrained_path = osp.join(pretrained_cfg.output_dir,
                               'check_point_{}.pth'.format(option.model.day_disp_check_point))
    weights = torch.load(pretrained_path, map_location='cpu')
    return weights


if __name__ == '__main__':
    # parse args
    args = parse_args()
    # config
    cfg = Config.fromfile(osp.join('configs/', f'{args.config}.yaml'))
    cfg.test = args.test
    # print message
    print('Now evaluating with {}...'.format(os.path.basename(args.config)))
    # device
    device = torch.device('cuda:0')
    # read list file
    root_dir = ROBOTCAR_ROOT[args.root_dir] if args.root_dir in ROBOTCAR_ROOT else args.root_dir
    test_items = read_list_from_file(os.path.join(root_dir, 'test_split.txt'), 1)
    test_items = sorted(test_items)
    # store results
    predictions = []
    # model
    model_name = cfg.model.name
    net: pytorch_lightning.LightningModule = MODELS.build(name=model_name, option=cfg)
    net.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    net.to(device)
    net.eval()
    print('Successfully load weights from check point {}.'.format(args.checkpoint))
    # transform
    crop = CenterCrop(*_CROP_SIZE)
    to_tensor = ToTensor()
    # no grad
    with torch.no_grad():
        # predict
        for idx, item in enumerate(tqdm(test_items)):
            # path
            path = os.path.join(root_dir, 'rgb/', '{}.png'.format(item))
            # read image
            rgb = cv2.imread(path)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            # crop
            rgb = crop(rgb)
            gray = crop(gray)
            # resize
            rgb = cv2.resize(rgb, (cfg.dataset['width'], cfg.dataset['height']), interpolation=cv2.INTER_LINEAR)
            gray = cv2.resize(gray, (cfg.dataset['width'], cfg.dataset['height']), interpolation=cv2.INTER_LINEAR)
            # to tensor
            t_rgb = to_tensor(rgb).unsqueeze(0).to(device)
            t_gray = to_tensor(gray).unsqueeze(0).to(device)
            # feed into net
            outputs = net({('color', 0, 0): t_rgb,
                        ('color_aug', 0, 0): t_rgb,
                        ('color_gray', 0, 0): t_gray})
            disp = outputs[("disp", 0, 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            depth = depth.cpu()[0, 0, :, :].numpy()
            # append
            predictions.append(depth)
            if args.vis:
                scaled_disp = scaled_disp.cpu()[0, 0, :, :].numpy()
                out_fn = os.path.join("vis/rc", '{}_depth.png'.format("%05d" %idx))
                color_fn = os.path.join("vis/rc", '{}_rgb.png'.format("%05d" %idx))
                save_disp(rgb, scaled_disp, out_fn, color_fn, max_p=95, dpi=256)

    # stack
    predictions = np.stack(predictions, axis=0)
    # save
    np.save(os.path.join(_OUT_DIR, 'predictions.npy'), predictions, allow_pickle=False)
    # show message
    tqdm.write('Done.')
