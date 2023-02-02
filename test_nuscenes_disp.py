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

from datasets import NUSCENES_ROOT
from models import MODELS
from models.utils import disp_to_depth
from utils import read_list_from_file, save_color_disp, save_disp
import time
# output dir
_OUT_DIR = 'evaluation/ns_result/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str, help='weather')
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--test', type=int, default=1)
    parser.add_argument('--vis', type=int, default=0)
    return parser.parse_args()


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
    test_items = read_list_from_file(osp.join(NUSCENES_ROOT['split'], '{}_test_split.txt'.format(args.root_dir)), 1)
    test_items = sorted(test_items)
    # store results
    predictions = []
    # model
    model_name = cfg.model.name
    net: pytorch_lightning.LightningModule = MODELS.build(name=model_name, option=cfg)
    net.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    net.to(device)
    net.eval()
    print('Successfully load weights from {}.'.format(args.checkpoint))
    # transform
    to_tensor = ToTensor()
    # visualization
    img_lists = []
    if args.visualization:
        visualization_dir = os.path.join(_OUT_DIR, 'visualization/')
        if not os.path.exists(visualization_dir):
            os.mkdir(visualization_dir)
    # no grad
    total_time = 0
    with torch.no_grad():
        # predict
        for idx, item in enumerate(tqdm(test_items)):
            # path
            path = osp.join(NUSCENES_ROOT['test_color'], item + '.jpg')
            # read image
            rgb = cv2.imread(path)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            # resize
            rgb = cv2.resize(rgb, (cfg.dataset['width'], cfg.dataset['height']), interpolation=cv2.INTER_LINEAR)
            gray = cv2.resize(gray, (cfg.dataset['width'], cfg.dataset['height']), interpolation=cv2.INTER_LINEAR)
            # to tensor
            t_rgb = to_tensor(rgb).unsqueeze(0).to(device)
            t_gray = to_tensor(gray).unsqueeze(0).to(device)
            # feed into net
            a = time.time()
            outputs = net({('color', 0, 0): t_rgb,
                           ('color_aug', 0, 0): t_rgb,
                           ('color_gray', 0, 0): t_gray})
            disp = outputs[("disp", 0, 0)]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            b = time.time()
            total_time += (b - a)
            depth = depth.cpu()[0, 0, :, :].numpy()
            # append
            predictions.append(depth)
            # visualization
            if args.vis:
                scaled_disp = scaled_disp.cpu()[0, 0, :, :].numpy()
                out_fn = os.path.join("vis/ns", '{}_depth.png'.format("%03d" %idx))
                color_fn = os.path.join("vis/ns_", '{}_rgb.png'.format("%03d" %idx))
                save_color_disp(rgb[:, :, ::-1], scaled_disp, out_fn, max_p=95, dpi=256)
                save_disp(rgb[:, :, ::-1], scaled_disp, out_fn, color_fn, max_p=95, dpi=256)
    print("=========================")
    fps = int(500 / total_time)
    print(fps)
    # stack
    predictions = np.stack(predictions, axis=0)
    # save
    np.save(os.path.join(_OUT_DIR, 'predictions.npy'), predictions, allow_pickle=False)
    # show message
    tqdm.write('Done.')