import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.append('..')

from ui import PyTable
from utils import read_list_from_file
from datasets import NUSCENES_ROOT
from tqdm import trange, tqdm


# target size


def compute_metrics(pred, gt):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    result = {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log, 'a1': a1, 'a2': a2, 'a3': a3}
    return result


def print_table(title, data, str_format):
    table = PyTable(list(data.keys()), title)
    table.add_item({k: str_format.format(v) for k, v in data.items()})
    table.print_table()


def evaluate():
    # check length
    pred_len, gt_len = len(pred_depth), len(gt_depth)
    assert pred_len == gt_len, 'The length of predictions must be same as ground truth.'
    # store result
    
    errors = {'abs_rel': [], 'sq_rel': [], 'rmse': [], 'rmse_log': [], 'a1': [], 'a2': [], 'a3': []}
    file_name = "./ns_ckpt_err.txt"
    # compute loss
    for i in trange(gt_len):
        # get item
        pred, gt = pred_depth[i], gt_depth[i]
        mask = (gt > args.min_depth) & (gt < args.max_depth)
        # resize
        gt_h, gt_w = gt.shape
        pred = cv2.resize(pred, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
        # get values
        pred_vals, gt_vals = pred[mask], gt[mask]
        if gt_vals.size <= 0:
            raise ValueError('The size of ground truth is zero.')
        # compute scale
        scale = np.median(gt_vals) / np.median(pred_vals)
        pred_vals *= scale
        pred_vals = np.clip(pred_vals, args.min_depth, args.max_depth)
        # compute error
        error = compute_metrics(pred_vals, gt_vals)
        # add
        for k in errors:
            errors[k].append(error[k])
    # compute mean
    errors = {k: np.mean(v).item() for k, v in errors.items()}
    # display
    tqdm.write('Done.')
    # output
    print_table('Evaluation Result', errors, '{:.3f}')
    # save result
    file_name = args.output_file_name
    if file_name is not None:
        out_dir = osp.dirname(file_name)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(file_name, 'w') as fo:
            json.dump(errors, fo)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('weather', type=str, help='Weather.', choices=['day', 'night'])
    parser.add_argument('--pred_dir', type=str, default='ns_result/', help='Directory where predictions stored.')
    parser.add_argument('--max_depth', type=float, default=60.0, help='Maximum depth value.')
    parser.add_argument('--min_depth', type=float, default=1e-5, help='Minimum depth value.')
    parser.add_argument('--output_file_name', type=str, default=None, help='File name for saving result.')

    return parser.parse_args()


def read_gt():
    result = []
    # print(osp.join(NUSCENES_ROOT['split'], '{}_test_split.txt'.format(args.weather)))
    test_items = read_list_from_file(osp.join("../", NUSCENES_ROOT['split'], '{}_test_split.txt'.format(args.weather)))
    test_items = sorted(test_items)
    for item in test_items:
        depth = np.load(osp.join("../", NUSCENES_ROOT['test_gt'], item + '.npy'))
        result.append(depth)
    result = np.stack(result, axis=0)
    return result


if __name__ == '__main__':
    args = parse_args()
    assert 0.0 < args.min_depth < 1.0
    gt_depth = read_gt()
    pred_depth = np.load(os.path.join(args.pred_dir, 'predictions.npy'))
    evaluate()
