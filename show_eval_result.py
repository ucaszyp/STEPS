import argparse
import json
import os

import numpy as np

from ui import PyTable


def compute_top_result(items, headers, sort_key):
    # compute final result, pick up top half
    reverse = sort_key in ['a1', 'a2', 'a3']
    items = sorted(items, key=lambda x: x[sort_key], reverse=reverse)
    num_items = len(items)
    num_pick = num_items // 2
    if num_items > 1:
        items = items[: num_pick]
    mean = {'epoch': 'top{}'.format(num_pick)}
    # compute mean
    for k in headers:
        if k != 'epoch':
            vals = [item[k] for item in items]
            mean[k] = '{:.4f}'.format(np.mean(vals))
    return mean


def compute_mean_result(items, headers):
    mean = {'epoch': 'avg'}
    # compute mean
    for k in headers:
        if k != 'epoch':
            vals = [item[k] for item in items]
            mean[k] = '{:.4f}'.format(np.mean(vals))
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tgt_dir', type=str, help='Target directory to load evaluation data.')
    parser.add_argument('--title', type=str, default='Evaluation Result', help='Title of table.')
    args = parser.parse_args()

    # create table
    table = PyTable(['epoch', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3'], args.title)
    # key for sorting
    sort_key = 'rmse'
    # read items
    items = []
    results = []
    for f in os.scandir(args.tgt_dir):
        if f.is_file() and f.name.endswith('.json'):
            with open(f.path, 'r') as fo:
                json_item = json.load(fo)
                items.append(json_item)
                item = {'epoch': f.name[:-5].split('_')[-1]}
                for k, v in json_item.items():
                    item[k] = '{:.4f}'.format(v)
                results.append(item)
    # compute mean and top result
    top_result = compute_top_result(items[:], table.headers, sort_key)
    mean_result = compute_mean_result(items[:], table.headers)
    # generate table
    results = sorted(results, key=lambda x: int(x['epoch']))
    for item in results:
        table.add_item(item)
    # add mean
    table.add_split_line()
    table.add_item(mean_result)
    table.add_item(top_result)
    # print
    table.print_table()


if __name__ == '__main__':
    main()
