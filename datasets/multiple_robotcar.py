import random

from torch.utils.data import Dataset

from .robot_car import RobotCarSequence


class MultipleRobotCar(Dataset):
    """
    Dataset that handles multiple robotcar subset
    """
    def __init__(self, subsets: list, frame_ids: (list, tuple), master: str, augment=True, down_scale=False,
                 num_out_scales=1, gen_equ=False, shuffle=False, **kwargs):
        """
        Initialize
        :param subsets: params for sets
        :param frame_ids:
        :param augment:
        :param down_scale:
        :param shuffle: if shuffle is true, then the return order of non-master subset items is not fixed.
        :param kwargs: other keyword parameters
        """
        # params
        self._shuffle = shuffle
        self._master = master
        self._orders = {}
        # prepare dataset
        equ_limit = kwargs.get('equ_limit', None)
        sets = {}
        for s in subsets:
            sets[s['name']] = RobotCarSequence(s['root_dir'], frame_ids, augment, down_scale, num_out_scales, gen_equ,
                                               equ_limit=equ_limit)
        self._subsets = sets
        # compute data length
        self._data_len = {k: len(v) for k, v in sets.items()}
        # shuffle
        if shuffle:
            self.make_orders()
        # print message
        print('Master: {}, Shuffle: {}, Length: {}.'.format(self._master, self._shuffle, len(self)))

    def make_orders(self):
        master_len = self._data_len[self._master]
        for name, length in self._data_len.items():
            if name != self._master:
                ids = list(range(length)) if length <= master_len else random.sample(range(length), master_len)
                random.shuffle(ids)
                self._orders[name] = ids

    def when_epoch_over(self):
        """
        Make sure this function be executed after every epoch, otherwise the pair combination would be fixed.
        :return:
        """
        if self._shuffle:
            self.make_orders()

    def __getitem__(self, idx):
        item = {}
        for name, subset in self._subsets.items():
            if name == self._master:
                item[name] = subset[idx]
            else:
                # the index for other subset
                sub_idx = idx % self._data_len[name]
                if self._shuffle:
                    item[name] = subset[self._orders[name][sub_idx]]
                else:
                    item[name] = subset[sub_idx]
        return item

    def __len__(self):
        return self._data_len[self._master]
