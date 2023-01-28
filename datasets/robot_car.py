import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from utils import read_list_from_file
from transforms import ResizeWithIntrinsic, RandomHorizontalFlipWithIntrinsic, CenterCropWithIntrinsic, EqualizeHist
from torchvision.transforms import ToTensor
from .common import ROBOTCAR_ROOT

# rgb extension name
_RGB_EXT = '.png'
# crop size
_CROP_SIZE = (1152, 640)
# half size
_HALF_SIZE = (768, 384)
# limit of equ
_EQU_LIMIT = 0.008
# nuscenes size
_NUSCENES_SIZE = (768, 384)


#
# Utils
#
def read_chunks(fn, drt):
    """
    Get all files from list file
    """
    # get all items
    items = read_list_from_file(fn, 1)
    # process
    result = []
    chunk = []
    for item in items:
        if item.startswith('-----'):
            result.append(chunk[:])
            chunk.clear()
        else:
            chunk.append(os.path.join(drt, item))
    if len(chunk) > 0:
        result.append(chunk[:])
    # return
    return result


#
# Data set
#
class RobotCarSequence(Dataset):
    """
    Oxford RobotCar data set.
    """

    def __init__(self, root_dir, frame_ids: (list, tuple), augment=True, down_scale=False, num_out_scales=5,
                 gen_equ=False, equ_limit=_EQU_LIMIT, resize=False):
        """
        Initialize
        :param root_dir: root directory
        :param frame_ids: index of frames
        :param augment: whether to augment
        :param down_scale: whether to down scale images to half of that before
        :param num_out_scales: number of output scales
        :param gen_equ: whether to generate equ image
        :param equ_limit: limit of equ
        :param resize: whether to resize to the same size as nuscenes
        """
        # assert
        assert len(frame_ids) % 2 == 1
        # set parameters
        # self._root_dir = ROBOTCAR_ROOT[root_dir] if root_dir in ROBOTCAR_ROOT else root_dir
        self._frame_ids = frame_ids
        self._need_augment = augment
        self._num_out_scales = num_out_scales
        self._gen_equ = gen_equ
        self._equ_limit = equ_limit if equ_limit is not None else _EQU_LIMIT
        self._down_scale = down_scale and (not resize)
        if self._down_scale:
            self._width, self._height = _HALF_SIZE
        else:
            self._width, self._height = _CROP_SIZE
        self._need_resize = resize
        # read all chunks
        if root_dir in ['day', 'night']:
            drt = ROBOTCAR_ROOT[root_dir]
            chunks = read_chunks(os.path.join(drt, 'train_split.txt'), os.path.join(drt, 'rgb/'))
        elif root_dir == 'both':
            day_chunks = read_chunks(os.path.join(ROBOTCAR_ROOT['day'], 'train_split.txt'),
                                     os.path.join(ROBOTCAR_ROOT['day'], 'rgb/'))
            night_chunks = read_chunks(os.path.join(ROBOTCAR_ROOT['night'], 'train_split.txt'),
                                       os.path.join(ROBOTCAR_ROOT['night'], 'rgb/'))
            chunks = day_chunks + night_chunks
        else:
            raise ValueError(f'Unknown root_dir: {root_dir}.')
        # read intrinsic
        self._k = self.load_intrinsic()
        # get sequence
        self._sequence_items = self.make_sequence(chunks)
        # transforms
        self._to_tensor = ToTensor()
        if self._need_augment:
            self._flip = RandomHorizontalFlipWithIntrinsic(0.5)
        # crop
        if self._need_resize:
            self._crop = CenterCropWithIntrinsic(self._width, self._width // 2)
        else:
            self._crop = CenterCropWithIntrinsic(*_CROP_SIZE)
        # resize
        if self._down_scale:
            self._resize = ResizeWithIntrinsic(*_HALF_SIZE)
        elif self._need_resize:
            self._resize = ResizeWithIntrinsic(*_NUSCENES_SIZE)
        else:
            self._resize = None
        # print message
        print('Root: {}, Frames: {}, Augment: {}, DownScale: {}, '
              'Equ_Limit: {}.'.format(root_dir, frame_ids, augment, self._down_scale, self._equ_limit))
        print('Total items: {}.'.format(len(self)))

    def load_intrinsic(self):
        """
        Load and parse intrinsic matrix
        :return:
        """
        src_k = np.load(os.path.join(ROBOTCAR_ROOT['night'], 'intrinsic.npy')).astype(np.float32)
        fx, cx = src_k[0, 0], src_k[0, 2]
        fy, cy = src_k[1, 1], src_k[1, 2]
        intrinsic = np.array([
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32)
        return intrinsic

    def pack_data(self, src_colors: dict, src_K: np.ndarray, num_scales: int):
        out = {}
        h, w, _ = src_colors[0].shape
        # Note: the numpy ndarray and tensor share the same memory!!!
        src_K = torch.from_numpy(src_K)
        # transform
        # equ_hist = EqualizeHist(src_colors[0], limit=self._equ_limit)
        # process
        for s in range(num_scales):
            # get size
            rh, rw = h // (2 ** s), w // (2 ** s)
            # K and inv_K
            K = src_K.clone()
            if s != 0:
                K[0, :] = K[0, :] * rw / w
                K[1, :] = K[1, :] * rh / h
            out['K', s] = K
            out['inv_K', s] = torch.inverse(K)
            # color
            for fi in self._frame_ids:
                # get color
                color = src_colors[fi]
                color_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                # equ_color = equ_hist(color)
                # to tensor
                color = self._to_tensor(color)
                # equ_color = self._to_tensor(equ_color)
                color_gray = self._to_tensor(color_gray)
                # resize
                if s != 0:
                    color = F.interpolate(color.unsqueeze(0), (rh, rw), mode='area').squeeze(0)
                    # equ_color = F.interpolate(equ_color.unsqueeze(0), (rh, rw), mode='area').squeeze(0)
                    color_gray = F.interpolate(color_gray.unsqueeze(0), (rh, rw), mode='area').squeeze(0)
                # (name, frame_idx, scale)
                out['color', fi, s] = color
                out['color_aug', fi, s] = color
                out['color_gray', fi, s] = color_gray
                # if self._gen_equ:
                #     out['color_equ', fi, s] = equ_color
        return out

    def make_sequence(self, chunks: (list, tuple)):
        """
        Make sequence from given folders
        :param chunks:
        :return:
        """
        # store items
        items = []
        # scan
        for fs in chunks:
            # get length
            frame_length = len(self._frame_ids)
            min_id, max_id = min(self._frame_ids), max(self._frame_ids)
            total_length = len(fs)
            if total_length < frame_length:
                continue
            # pick sequence
            for i in range(abs(min_id), total_length - abs(max_id)):
                item = [fs[i + fi] for fi in self._frame_ids]
                items.append(item)
        # return
        return items

    def __getitem__(self, idx):
        """
        Return item according to given index
        :param idx: index
        :return:
        """
        # get item
        item = self._sequence_items[idx]
        # read data
        # rgbs = [cv2.imread(os.path.join(self._root_dir, 'rgb/', '{}.png'.format(p))) for p in item]
        rgbs = [cv2.imread(p + '.png') for p in item]
        for rgb, path in zip(rgbs, item):
            assert rgb is not None, f'{path} reads None.'
        intrinsic = self._k.copy()
        # crop
        intrinsic, rgbs = self._crop(intrinsic, *rgbs, inplace=False, unpack=False)
        # rescale
        if self._resize is not None:
            intrinsic, rgbs = self._resize(intrinsic, *rgbs)
        # augment
        if self._need_augment:
            intrinsic, rgbs = self._flip(intrinsic, *rgbs, unpack=False)
        # get colors
        colors = {}
        # color
        for i, fi in enumerate(self._frame_ids):
            colors[fi] = rgbs[i]
        # pack
        result = self.pack_data(colors, intrinsic, self._num_out_scales)
        # return
        return result

    def __len__(self):
        return len(self._sequence_items)