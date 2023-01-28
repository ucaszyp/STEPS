import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from utils import read_list_from_file
from transforms import ResizeWithIntrinsic, RandomHorizontalFlipWithIntrinsic, EqualizeHist, CenterCropWithIntrinsic
from torchvision.transforms import ToTensor
from .common import NUSCENES_ROOT


# full size
_FULL_SIZE = (768, 384)
# half size
_HALF_SIZE = (384, 192)
# limit of equ
_EQU_LIMIT = 0.004
# robotcar size
_ROBOTCAR_SIZE = (576, 320)


#
# Data set
#
class nuScenesSequence(Dataset):
    """
    Oxford RobotCar data set.
    """
    def __init__(self, weather, frame_ids: (list, tuple), augment=True, down_scale=False, num_out_scales=5,
                 gen_equ=False, equ_limit=_EQU_LIMIT, resize=False):
        """
        Initialize
        :param weather: day or night
        :param frame_ids: index of frames
        :param augment: whether to augment
        :param down_scale: whether to down scale images to half of that before
        :param num_out_scales: number of output scales
        :param gen_equ: whether to generate equ image
        :param equ_limit: limit of equ
        :param resize: whether to resize to the same size as robotcar
        """
        # set parameters
        self._root_dir = NUSCENES_ROOT['sequence']
        # self._adv_dir = "/data2/zyp/CARLA_EPE/depth"
        self._frame_ids = frame_ids
        self._need_augment = augment
        self._num_out_scales = num_out_scales
        self._gen_equ = gen_equ
        self._equ_limit = equ_limit
        self._need_resize = resize
        self._down_scale = down_scale and (not resize)
        if self._down_scale:
            self._width, self._height = _HALF_SIZE
        else:
            self._width, self._height = _FULL_SIZE
        # read all chunks
        if weather in ['day', 'night']:
            chunks = self.read_chunks(os.path.join(NUSCENES_ROOT['split'], '{}_train_split.txt'.format(weather)))
        elif weather == 'both':
            day_chunks = self.read_chunks(os.path.join(NUSCENES_ROOT['split'], 'day_train_split.txt'))
            night_chunks = self.read_chunks(os.path.join(NUSCENES_ROOT['split'], 'night_train_split.txt'))
            chunks = day_chunks + night_chunks
        else:
            raise ValueError(f'Unknown weather parameter: {weather}.')
        # get sequence
        self._sequence_items = self.make_sequence(chunks)
        # transforms
        self._to_tensor = ToTensor()
        if self._need_augment:
            self._flip = RandomHorizontalFlipWithIntrinsic(0.5)
        # crop
        if self._need_resize:
            self._crop = CenterCropWithIntrinsic(round(1.8 * self._height), self._height)
        else:
            self._crop = None
        # resize
        if self._down_scale:
            self._resize = ResizeWithIntrinsic(*_HALF_SIZE)
        elif self._need_resize:
            self._resize = ResizeWithIntrinsic(*_ROBOTCAR_SIZE)
        else:
            self._resize = None
        # print message
        # self._adv_list = sorted(os.listdir(self._adv_dir))
        print('Frames: {}, Augment: {}, DownScale: {}, '
              'Equ_Limit: {}.'.format(frame_ids, augment, self._down_scale, self._equ_limit))
        # print('Adv_depth items: {}.'.format(len(self._adv_list)))
        print('Total items: {}.'.format(len(self)))

    def read_chunks(self, split_file):
        result = []
        scenes = read_list_from_file(split_file, 1)
        for scene in scenes:
            scene_path = os.path.join(self._root_dir, scene)
            colors = sorted(read_list_from_file(os.path.join(scene_path, 'file_list.txt'), 1))
            colors = [os.path.join(scene, color) for color in colors]
            chunk = {
                'colors': colors,
                'k': np.load(os.path.join(scene_path, 'intrinsic.npy'))
            }
            result.append(chunk)
        return result

    def pack_data(self, src_colors: dict, src_K: np.ndarray, num_scales: int):
        out = {}
        h, w, _ = src_colors[0].shape
        # Note: the numpy ndarray and tensor share the same memory!!!
        src_K = torch.from_numpy(src_K)
        src_K = src_K.to(torch.float32)
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
                # to tensor
                color = self._to_tensor(color)
                color_gray = self._to_tensor(color_gray)
                # resize
                if s != 0:
                    color = F.interpolate(color.unsqueeze(0), (rh, rw), mode='area').squeeze(0)
                    color_gray = F.interpolate(color_gray.unsqueeze(0), (rh, rw), mode='area').squeeze(0)
                # (name, frame_idx, scale)
                out['color', fi, s] = color
                out['color_aug', fi, s] = color
                out['color_gray', fi, s] = color_gray
        # adv_depth = self._to_tensor(adv_depth)
        # out['adv_depth', 0, 0] = adv_depth
                # if self._gen_equ:
                #     out['color_equ', fi, s] = equ_color
                   
        # print("===========K & inv_K============")
        # print(out['K', 0])
        # print(out['inv_K', 0])

        return out

    def make_sequence(self, chunks: (list, tuple)):
        """
        Make sequence from given folders
        :param chunks:
        :return:
        """
        # store items
        result = []
        # scan
        for chunk in chunks:
            fs = chunk['colors']
            # get length
            frame_length = len(self._frame_ids)
            min_id, max_id = min(self._frame_ids), max(self._frame_ids)
            total_length = len(fs)
            if total_length < frame_length:
                continue
            # pick sequence
            for i in range(abs(min_id), total_length - abs(max_id)):
                items = [fs[i + fi] for fi in self._frame_ids]
                result.append({'sequence': items, 'k': chunk['k']})
        # return
        return result

    def __getitem__(self, idx):
        """
        Return item according to given index
        :param idx: index
        :return:
        """
        # get item
        item = self._sequence_items[idx]
        # read data
        rgbs = [cv2.imread(os.path.join(self._root_dir, p)) for p in item['sequence']]

        # index = idx % len(self._adv_list)
        # adv_depth = cv2.imread(os.path.join(self._adv_dir, self._adv_list[index]))
        # adv_depth = adv_depth[:,:,0]
        # adv_depth = cv2.resize(adv_depth, (960, 540))
        # adv_depth = adv_depth[78:462, 96:864]
        # adv_depth = np.clip(adv_depth, 1.0, 100.0)
        # adv_depth = 1 / adv_depth

        intrinsic = item['k'].copy()
        # crop
        if self._crop is not None:
            intrinsic, rgbs = self._crop(intrinsic, *rgbs, inplace=False, unpack=False)
        # down scale
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
