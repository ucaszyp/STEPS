import cv2
import random
import numpy as np

from typing import List


#
# transforms
#
class RandomCrop:
    """
    Random crop
    """

    def __init__(self, *size):
        self.w, self.h = size

    def __call__(self, *imgs, inplace=False):
        """
        Perform random crop
        :param imgs: images to process
        :param inplace: whether to perform random crop in place
        """
        # assert
        assert len(imgs) > 0
        # rgb shape
        h, w, _ = imgs[0].shape
        # h range
        h_start = random.randint(0, w - self.w)
        h_end = h_start + self.w
        # v range
        v_start = random.randint(0, h - self.h)
        v_end = v_start + self.h
        # store results
        results = []
        # process others
        for img in imgs:
            # img result
            if inplace:
                img_result = img[v_start: v_end, h_start: h_end]
            else:
                img_result = img[v_start: v_end, h_start: h_end].copy()
            # append to result
            results.append(img_result)
        # handle results
        results = results if len(results) > 1 else results[0]
        # return
        return results


class RandomCropWithMargin:
    """
    Random crop a area within a margin
    """

    def __init__(self, size, margin):
        """
        Initialize
        :param size: crop size
        :param margin: crop margin(left, right, top, bottom)
        """
        self.w, self.h = size
        self.m_l, self.m_r, self.m_t, self.m_b = margin

    def __call__(self, rgb, *others, inplace=False):
        """
        Process
        :param rgb: rgb image
        :param others: other images need to be crop
        :param inplace: whether to process image in place
        :return:
        """
        # get rgb shape
        rgb_h, rgb_w, rgb_c = rgb.shape
        # rgb must be 'channel last'
        assert rgb_c == 3
        # randomly select h range
        h_start = random.randint(self.m_l, rgb_w - self.m_l - self.m_r - self.w)
        h_end = h_start + self.w
        # randomly select v range
        v_start = random.randint(self.m_t, rgb_h - self.m_t - self.m_b - self.h)
        v_end = v_start + self.h
        # rgb result
        if inplace:
            rgb_result = rgb[v_start: v_end, h_start: h_end, :]
        else:
            rgb_result = rgb[v_start: v_end, h_start: h_end, :].copy()
        # store results
        results = [rgb_result]
        # process others
        for img in others:
            # get shape
            img_h, img_w = img.shape[:2]
            # img and rgb must be in same size
            assert rgb_h == img_h and rgb_w == img_w
            # img result
            if inplace:
                img_result = img[v_start: v_end, h_start: h_end]
            else:
                img_result = img[v_start: v_end, h_start: h_end].copy()
            # append to result
            results.append(img_result)
        # handle results
        results = results if len(results) > 1 else results[0]
        # return
        return results


class CenterCrop:
    """
    Center crop a given image
    """

    def __init__(self, *size):
        """
        Initialize
        :param width: crop width
        :param height: crop height
        """
        self.w, self.h = size

    def __call__(self, *imgs, inplace=False, unpack=True):
        """
        Process
        :param imgs: 'channel last' images to process
        :param inplace: whether to process image in place
        :param unpack:
        :return:
        """
        # assert
        assert len(imgs) > 0
        # shape
        h, w = imgs[0].shape[:2]
        # h range
        h_start = max((w - self.w) // 2, 0)
        h_end = min(h_start + self.w, w)
        # v range
        v_start = max((h - self.h) // 2, 0)
        v_end = min(v_start + self.h, h)
        # store results
        results = []
        # process others
        for img in imgs:
            # img result
            if inplace:
                img_result = img[v_start: v_end, h_start: h_end]
            else:
                img_result = img[v_start: v_end, h_start: h_end].copy()
            # append to result
            results.append(img_result)
        # handle results
        if unpack and len(results) == 1:
            results = results[0]
        # return
        return results


class RandomHorizontalFlip:
    """
    Random horizontal flip
    """

    def __init__(self, p):
        """
        Initialize
        :param p: probability
        """
        self.p = p

    def __call__(self, *images):
        """
        Process
        :param images: 'channel last' images to be processed
        """
        # assert
        assert len(images) > 0
        # store results
        results = []
        # process
        if random.random() < self.p:
            for img in images:
                results.append(img[:, ::-1].copy())
        else:
            results = images
        # handle results
        results = results if len(results) > 1 else results[0]
        # return
        return results


class RandomHorizontalFlipWithIntrinsic:
    """
    Random horizontal flip with intrinsic
    """

    def __init__(self, p):
        """
        Initialize
        :param p: probability
        """
        self.p = p

    def __call__(self, intrinsic, *imgs, unpack=True):
        """
        Process
        :param intrinsic: intrinsic matrix
        :param imgs: 'channel last' images to be processed
        :param unpack:
        """
        # assert
        assert len(imgs) > 0
        # store results
        img_results = []
        # get w
        h, w, _ = imgs[0].shape
        # process
        if random.random() < self.p:
            for img in imgs:
                img_results.append(img[:, ::-1].copy())
            intrinsic_result = intrinsic.copy()
            intrinsic_result[0, 2] = w - intrinsic[0, 2]
        else:
            img_results = imgs
            intrinsic_result = intrinsic
        # handle results
        if unpack and len(img_results) == 1:
            img_results = img_results[0]
        # return
        return intrinsic_result, img_results


class CenterCropWithIntrinsic:
    """
    Center crop images along with intrinsic
    """

    def __init__(self, *size):
        """
        Initialize
        :param size: image size after crop
        """
        self._w, self._h = size

    def __call__(self, intrinsic, *imgs, inplace=False, unpack=True):
        """
        Process
        :param intrinsic:
        :param imgs:
        :param unpack:
        :return:
        """
        # assert
        assert len(imgs) > 0
        # transform
        crop = CenterCrop(self._w, self._h)
        # size of target image
        h, w = imgs[0].shape[:2]
        # compute margin
        m_x = (w - self._w) / 2.0
        m_y = (h - self._h) / 2.0
        # check
        assert m_x >= 0 and m_y >= 0
        # crop
        imgs = crop(*imgs, inplace=inplace, unpack=unpack)
        intrinsic = intrinsic.copy()
        intrinsic[0, 2] -= m_x
        intrinsic[1, 2] -= m_y
        # return
        return intrinsic, imgs


class RandomRescaleCrop:
    """
    Randomly rescale and crop image to keep same size as before
    """

    def __init__(self, out_size, min_scale_factor=0.8, max_scale_factor=1.2):
        """
        Initialize
        :param out_size: output size, (w, h)
        :param min_scale_factor: min scale factor
        :param max_scale_factor: max scale factor
        """
        # check
        assert 0.2 <= min_scale_factor < 1.5
        assert 1.01 <= max_scale_factor < 1.5
        assert min_scale_factor < max_scale_factor
        # parameters
        self._out_size = out_size
        self._min_scale_factor = min_scale_factor
        self._max_scale_factor = max_scale_factor

    def __call__(self, *imgs, interpolation='linear', crop_mode='random', inplace=False):
        """
        Process
        :param imgs: input images with 3 color channels
        :param interpolation: interpolation mode to rescale image
        :param crop_mode: center or random crop
        :return:
        """
        # assert
        assert len(imgs) > 0
        # get original size
        h, w, _ = imgs[0].shape
        # crop
        if crop_mode == 'center':
            crop = CenterCrop(*self._out_size)
        elif crop_mode == 'random':
            crop = RandomCrop(*self._out_size)
        else:
            raise ValueError('Unsupported crop mode: {}.'.format(crop_mode))
        # output size
        o_h = random.randint(round(h * self._min_scale_factor), round(h * self._max_scale_factor))
        o_w = random.randint(round(w * self._min_scale_factor), round(w * self._max_scale_factor))
        # resize
        if interpolation == 'linear':
            resize_mode = cv2.INTER_LINEAR
        elif interpolation == 'nearest':
            resize_mode = cv2.INTER_NEAREST
        else:
            raise ValueError('Unsupported interpolation mode: {}.'.format(interpolation))
        result = [cv2.resize(img, (o_w, o_h), interpolation=resize_mode) for img in imgs]
        # crop
        result = crop(*result, inplace=inplace)
        # return
        return result


class RandomRescaleCropWithIntrinsic:
    """
    Randomly rescale and center crop image to keep same size as before, along with intrinsic
    """

    def __init__(self, out_size, min_scale_factor=0.8, max_scale_factor=1.2):
        """
        Initialize
        :param out_size: output size, (w, h)
        :param min_scale_factor: min scale factor
        :param max_scale_factor: max scale factor
        """
        # check
        assert 0.2 <= min_scale_factor < 1.5
        assert 1.01 <= max_scale_factor < 1.5
        assert min_scale_factor < max_scale_factor
        # parameters
        self._out_size = out_size
        self._min_scale_factor = min_scale_factor
        self._max_scale_factor = max_scale_factor

    def __call__(self, intrinsic, *imgs, interpolation='nearest', inplace=False):
        """
        Process
        :param intrinsic: intrinsic matrix
        :param imgs: input images with 3 color channels
        :param interpolation: interpolation mode to rescale image
        :param crop_mode: center or random crop
        :return:
        """
        # assert
        assert len(imgs) > 0
        # get original size
        h, w, _ = imgs[0].shape
        # crop
        crop = CenterCrop(*self._out_size)
        # output size
        o_h = random.randint(round(h * self._min_scale_factor), round(h * self._max_scale_factor))
        o_w = random.randint(round(w * self._min_scale_factor), round(w * self._max_scale_factor))
        # resize
        if interpolation == 'linear':
            resize_mode = cv2.INTER_LINEAR
        elif interpolation == 'nearest':
            resize_mode = cv2.INTER_NEAREST
        else:
            raise ValueError('Unsupported interpolation mode: {}.'.format(interpolation))
        img_result = [cv2.resize(img, (o_w, o_h), interpolation=resize_mode) for img in imgs]
        # crop
        img_result = crop(*img_result, inplace=inplace)
        # compute intrinsic result
        x_scale = o_w / w
        y_scale = o_h / h
        t_w, t_h = self._out_size
        x_offset = (o_w - t_w) / 2.0
        y_offset = (o_h - t_h) / 2.0
        intrinsic_result = intrinsic.copy()
        intrinsic_result[0, :] *= x_scale
        intrinsic_result[1, :] *= y_scale
        intrinsic_result[0, 2] -= x_offset
        intrinsic_result[1, 2] -= y_offset
        # return
        return intrinsic_result, img_result


class RandomHorizontalFlipWithNormalIntrinsic:
    """
    Randomly horizontal flip with normal and intrinsic
    """

    def __init__(self, p):
        """
        Initialize
        :param p: probability
        """
        # assert
        assert 0 <= p <= 1
        # parameters
        self._p = p

    @staticmethod
    def horizontal_flip_rgb(img: np.ndarray):
        """
        Horizontally flip a given image.
        :param img: channel last image
        :return:
        """
        # channel last
        result = img[:, ::-1, :].copy()
        # return
        return result

    @staticmethod
    def horizontal_flip_normal(normal: np.ndarray):
        """
        Horizontally flip a normal map
        :param normal: channel first normal map
        :return:
        """
        result = normal[:, :, ::-1].copy()
        result[0, :, :] *= -1.0
        return result

    def __call__(self, rgb: np.ndarray, normal: np.ndarray, intrinsic: np.ndarray):
        """
        Process
        :param rgb: 'channel-last' rgb image
        :param normal: 'channel-first' normal image
        :param intrinsic: intrinsic matrix
        :return:
        """
        # assert
        assert rgb.shape[-1] == 3 and normal.shape[0] == 3
        # processing
        if random.random() < self._p:
            # get width
            w = rgb.shape[1]
            # flip image
            rgb = self.horizontal_flip_rgb(rgb)
            # flip normal
            normal = self.horizontal_flip_normal(normal)
            # flip intrinsic
            intrinsic = intrinsic.copy()
            intrinsic[0, 2] = w - intrinsic[0, 2]
            return rgb, normal, intrinsic
        else:
            return rgb, normal, intrinsic


class ResizeWithIntrinsic:
    """
    Resize given images with intrinsic matrix
    """

    def __init__(self, *size):
        self._w, self._h = size

    def __call__(self, intrinsic: np.ndarray, *imgs):
        # assert
        assert len(imgs) > 0
        # get shape
        h, w = imgs[0].shape[:2]
        # get scale factor
        factor_x = self._w / w
        factor_y = self._h / h
        # process
        result_imgs = [cv2.resize(img, (self._w, self._h), interpolation=cv2.INTER_LINEAR) for img in imgs]
        K = intrinsic.copy()
        K[0, :] *= factor_x
        K[1, :] *= factor_y
        # return
        return K, result_imgs


def color_equalize_hist(color: np.ndarray):
    b, g, r = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    b = cv2.equalizeHist(b)
    g = cv2.equalizeHist(g)
    r = cv2.equalizeHist(r)
    return np.stack([b, g, r], axis=-1)


class EqualizeHist:
    def __init__(self, tgt_img: np.ndarray, limit=0.02):
        """
        Get color map from given RGB image
        :param tgt_img: target image
        :param limit: limit of density
        """
        self._limit = limit
        self._color_map = [self.get_color_map(tgt_img[:, :, i]) for i in range(3)]

    def get_color_map(self, img: np.ndarray):
        assert img.dtype == np.uint8
        # get shape
        h, w = img.shape
        num_pixels = h * w
        # get hist
        hist, _ = np.histogram(img.flatten(), 256, [0, 256])
        limit_pixels = int(num_pixels * self._limit)
        # get number of overflow and clip
        num_overflow = np.sum(np.clip(hist - limit_pixels, a_min=0, a_max=None))
        hist = np.clip(hist, a_min=0, a_max=limit_pixels)
        # add
        hist += np.round(num_overflow / 256.0).astype(np.int)
        # get cdf
        cdf = hist.cumsum()
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
        # return
        return cdf

    def __call__(self, img: np.ndarray):
        chs = [self._color_map[i][img[:, :, i]] for i in range(3)]
        return np.stack(chs, axis=-1)
