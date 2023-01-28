import numpy as np

from sklearn.decomposition import PCA


def feature_decomposition(feature: np.ndarray, out_channels: int, in_channel_order='last', out_channel_order='last'):
    """
    Decompose feature to given channels
    :param feature:
    :param out_channels:
    :param in_channel_order: channel's order of input, 'first' or 'last'
    :param out_channel_order: channel's order of output, 'first' or 'last'
    :return:
    """
    # check params
    assert feature.ndim == 3, 'The number of dimension of input feature must be 3, i.e. (C, H, W).'
    assert in_channel_order in ['first', 'last'], 'Unknown channel order: {}.'.format(in_channel_order)
    assert out_channel_order in ['first', 'last'], 'Unknown channel order: {}.'.format(out_channel_order)
    # handle input feature
    if in_channel_order == 'first':
        feature = np.moveaxis(feature, 0, -1)
    # get shape
    h, w, c = feature.shape
    # pca
    pca = PCA(out_channels)
    # reshape, to (h * w, c)
    x = feature.reshape((-1, c))
    # decomposition, to (h * w, out_channels)
    y = pca.fit_transform(x)
    y = np.squeeze(y)
    # reshape and return
    if y.ndim == 1:
        return y.reshape((h, w))
    else:
        y = y.reshape((h, w, out_channels))
        if out_channel_order == 'first':
            y = np.moveaxis(y, -1, 0)
        return y
