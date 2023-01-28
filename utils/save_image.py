import matplotlib.pyplot as plt
import numpy as np
import cv2


def save_color_disp_separately(color: np.ndarray, disp: np.ndarray, fn: str, max_p: int = 100,
                               disp_cmap: str = 'magma'):
    # preprocess
    vmax = np.percentile(disp, max_p)
    # save
    plt.imsave(fn + '_color.png', color)
    plt.imsave(fn + '_disp.png', disp, cmap=disp_cmap, vmax=vmax)


def save_colors_separately(in_color: np.ndarray, out_color: np.ndarray, fn: str):
    cv2.imwrite(fn + '_input.png', in_color)
    cv2.imwrite(fn + '_output.png', out_color)
