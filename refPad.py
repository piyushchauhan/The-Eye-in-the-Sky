import math
import numpy as np
"""
# a[:,:,0]
array([[ 0,  4,  8, 12],
       [16, 20, 24, 28],
       [32, 36, 40, 44],
       [48, 52, 56, 60]])
# ans = refPad(a, 3)
# ans[:,:,0]
array([[ 0,  4,  8, 12,  8,  4],
       [16, 20, 24, 28, 24, 20],
       [32, 36, 40, 44, 40, 36],
       [48, 52, 56, 60, 56, 52],
       [32, 36, 40, 44, 40, 36],
       [16, 20, 24, 28, 24, 20]])
"""
def refPad(x, patch_sz):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width,
                            n_channels), dtype=x.dtype)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 2, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 2, :]
    return ext_x
