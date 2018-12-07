import math
import numpy as np

def rowslice(image, gtimage, ptsz):
    n = math.floor(image.shape[1]/(ptsz//2))
    w = image.shape[1]
    rarr = range(n+1)
    X_row = []
    Y_row = []
    for i in rarr[:n-1]:
        X_row += [image[:, (ptsz//2)*i:(ptsz//2)*(i+2), :]]
        Y_row += [gtimage[:, (ptsz//2)*i:(ptsz//2)*(i+2), :]]
    if image.shape[1] % (ptsz//2) != 0:
        X_row += [image[:, w-ptsz:, :]]
        Y_row += [gtimage[:, w-ptsz:, :]]
    
    return np.array(X_row), np.array(Y_row)


def slice(image, gtimage, ptsz):
    assert image.shape == gtimage.shape, "Image shapes of the given images do not match"
    m = math.floor(image.shape[0]/(ptsz//2))
    h = image.shape[0]
    rarr = range(m+1)
    X = []
    Y = []
    for i in rarr[:m-1]:
        [X_row, Y_row] = rowslice(image[(ptsz//2)*i:(ptsz//2)*(i+2), :, :], gtimage[(ptsz//2)*i:(ptsz//2)*(i+2), :, :], ptsz)
        for X_ in X_row:
            X += [X_]
        for Y_ in Y_row:
            Y += [Y_]
    if h % (ptsz//2) != 0:
        [X_row, Y_row] = rowslice(image[h-ptsz:, :, :], gtimage[h-ptsz:, :, :], ptsz)
        for X_ in X_row:
            X += [X_]
        for Y_ in Y_row:
            Y += [Y_]
    return np.array(X), np.array(Y)
