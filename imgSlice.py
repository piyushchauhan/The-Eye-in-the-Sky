import math
import numpy as np
import logging
import sys
logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s- %(message)s')
# logging.disable(sys.maxsize)

def rowslice(image, gtimage, ptsz):
    n = math.floor(image.shape[1]/(ptsz//2))
    w = int(gtimage.shape[1])
    rarr = range(n+1)
    X_row = np.array([]).reshape((0,ptsz,ptsz,image.shape[2]))
    Y_row = np.array([]).reshape((0,ptsz,ptsz,gtimage.shape[2]))
    for i in rarr[:n-1]:
        # logging.debug("     {}: ({}:{})".format(i,i*ptsz//2, (i+2)*ptsz//2))
        X_row = np.append(X_row,np.array([image[:, (ptsz//2)*i:(ptsz//2)*(i+2), :]]), axis=0)
        Y_row = np.append(Y_row,np.array([gtimage[:, (ptsz//2)*i:(ptsz//2)*(i+2), :]]), axis=0)
    if image.shape[1] % (ptsz//2) != 0:
        X_row = np.append(X_row,np.array([image[:, w-ptsz:, :]]), axis=0)
        Y_row = np.append(Y_row,np.array([gtimage[:, w-ptsz:, :]]), axis=0)   
    return np.array(X_row), np.array(Y_row)


def imgSlice(image, gtimage, ptsz):
    assert image.shape[:2] == gtimage.shape[:2], "Image shapes of the given images do not match"
    h = int(gtimage.shape[0])
    m = math.floor(image.shape[0]/(ptsz//2))
    rarr = range(m+1)
    X = np.array([]).reshape((0,ptsz,ptsz,image.shape[2]))
    Y = np.array([]).reshape((0,ptsz,ptsz,gtimage.shape[2]))
    for i in rarr[:m-1]:
        # logging.debug("{}:->".format(i+1))
        X_row, Y_row = rowslice(image[(ptsz//2)*i:(ptsz//2)*(i+2), :, :], gtimage[(ptsz//2)*i:(ptsz//2)*(i+2), :, :], ptsz)
        X = np.append(X,X_row,axis=0)
        Y = np.append(Y,Y_row,axis=0)

    if h % (ptsz//2) != 0:
        [X_row, Y_row] = rowslice(image[h-ptsz:, :, :], gtimage[h-ptsz:, :, :], ptsz)
        X = np.append(X,X_row,axis=0)
        Y = np.append(Y,Y_row,axis=0)
    return np.array(X), np.array(Y)




