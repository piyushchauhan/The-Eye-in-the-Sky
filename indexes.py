# Red: 0, Green,, Blue:2, NIR:3

import numpy as np

def addNewChSat(image, channel):
    return np.concatenate((image, channel.reshape(tuple(list(image.shape[:2]) + [1]))), axis=2)

def addIndices(satImg):
    satImg2 = addNewChSat(satImg, evi2(satImg))
    satImg2 = addNewChSat(satImg2, msavi2(satImg))
    satImg2 = addNewChSat(satImg2, ndvi(satImg))
    satImg2 = addNewChSat(satImg2, ndwi(satImg))
    satImg2 = addNewChSat(satImg2, osavi2(satImg))
    satImg2 = addNewChSat(satImg2, savi(satImg))
    # logging.debug(str(i) + str(satImg.shape))
    return satImg2

def ndvi(image):
    return np.nan_to_num((image[:, :, 3] - image[:, :, 0])/(image[:, :, 3] + image[:, :, 0]))


def ndwi(image):
    return np.nan_to_num((image[:, :, 3] - image[:, :, 1])/(image[:, :, 1] + image[:, :, 3]))


def evi(image):
    assert not bool(image[:, :, 3] + 6*image[:, :, 0] - 7.5*image[:, :, 2] + 1), "Division by zero"
    return np.nan_to_num(2.5*((image[:, :, 3] - image[:, :, 0])/(image[:, :, 3] + 6*image[:, :, 0] - 7.5*image[:, :, 2] + 1)))


def savi(image):
    return np.nan_to_num((1.5)*((image[:, :, 3] - image[:, :, 0])/(image[:, :, 3] + image[:, :, 0] + 0.5)))


def evi2(image):
    return np.nan_to_num(2.5*(image[:, :, 3] - image[:, :, 0])/(image[:, :, 3] + 2.4*image[:, :, 0] + 1))


def msavi2(image):
    nir = image[:, :, 3]
    red = image[:, :, 0]
    sqr = np.sqrt((2*nir+1)**2 - 8*(nir - red))
    msavi2 = (2*nir+1-sqr)/2
    return np.nan_to_num(msavi2)


def osavi2(image):
    return (image[:, :, 3] - image[:, :, 0])/(image[:, :, 3] + image[:, :, 0] + 0.16)
