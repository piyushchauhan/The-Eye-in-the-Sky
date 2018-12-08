from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from indexes import *
from rgb2Classes import rgb2Classes
from refPad import *
import numpy as np
import os
import sys
import logging
from imgSlice import imgSlice

logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s- %(message)s')
# logging.disable(sys.maxsize)

# Global Constants
patchSize = 256
n_classes = 9

# Load Data
loc = "data/"

x, y = [], []
def addNewch(image, channel):
    return np.concatenate((image, channel.reshape(tuple(list(image.shape[:2]) + [1]))), axis=2)


for i in os.listdir(loc + "sat/"):
    satIm = io.imread(loc + "sat/"+ i)
    gtIm = io.imread(loc + "gt/" + i)
    val = int(i.split(".")[0])
    # Adding Indices as channels
    # satIm = np.concatenate(satIm, evi(satIm).reshape(list(satIm.shape[:2]) + [1]), axis=2)
    satIm = addNewch(satIm, evi2(satIm))
    satIm = addNewch(satIm, msavi2(satIm))
    satIm = addNewch(satIm, ndvi(satIm))
    satIm = addNewch(satIm, ndwi(satIm))
    satIm = addNewch(satIm, osavi2(satIm))
    satIm = addNewch(satIm, savi(satIm))
    logging.debug(str(i) + str(satIm.shape))

    x.insert(val, satIm)
    y.insert(val, gtIm)


x = np.array(x)
y = np.array(y)
logging.debug("x.shape: " + str(x.shape))
logging.debug("x[0].shape: " + str(x[0].shape))
logging.debug("y.shape: " + str(y.shape))
logging.debug("y[0].shape: " + str(y[0].shape))

# TODO Data augumentation





# Convert RGB to n_classes channels
for i in os.listdir(loc + "gt/"):
    j = int(i.split(".")[0])
    y[j - 1] = rgb2Classes(y[j - 1])
logging.debug("y.shape: " + str(y.shape))
logging.debug("y[0].shape: " + str(y[0].shape))

in_ch = x[0].shape[2]
out_ch = y[0].shape[2]
# Reflection Padding
xPatchs, yPatchs = [], []
xPatchs = np.array(xPatchs).reshape((0,patchSize,patchSize,in_ch))
yPatchs = np.array(yPatchs).reshape((0,patchSize,patchSize,out_ch))
for i in range(x.shape[0]):
    x[i] = refPad(x[i], patchSize)
    y[i] = refPad(y[i], patchSize)
    logging.debug("sat/" + str(i) + ".tif is padded to " + str(x[i].shape))
    logging.debug("gt/" + str(i) + ".tif is padded to " + str(y[i].shape))
    # Slice into the patchs
    xt, yt = imgSlice(x[i],y[i],patchSize)
    xPatchs = np.append(xPatchs,xt,axis=0)
    yPatchs = np.append(yPatchs,yt,axis=0)

logging.debug("xPatchs: " + str(xPatchs.shape))
logging.debug("yPatchs: " + str(yPatchs.shape))

# TODO Put into the model model.fit
# TODO Stitch all patchs
# TODO Model summary

