from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from skimage import io
from indexes import *
from rgb2Classes import rgb2Classes
from refPad import *
import numpy as np
import os
import sys
from imgSlice import imgSlice
from unet_model import *
from preProcessing import *
import logging
logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s - %(message)s')
# logging.disable(sys.maxsize)

# Global Constants
patchSize = 256
n_classes = 9
epochs = 100

# Load Data
loc = "data/"
in_ch = 10
out_ch = 9
x, y = [], []

for i in os.listdir(loc + "sat/"):
    satIm = io.imread(loc + "sat/"+ i)
    gtIm = io.imread(loc + "gt/" + i)
    val = int(i.split(".")[0])
    logging.debug(str(i) + "\t" + str(satIm.shape))
    x.insert(val, satIm)
    y.insert(val, gtIm)


x = np.array(x)
y = np.array(y)
logging.debug("x.shape: " + str(x.shape))
logging.debug("x[0].shape: " + str(x[0].shape))
logging.debug("y.shape: " + str(y.shape))
logging.debug("y[0].shape: " + str(y[0].shape))

# Reflection Padding
xPatchs, yPatchs = [], []
xPatchs = np.array(xPatchs).reshape((0,patchSize,patchSize,x[0].shape[2]))
yPatchs = np.array(yPatchs).reshape((0,patchSize,patchSize,y[0].shape[2]))
for i in range(x.shape[0]):
    x[i] = refPad(x[i], patchSize)
    y[i] = refPad(y[i], patchSize)
    logging.debug("sat/" + str(i + 1) + ".tif is padded to " + str(x[i].shape))
    logging.debug("gt/" + str(i + 1) + ".tif is padded to " + str(y[i].shape))
    # Slice into the patchs
    xt, yt = imgSlice(x[i],y[i],patchSize)
    xPatchs = np.append(xPatchs,xt,axis=0)
    yPatchs = np.append(yPatchs,yt,axis=0)

logging.debug("xPatchs: " + str(xPatchs.shape))
logging.debug("yPatchs: " + str(yPatchs.shape))

# for xPatch, yPatch in zip(xPatchs,yPatchs):
    

# TODO Data augumentation
image_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rescale=1./255,
        rotation_range=90,
        validation_split=0.25,
        # preprocessing_function=addIndices,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True)

mask_datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rescale=1./255,
        rotation_range=90,
        validation_split=0.25,
        # preprocessing_function=rgb2Classes,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True
        )

seed = 1
image_datagen.fit(xPatchs, augment=True, seed=seed)
mask_datagen.fit(yPatchs, augment=True, seed=seed)

image_generator = image_datagen.flow(xPatchs, seed=seed)
mask_generator = mask_datagen.flow(yPatchs, seed=seed)

train_generator = zip(image_generator, mask_generator)

# Load model
in_ch = 4
model = unet_model(out_ch, patchSize, in_ch)
logging.debug("Model create succes")

model_checkpoint = ModelCheckpoint(os.path.join("weights", "uNetWeights.hdf5"), monitor='loss',verbose=1, save_best_only=True)

# Put into the model model.fit
model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50,
    validation_steps=800,
    callbacks=[model_checkpoint])

logging.debug("Training done")
# TODO Stitch all patchs
# TODO Model summary

model.summary()
model.save_weights(os.path.join("weights", "uNetWeightsLast.hdf5"))