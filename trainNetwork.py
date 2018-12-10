from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard
from skimage import io
from indexes import addIndices
from rgb2Classes import rgb2Classes
from refPad import refPad
import numpy as np
import os
import sys
from imgSlice import imgSlice
from unet_model import unet_model
# from preProcessing import *
import logging
logging.basicConfig(level=logging.DEBUG,
			format=' %(asctime)s - %(levelname)s - %(message)s')
# logging.disable(sys.maxsize)

# Global Constants
patchSize = 256
n_classes = 9
in_ch = 10
out_ch = 9
epochs = 50
loc = "data/"
saveModelto = os.path.join("weights", "unetweights.hdf5")
class TestCallback(callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True)
checkpoint = ModelCheckpoint(saveModelto, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


# Load Data
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
	

# Data augumentation
image_datagen = ImageDataGenerator(
	samplewise_center=True,
	samplewise_std_normalization=True,
	rescale=1./255,
	rotation_range=90,
	# validation_split=0.25,
	shear_range=0.2,
	zoom_range=0.2,
	vertical_flip=True,
	horizontal_flip=True)

mask_datagen = ImageDataGenerator(
	samplewise_center=True,
	samplewise_std_normalization=True,
	rescale=1./255,
	rotation_range=90,
	# validation_split=0.25,
	shear_range=0.2,
	zoom_range=0.2,
	vertical_flip=True,
	horizontal_flip=True
	)

logging.debug("Image and mask generators are ready")
seed = 1
image_datagen.fit(xPatchs, augment=True, seed=seed)
mask_datagen.fit(yPatchs, augment=True, seed=seed)

image_generator = image_datagen.flow(xPatchs, seed=seed)
mask_generator = mask_datagen.flow(yPatchs, seed=seed)

train_generator = zip(image_generator, mask_generator)

# Load model
model = unet_model(out_ch, patchSize, in_ch, gpus=6)
logging.debug("Model create succes")

model_checkpoint = ModelCheckpoint(os.path.join("weights", "uNetWeights.hdf5"), monitor='loss',verbose=1, save_best_only=True)

# Put into the model model.fit
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=2000,
#     validation_steps=800,
#     callbacks=[model_checkpoint])

for e in range(epochs):
	logging.debug("Epoch {}".format(e))
	batches = 0
	for x_batch, y_batch in train_generator:
		   local_x_Train = np.array([]).reshape((0,patchSize,patchSize,in_ch))
		   local_y_Trains = np.array([]).reshape((0,patchSize,patchSize,out_ch))
		   for x_ in x_batch:
			   x_ = addIndices(x_)
			   local_x_Train = np.append(local_x_Train, x_.reshape(([1]+list(x_.shape))), axis=0)
		   for y_ in y_batch:
			   y_ = rgb2Classes(y_)
			   local_y_Trains = np.append(local_y_Trains, y_.reshape(([1]+list(y_.shape))), axis=0)
		   model.fit(local_x_Train[:32],local_y_Trains[:32],validation_data=(local_x_Train[32:36],local_y_Trains[32:36]), callbacks= [TestCallback((local_x_Train[36:],local_y_Trains[36:])), checkpoint])
		   batches += 1
		   if batches >= len(xPatchs)/40:
			   break

logging.debug("Training done")

model.summary()
model.save_weights(os.path.join("weights", "uNetWeightsLast.hdf5"))