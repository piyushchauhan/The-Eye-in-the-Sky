from skimage import io
from keras.models import model_from_json, Model
import numpy as np
import os
from indexes import addIndices
from refPad import refPad
from imgSlice import imgSlice
from stitchPatch import stitch
from thresholding import selection
from results import *
import logging
import sys
logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s- %(message)s')
# logging.disable(sys.maxsize)

# Global constants
testDir = os.path.join("data", "sat_test")
patchSize = 256

# Load weights
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(os.path.join("wights"))
print("Loaded model from disk")

resultDir = "resultDir/" ## TO BE FILLED
for i in os.listdir(testDir):
    # Load Image(s) to be predicted
    im = io.imread(os.path.join(testDir,i))
    finalShape = im.shape

    # Data augumentation
    rotIm = np.rot90(im)

    # Adding Indices as channels
    imIndexed = addIndices(im)
    rotImIndexed = addIndices(rotIm)
    del im, rotIm

    # Reflection padding
    imPadded = refPad(imIndexed, patchSize)
    rotImPaded = refPad(rotImIndexed, patchSize)
    del imIndexed, rotImIndexed

    # Slice into the patchs
    imPatchs, _ = imgSlice(imPadded, imPadded, patchSize)
    rotImPatchs, _ = imgSlice(rotImPaded, rotImPaded, patchSize)
    del imPadded, rotImPaded 
    
    # predictPatchs = 

    # TODO Make prediction patch wise model.predict()
    imPredPatchs = model.predict(imPatchs)
    rotImPredPatch = model.predict(rotImPatchs)
    
    # Stitch all patch
    imStiched = stitch(imPredPatchs, finalShape[:2])
    rotImStiched = stitch(rotImPredPatch, finalShape[:2])

    # TODO Combine augumented predictions
    # Rotate in reverse
    imCopy = np.rot90(rotImStiched, -1)
    imRes = np.sqrt(imCopy, imStiched, dtype=np.uint8)
    classWiseIms = imRes[:finalShape[0],:finalShape[1],:]
    
    # TODO Convert multichannel image to RGB
    finalResult = selection(classWiseIms/)

    io.imsave(resultDir+i, finalResult)

# TODO Evalution of the model
result_metric(testDir+"gt/", resultDir)