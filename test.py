from refPad import *
from slice import slice
from stitchPatch import stitch
from skimage import io
import numpy as np
import os


# logging.disable(sys.maxsize)


ptsz = 256
loc = "data/"
import logging

logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s- %(message)s')
sat = io.imread(loc+"sat/4.tif")
gt = io.imread(loc+"gt/4.tif")
finalShape = gt.shape
logging.debug("sat.shape:" + str(sat.shape) + "\tgt.shape:" + str(gt.shape))

satref, gtref = refPad(sat,ptsz), refPad(gt, ptsz)
logging.debug("satref.shape:"+str(satref.shape)+"\tgtref.shape:"+str(gtref.shape))
satPatchs, gtPatchs = slice(satref, gtref, ptsz)
logging.debug("satPatchs.shape:"+str(satPatchs.shape)+"\tgtPatchs.shape:"+str(gtPatchs.shape))
stitchedGt = stitch(gtPatchs, gtref.shape[:2])
logging.debug("Patchs of GT stiched")
logging.debug("Images Saved with shape " + str(finalShape))
io.imsave("test.tif", stitchedGt[:finalShape[0],:finalShape[1]])





