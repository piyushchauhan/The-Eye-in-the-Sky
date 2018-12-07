import logging
import numpy as np
import sys

logging.basicConfig(level=logging.DEBUG,
                    format=' %(asctime)s - %(levelname)s- %(message)s')
# logging.disable(sys.maxsize)

def weight(point, patchSz):
    x, y = point
    ep = 10**-6
    x0, y0 = patchSz/2, patchSz/2
    return 1 - 2 * ((x - x0)**2 + (y - y0)**2) * (1 - ep) / (patchSz**2)


def stitch(patchs, shape):
    """
    Input:  patchs  - list of all patchs(square) each having "n_channels" channels
            shape   - shape (height and widh) of the image that will be stiched
                        shape[0] is height
                        shape[1] is width
    Outut:  img     - final image containing the stiched patchs with dim (shape[0], shape[1], n_channels)
    """
    n_channels = patchs.shape[3]
    fImgShape = tuple(list(shape).append(n_channels))
    finalImgH, finalImgW = shape
    ptsz = patchs.shape[1]

    # Weights each pixel in final image
    weights = np.zeros(fImgShape, dtype=np.float32)
    # Final stiched images will go in finalImage
    finalImage = np.zeros(fImgShape, dtype=np.float32)

    for k in range(patchs.shape[0]):
        # For each patch
        i = k // (finalImgW//ptsz)
        j = k % (finalImgH//ptsz)
        x0, x1 = i * ptsz, (i + 1) * ptsz
        y0, y1 = j * ptsz, (j + 1) * ptsz
        logging.debug("Patch:%s (%s,%s) (%s,%s)"%(k, x0, y0, x1, y1))
        # Calculate the weight of each pixel in the patch 
        wtPatch = np.zeros((ptsz,ptsz,n_channels), dtype=np.float32)

        for x in range(ptsz):                     
            for y in range(ptsz):
                # casting weights into all channels
                wtPatch[x,y,:] = weight((x,y),ptsz)

        for ch in range(1,n_channels):
            wtPatch[:,:,ch] = wtPatch[:,:,0]
        # Putting the patch into the final Image
        finalImage[x0:x1,y0:y1,:] += np.multiply(patchs[k], wtPatch)
        # Adding the weight of the patch to the final Image
        weights[x0:x1,y0:y1,:] += wtPatch

    logging.debug("All patchs done")

            
    finalImage = np.divide(finalImage, weights)
    return finalImage
