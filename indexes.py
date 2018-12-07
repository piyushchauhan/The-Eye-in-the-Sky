#Red: 0, Green:1, Blue:2, NIR:3

import numpy as np

def ndvi(image):
    return (image[:,:,3] - image[:,:0])/(image[:,:,3] + image[:,:0])

def ndwi(image):
    return (image[:,:,3] - image[:,:1])/(image[:,:,1] + image[:,:1])

def evi(image):
    return 2.5*( (image[:,:,3] - image[:,:0])/(image[:,:,3] + 6*image[:,:,0] - 7.5*image[:,:,2] + 1) )

def savi(image):
    return (1.5)*((image[:,:,3] - image[:,:0])/(image[:,:,3] + image[:,:0] + 0.5))

def evi2(image):
    return 2.5*(image[:,:,3] - image[:,:0])/(image[:,:,3] + 2.4*image[:,:0] + 1)

def msavi2(image):
    nir = image[:,:,3]
    red = image[:,:,0]
    sqr = np.sqrt( (2*nir+1)**2 - 8*(nir - red) )
    msavi2 = (2*nir+1-sqr)/2
    return msavi2

def osavi2(image):
    return (image[:,:,3] - image[:,:,0])/(image[:,:,3] + image[:,:,0] + 0.16)