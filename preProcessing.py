# from keras.applications import xception
from keras_applications import xception
from keras.preprocessing.image import img_to_array, array_to_img
from indexes import addIndices
from rgb2Classes import rgb2Classes

def imagePreprocess(x):
    img = xception.preprocess_input(x)
    return array_to_img(addIndices(img))

def maskPreprocess(x):
    img = xception.preprocess_input(x)
    return array_to_img(rgb2Classes(img))