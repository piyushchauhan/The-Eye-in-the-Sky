# Don't change the original output of the U-net with this function
# Number of output channels assumed to be 9
import numpy as np

thresholds = {
    "Water": 0.4,
    "Trees": 0.4,
    "Grass": 0.4,
    "Roads": 0.5,
    "Buildings": 0.5,
    "Soil": 0.35,
    "Rails": 0.35,
    "Pools": 0.3,
    "Unlabelled": 0.6
}

ch_order = ["Water", "Trees", "Grass", "Trains",
            "Soil", "Roads", "Buildings", "Pools", "Unlabelled"]

Colors = {
    "Water": [0, 0, 150],
    "Trees": [0, 125, 0],
    "Grass": [0, 255, 0],
    "Trains": [255, 255, 0],
    "Soil": [150, 80, 0],
    "Roads": [0, 0, 0],
    "Unlabelled": [255, 255, 255],
    "Buildings": [100, 100, 100],
    "Pools": [150, 150, 255],
}


def selection(image):

    ch_vals = []
    for ch in ch_order:
        ch_vals.append(Colors[ch])
    ch_vals = np.array(ch_vals)

    classCh = thresholdings(image)

    # Output image to be rgb
    output = np.full((image.shape[0], image.shape[1], 3), 255)

    th_ch = np.zeros(image.shape)

    for i in range(9):
        th_tmp = np.full(
            (image.shape[0], image.shape[1]), thresholds[ch_order[i]])
        th_ch[:, :, i] = th_tmp

    im_th = image - th_ch
    im_th = np.logical_and(classCh, im_th)
    im_th = np.logical_not(classCh) + im_th
    color_vals = np.argmin(im_th, axis=2)

    output = ch_vals[color_vals]

    return output


def thresholdings(image):
    for i in range(9):
        image[:, :, i] = image[:, :, i] > thresholds[ch_order[i]]

    return image
