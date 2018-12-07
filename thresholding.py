def thresholdings(image):
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

    ch_order = ["Water", "Trees", "Grass", "Trains", "Soil", "Roads", "Buildings", "Pools", "Unlabelled"]

    for i in range(9):
        image[:,:,i] = image[:,:,i] > thresholds[ch_order[i]]
    
    return image
