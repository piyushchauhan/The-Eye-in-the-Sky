from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import model_from_json


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x


N_BANDS = 4
N_CLASSES = 8  # Roads, Trees, Bare Soil, Rails, Buildings, Grass, Water, Pools
CLASS_WEIGHTS = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
N_EPOCHS = 150
UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 150
TRAIN_SZ = 4000  # train size
VAL_SZ = 1000    # validation size


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

# all availiable ids: from "01" to "24"
trainIds = [str(i) for i in range(1, 15)]


# if __name__ == '__main__':
X_DICT_TRAIN = dict()
Y_DICT_TRAIN = dict()
X_DICT_VALIDATION = dict()
Y_DICT_VALIDATION = dict()

def loadImages():
    print('Reading images')
    for img_id in trainIds:
        img_m = normalize(tiff.imread(
            './data/sat/{}.tif'.format(img_id)).transpose([1, 2, 0]))
        mask = tiff.imread('./data/gt/{}.tif'.format(img_id)
                        ).transpose([1, 2, 0]) / 255
        # use 75% of image as train and 25% for validation
        train_xsz = int(3/4 * img_m.shape[0])
        X_DICT_TRAIN[img_id] = img_m[:train_xsz, :, :]
        Y_DICT_TRAIN[img_id] = mask[:train_xsz, :, :]
        X_DICT_VALIDATION[img_id] = img_m[train_xsz:, :, :]
        Y_DICT_VALIDATION[img_id] = mask[train_xsz:, :, :]
        print(img_id + ' read')
    print('Images were read')


def train_net():
    print("start train net")
    x_train, y_train = get_patches(
        X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
    x_val, y_val = get_patches(
        X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
    model = get_model()
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
    #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
    #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
    model_checkpoint = ModelCheckpoint(
        weights_path, monitor='val_loss', save_best_only=True)
    csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
    tensorboard = TensorBoard(
        log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
              verbose=2, shuffle=True,
              callbacks=[model_checkpoint, csv_logger, tensorboard],
              validation_data=(x_val, y_val))
    return model

loadImages()
model = train_net()

# evaluate the model
scores_train = model.evaluate(x_train, y_train, verbose=0)
scores_val = model.evaluate(x_val, y_val, verbose=0)
# print("Training %s: %.2f%%" % (model.metrics_names[1], scores_train[1]*100))
print("Validation %s: %.2f%%" % (model.metrics_names[1], scores_val[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
