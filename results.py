import sklearn.metrics as sklm
from scipy.misc import toimage
from scipy.misc import imread
from skimage import io
import matplotlib.pyplot as ppl
import pandas as pd
import seaborn as sn
from colormap import rgb2hex
import numpy as np
import os

def result_metric(path_t, path_p):
	def read_im(file, path):
		PATH = path
		files = []
		x = []
		for path,dirs,files in os.walk(PATH):
			for filename in file:
				fullpath = os.path.join(path, filename)
				img = imread(fullpath)
				x.append(img)
		return(x)

	def eval(y_tru, y_pred):
		col_tru = decipher(y_tru)
		col_pred = decipher(y_pred)

		c_mat = sklm.confusion_matrix(col_tru, col_pred)
		c_acc = sklm.accuracy_score(col_tru, col_pred)
		c_ckap = sklm.cohen_kappa_score(col_tru, col_pred)

		return c_mat, c_acc, c_ckap

	def decipher(x):
		#Roads, Buildings, Trees, Grass, Bare Soil, Water, Railways, Swimming pools
		#Water = [0,0,150] Trees = [0,125,0] Grass = [0,255,0] Trains = [255,255,0] Soil = [150,80,0]
		#Roads = [0,0,0] Unlabelled = [255,255,255] Buildings = [100,100,100] Pools = [150,150,255]
		road = 0
		road_col = np.array([0,0,0])
		build = 0
		build_col = np.array([100,100,100])
		tree = 0
		tree_col = np.array([0,125,0])
		grass = 0
		grass_col = np.array([0,255,0])
		soil = 0
		soil_col = np.array([150,80,0])
		water = 0
		water_col = np.array([0,0,150])
		rail = 0
		rail_col = np.array([255,255,0])
		pool = 0
		pool_col = np.array([150,150,255])
		un = 0
		un_col = np.array([255,255,255])
		fin = []
		for i in x:
			print("Image "+str(x.index(i)+1)+" processing...")
			i = np.array(i)
			for j in range(i.shape[0]):
				for k in range(i.shape[1]):
					if np.array_equal(i[j][k], road_col):
						fin.append(0)
					elif np.array_equal(i[j][k], build_col):
						fin.append(1)
					elif np.array_equal(i[j][k], tree_col):
						fin.append(2)
					elif np.array_equal(i[j][k], grass_col):
						fin.append(3)
					elif np.array_equal(i[j][k], soil_col):
						fin.append(4)
					elif np.array_equal(i[j][k], water_col):
						fin.append(5)
					elif np.array_equal(i[j][k], rail_col):
						fin.append(6)
					elif np.array_equal(i[j][k], pool_col):
						fin.append(7)
					elif np.array_equal(i[j][k], un_col):
						fin.append(8)

		return fin

	file_x_tru = []
	file_x_pred = []

	for i in range(1,15):
		file_x_tru.append(str(i)+".tif")

	file_x_pred = file_x_tru.copy()
	
	path_tru = path_t
	path_pred = path_p

	x_tru_im = read_im(file_x_tru, path_tru)
	x_pred_im = read_im(file_x_pred, path_pred)

	conmat, accuracy, cohen_kappa = eval(x_tru_im, x_pred_im)

	print("Accuracy: ", accuracy)
	print("Cohen Kappa: ", cohen_kappa)
	print("Confusion Matrix: ")
	print(conmat)

	labels = ['road', 'building', 'tree', 'grass', 'soil', 'water', 'rail', 'pool', 'unlabelled']

	df_cm = pd.DataFrame(conmat, labels, labels)

	sn.set(font_scale=1.4)
	sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})
	ppl.show()

	return conmat, accuracy, cohen_kappa
