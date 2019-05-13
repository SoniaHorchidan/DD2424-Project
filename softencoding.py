import numpy as np
import os
import sys
import cv2
from scipy import misc
from sklearn.neighbors import NearestNeighbors
import numpy as np
from manipulate_data import convert_rgb_to_lab


def decode(encoding):
	cord = np.load("pts_in_hull.npy")
	temperature = 0.34
	encoding[encoding!=0] = np.exp(np.log(encoding[encoding!=0])/temperature)
	encoding = encoding/np.sum(encoding,axis=2)[:,:,np.newaxis]

	a_layer = np.sum(encoding * cord[:,0],axis=2)
	b_layer = np.sum(encoding * cord[:,1],axis=2)
	a = a_layer.astype(np.int32) + 128			# (a_layer + 90) / 190 * 256
	b = b_layer.astype(np.int32) + 128			# (b_layer + 90) / 190 * 256
	return a, b


def softencoding(a_channel, b_channel):
	cord = np.load("pts_in_hull.npy")

	nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(cord)

	encoding = np.zeros([64*64,313])
	a = a_channel.reshape(-1,1)
	b = b_channel.reshape(-1,1)
	X = np.column_stack((a,b))
	distances, indices = nbrs.kneighbors(X)
	sigma = 5
	wts = np.exp(-distances**2/(2*sigma**2))
	wts = wts/np.sum(wts,axis=1)[:,np.newaxis]

	encoding[np.arange(0,64*64)[:,np.newaxis],indices] = wts
	encoding = encoding.reshape(64,64,313)
	return encoding
