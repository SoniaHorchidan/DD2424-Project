import numpy as np
import os
import sys
import cv2
from scipy import misc
from sklearn.neighbors import NearestNeighbors
import numpy as np

def convert_rgb_to_lab(image):
	image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l_channel, a_channel, b_channel = cv2.split(image_lab)
	return l_channel, a_channel, b_channel

def softencoding(src):

	cord = np.load("pts_in_hull.npy")
	prob = np.load("prior_probs.npy")
	nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(cord)

	main = os.fsencode(src)

	for folder in os.listdir(main):
		foldername = src+"/"+os.fsdecode(folder)+"/images/"
		for file in os.listdir(os.fsencode(foldername)):
			filename = os.fsdecode(file)
			if filename.endswith( ('.JPEG', '.png', '.jpg') ) and not filename.startswith("."): 			# image extension we need
				image = cv2.imread(foldername+filename)
				l_channel, a_channel, b_channel = convert_rgb_to_lab(image)
				encoding = np.zeros([64*64,313])
				a = a_channel.reshape(-1,1)
				b = b_channel.reshape(-1,1)
				X = np.column_stack((a,b))
				distances, indices = nbrs.kneighbors(X)
				sigma = 5
				wts = np.exp(-distances**2/(2*sigma**2))
				wts = wts/np.sum(wts,axis=1)[:,np.newaxis]
				#encoding[:,indices] = wts * prob[indices[:,0],np.newaxis]
				encoding = wts.reshape(64,64,313)

				np.save("Dataset/softencoding/"+filename[:-5],encoding.astype(np.float16) )

				#print(wts)

softencoding("train")
