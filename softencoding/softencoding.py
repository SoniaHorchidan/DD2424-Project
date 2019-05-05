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
	
	#Create weights
	alpha = 1
	gamma = 0.5

	uni_probs = np.zeros_like(prob)
	uni_probs[prob!=0] = 1.
	uni_probs = uni_probs/np.sum(uni_probs)


	prior_mix = (1-gamma)*prob + gamma*uni_probs


	prior_factor = prior_mix**-alpha
	prior_factor = prior_factor/np.sum(prob*prior_factor) 

	###
	
	nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(cord)

	main = os.fsencode(src)

	for folder in os.listdir(main):
		foldername = src+"/"+os.fsdecode(folder)+"/"
		for file in os.listdir(os.fsencode(foldername)):
			filename = os.fsdecode(file)
			if filename.endswith( ('.JPEG', '.png', '.jpg') ) and not filename.startswith("."): 			# image extension we need

				image = cv2.imread(foldername+filename)
				image = cv2.resize(image,(16,16))
				l_channel, a_channel, b_channel = convert_rgb_to_lab(image)
				encoding = np.zeros([16*16,313])
				a = a_channel.reshape(-1,1)
				b = b_channel.reshape(-1,1)
				X = np.column_stack((a,b))
				distances, indices = nbrs.kneighbors(X)
				sigma = 5
				wts = np.exp(-distances**2/(2*sigma**2))
				wts = wts/np.sum(wts,axis=1)[:,np.newaxis]
				#removed class rebalancing
				encoding[np.arange(0,16*16)[:,np.newaxis],indices] = wts * prior_factor[indices[:,0],np.newaxis]
				encoding = encoding.reshape(16,16,313)

				np.save("../Dataset/Train/softencoding/"+filename[:-5],encoding.astype(np.float16) )

				#print(wts)

softencoding("../Dataset/Train")
