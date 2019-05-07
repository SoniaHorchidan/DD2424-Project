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

def pixel_analyzing(src):

	cord = np.load("pts_in_hull.npy")

	nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(cord)

	main = os.fsencode(src)
	result = np.zeros(313)

	for folder in os.listdir(main):
		foldername = src+"/"+os.fsdecode(folder)+"/images/"
		for file in os.listdir(os.fsencode(foldername)):
			filename = os.fsdecode(file)
			if filename.endswith( ('.JPEG', '.png', '.jpg') ) and not filename.startswith("."): 			# image extension we need
				image = cv2.imread(foldername+filename)
				l_channel, a_channel, b_channel = convert_rgb_to_lab(image)
				a = a_channel.reshape(-1,1)
				b = b_channel.reshape(-1,1)
				X = np.column_stack((a,b))
				distances, indices = nbrs.kneighbors(X)
				result[indices]+=1 #saved the indices in an array

	result = result/np.sum(result)
	sigma = 5
	wts = np.exp(-result ** 2 / (2 * sigma ** 2))
	wts = wts / np.sum([wts], axis=1)[:, np.newaxis]
	#print(wts)
	#print(np.sum(wts))
	np.save("prior_probs", arr=wts)

if __name__ == "__main__":
	pixel_analyzing("train")
