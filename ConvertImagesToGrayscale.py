""" Script used to convert color images to grayscale images, which will
be used for trainig

Command line arguments needed:
	src (first argument) - the path to the directory containing the color
						   images
	dest (second argument) - path to the directory which will contain the 
							grayscale images
Output: all the images from the src folder, converted to grayscale and saved 
		to the dest directory
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import sys

def rgb2gray(rgb):
	return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def main(argv):
	src = argv[1]
	dest = argv[2]
	folder = os.fsencode(src)
	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		if filename.endswith( ('.jpeg', '.png', '.jpg') ): # image extension we need
			img = mpimg.imread(src + '/' + filename)
			gray = rgb2gray(img)    
		plt.imsave(dest + '/' +  filename, gray, cmap=plt.get_cmap('gray'))


if __name__ == "__main__":
    main(sys.argv)
