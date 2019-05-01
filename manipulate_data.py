import numpy as np
import os
import sys
import cv2


'''
Function which converts the RGB picture to LAB and splits the channels
'''
def convert_rgb_to_lab(image):
	image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l_channel, a_channel, b_channel = cv2.split(image_lab)
	return l_channel, a_channel, b_channel


'''
Function which iterates over the images provided in the src directory and 
saves 2 copies of the image: one with the L channel (from the L_channels folder)
and one with the ab channels (from the ab_channels folder)
'''
def create_dataset(src, dest):
	folder = os.fsencode(src)
	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		if filename.endswith( ('.jpeg', '.png', '.jpg') ): 			# image extension we need
			# extract Lab channels
			image = cv2.imread(src + '/' + filename)
			l_channel, a_channel, b_channel = convert_rgb_to_lab(image)

			# save grayscale image
			cv2.imwrite(os.path.join(dest + "/L_channels", filename), l_channel)

			# save ab channels image
			neutrals = np.full(l_channel.shape, 128, dtype=np.uint8)
			merge_channels((filename, neutrals), (filename, a_channel, b_channel), dest = "Dataset", folder = "/ab_channels")


'''
Function which merges the given L and ab channels and saves the result inside the 
dest/folder directory
'''
def merge_channels(l_channel, ab_channels, dest, folder):
	a_channel = ab_channels[1]
	b_channel = ab_channels[2]
	l_chan = l_channel[1]
	filename = l_channel[0]
	merged_channels = cv2.merge((l_chan, a_channel, b_channel))
	final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
	cv2.imwrite(os.path.join(dest + "/" + folder, filename), final_image)


'''
Function which iterates over the images and creates the arrays containing the L or
ab values; helper function for the get_dataset function
'''
def get_dataset_info(src):
	set = []
	folder = os.fsencode(src)
	images = [img.decode("utf-8") for img in (os.listdir(folder))]
	sorted_images = sorted(images)
	for img in sorted_images:
		image = cv2.imread(src + '/' + img)
		l_channel, a_channel, b_channel = convert_rgb_to_lab(image)
		if src.endswith("L_channels"):
			set.append((img, l_channel))
		elif src.endswith("ab_channels"):
			set.append((img, a_channel, b_channel))
	return np.array(set)


'''
Function which iterates over the images from L_channels or ab_channels
directories and returns the matrices with the values of the l and ab 
channels
'''
def get_dataset(src):
	dataset = get_dataset_info(src + "/L_channels")
	labels = get_dataset_info(src + "/ab_channels")
	return dataset, labels


'''
Function which iterates over the images and merges the L and ab channels
'''
def merge_all_images(l_channels, ab_channels, dest = "Dataset", folder = "merged"):
	num_samples = len(l_channels)
	for index in range(num_samples):
		l_chans = l_channels[index]
		ab_chans = ab_channels[index]
		merge_channels(l_chans, ab_chans, dest, folder)
