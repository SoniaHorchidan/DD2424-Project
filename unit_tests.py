from softencoding import softencoding, decode
from manipulate_data import convert_rgb_to_lab, merge_channels
import numpy as np
import cv2


'''
Check if softencodings have exatcly 5 values different than 0 per pixel
'''
def test_softencoding():
	image_path = "Dataset/Test/images/0.JPEG"
	image = cv2.imread(image_path)
	image_small = cv2.resize(image,(16,16))
	l, a, b = convert_rgb_to_lab(image)
	l_small, a_small, b_small = convert_rgb_to_lab(image_small)
	encoding = softencoding(a_small, b_small)
	nonzeros_per_pixels = np.count_nonzero(encoding, axis = 2)
	num_diff_five = np.where(nonzeros_per_pixels != 5)
	assert(num_diff_five[0].size == 0)


'''
Save decode(encode(image)) and compare it visually with image 
'''
def test_decode():
	### TODO: find a better way?
	image_path = "Dataset/Test/images/0.JPEG"
	image = cv2.imread(image_path)
	image_small = cv2.resize(image,(16,16))
	l, a, b = convert_rgb_to_lab(image)
	l_small, a_small, b_small = convert_rgb_to_lab(image_small)
	a_small = a_small.astype(np.int32) - 128 
	b_small = b_small.astype(np.int32) - 128  
	encoding = softencoding(a_small, b_small)
	a_dec, b_dec = decode(encoding)
	new_img_path = "test_res0.JPEG"
	merge_channels((new_img_path, l.astype(dtype=np.uint8)), 
			(new_img_path, a_dec.astype(dtype=np.uint8), b_dec.astype(dtype=np.uint8)),
			 "Dataset/Test", "unit_testing", True)
	


if __name__== "__main__":
	test_softencoding()
	test_decode()