import tensorflow as tf
from network import Network
import cv2
import os
import manipulate_data
import numpy as np
import py_compile


def read_images(src):
	x = []
	folder = os.fsencode(src)
	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		if filename.endswith( ('.JPEG', '.png', '.jpg') ): 
			image = cv2.imread(os.path.join(src, filename))
			l_channel, a_channel, b_channel = manipulate_data.convert_rgb_to_lab(image)
			x.append([l_channel]) ##, a_channel, b_channel])
	x = np.array(x)
	x = np.rollaxis(x, 2, 1)
	x = np.rollaxis(x, 3, 2)
	return x


def read_encodings(src):
	y = []
	folder = os.fsencode(src)
	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		enc = np.load(os.path.join(src, filename))
		y.append(enc)
	y = np.array(y)
	return y


py_compile.compile("./manipulate_data.py")

train_dir = "../Dataset/Train/images"
test_dir = "../Dataset/Test/images"
soft_enc_train_dir = "../Dataset/Train/softencoding"
soft_enc_test_dir = "../Dataset/Test/softencoding"

# manipulate_data.create_testset(train_dir, test_dir)

x_train = read_images(train_dir)
x_test = read_images(test_dir)
y_train = read_encodings(soft_enc_train_dir)
y_test = read_encodings(soft_enc_test_dir)


sample = np.zeros((1, 64, 64, 1))
sample[0, :, :, :] = x_train[0, :, :, :]

net = Network(x_train, y_train, x_test, y_test)
net.train(10, 1)
pred = net.predict(sample)
print(pred.shape)
print(pred)