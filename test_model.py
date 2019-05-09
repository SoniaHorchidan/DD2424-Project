import tensorflow as tf
from network import Network, multimodal_cross_entropy
import cv2
import os
import manipulate_data
import numpy as np
import py_compile
from keras.models import load_model
import datetime
from softencoding import decode

def read_images(src):
	x = []
	x_small = []
	folder = os.fsencode(src)
	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		if filename.endswith( ('.JPEG', '.png', '.jpg') ): 
			image = cv2.imread(os.path.join(src, filename))
			image2 = cv2.resize(image,(16,16))

			l_channel, a_channel, b_channel = manipulate_data.convert_rgb_to_lab(image)
			l_channel2, a_channel2, b_channel2 = manipulate_data.convert_rgb_to_lab(image2)

			x.append([l_channel])
			x_small.append([l_channel2])
	x = np.array(x)
	x = np.rollaxis(x, 2, 1)
	x = np.rollaxis(x, 3, 2)

	x_small = np.array(x_small)
	x_small = np.rollaxis(x_small, 2, 1)
	x_small = np.rollaxis(x_small, 3, 2)
	return x, x_small


def read_encodings(src):
	y = []
	folder = os.fsencode(src)
	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		enc = np.load(os.path.join(src, filename))
		y.append(enc)
	y = np.array(y)
	return y


model_num = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
model_name = "model" + str(model_num) + ".h5"

train_dir = "Dataset/Train/images"
test_dir = "Dataset/Test/images"
soft_enc_train_dir = "Dataset/Train/softencoding"
soft_enc_test_dir = "Dataset/Test/softencoding"

# manipulate_data.create_testset(train_dir, test_dir)

x_train, x_train_small = read_images(train_dir)
x_test, x_test_small = read_images(test_dir)
y_train = read_encodings(soft_enc_train_dir)
y_test = read_encodings(soft_enc_test_dir)


num_pic = 1
sample = np.zeros((1, 64, 64, 1))
sample[0, :, :, :] = x_train[num_pic, :, :, :]
label = np.zeros((1, 16, 16, 313))
label[0, :, :, :] = y_train[num_pic, :, :, :]


net = Network(x_train, y_train, x_test, y_test)

### use when training a new model
model = net.train(1, 80)
model.save(model_name)

### use when loading a pretrained model
# model = load_model('model.h5', custom_objects={'loss': multimodal_cross_entropy(np.ones(313,))})
# net.set_loaded_model(model)

pred = net.predict(sample)

a, b = decode(pred.reshape((16, 16, 313)))
l = x_train_small[num_pic, :, :, 0];

manipulate_data.merge_channels(("test.jpg", l.astype(dtype=np.uint8)), 
	("test.jpg", a.astype(dtype=np.uint8), b.astype(dtype=np.uint8)), "Dataset", "merged", True)
