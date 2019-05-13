from keras.models import load_model
from network import multimodal_cross_entropy
from manipulate_data import convert_rgb_to_lab, merge_channels
from softencoding import decode, softencoding
from network import Network
import numpy as np
import os
import cv2


## Change with wanted model name
model_name = 'new_model.h5'
net = Network(model_name)
net.load()

test_folder = "Dataset/Test/images"

folder = os.fsencode(test_folder)
for file in os.listdir(folder):
	filename = os.fsdecode(file)
	if filename.endswith( ('.JPEG', '.png', '.jpg') ): 
		image = cv2.imread(os.path.join(test_folder, filename))
		image_small = cv2.resize(image,(16,16))

		l, a, b = convert_rgb_to_lab(image)

		x_test = np.empty((1, 64, 64, 1), dtype=np.float32)
		x_test[0, :, :, 0] = l / 255.

		pred = net.predict(x_test)

		a, b = decode(pred.reshape((16, 16, 313)))

		new_result_name = model_name + "_" + filename

		merge_channels((new_result_name, l.astype(dtype=np.uint8)), 
			(new_result_name, a.astype(dtype=np.uint8), b.astype(dtype=np.uint8)),
			 "Dataset", "merged", True)
