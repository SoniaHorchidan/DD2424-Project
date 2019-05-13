from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from glob import glob
import cv2
import os


model = VGG16()
ground_pictures_folder = "Dataset/Test/images"
predicted_pictures_folder = "Dataset/merged"


def evaluate_predictions(img, filename):
	resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_CUBIC)
	image = img_to_array(resized)
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	image = preprocess_input(image)
	yhat = model.predict(image)
	label = decode_predictions(yhat)
	label = label[0][0]
	print('%s: %s (%.2f%%)' % (filename, label[1], label[2]*100))


def semantic_interpretability():
	folder = os.fsencode(ground_pictures_folder)
	for file in os.listdir(folder):
		filename = os.fsdecode(file)
		if filename.endswith( ('.JPEG', '.png', '.jpg') ): 
			img = cv2.imread("Dataset/Test/images/" + filename, cv2.IMREAD_UNCHANGED)
			pred = glob(predicted_pictures_folder + "/*_" + filename + "*")[0]
			pred_img = cv2.imread(pred, cv2.IMREAD_UNCHANGED)
			evaluate_predictions(img, filename)
			evaluate_predictions(pred_img, "pred_" + filename)
			print("\n")


if __name__== "__main__":
	semantic_interpretability()