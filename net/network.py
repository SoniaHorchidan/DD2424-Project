import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import InputLayer
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage.interpolation as sni
import keras


"""base model"""
class Network(object):

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.history = AccuracyHistory()

        batch_shape = self.x_train[0, :, :, :].shape
        self.build_network(batch_shape)



    def build_network(self, input_shape):
        self.model = Sequential()

        ### Layer 1
        self.model.add(Conv2D(64, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         input_shape = input_shape,
                        data_format='channels_last',
                         name = "conv1_1"))
        self.model.add(Conv2D(64, kernel_size = 3, strides = 2,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv1_2"))
        self.model.add(BatchNormalization())

        ### Layer 2
        self.model.add(Conv2D(128, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv2_1"))
        self.model.add(Conv2D(128, kernel_size = 3, strides = 2,
                         activation = 'relu',
                         padding = "same",
                         dilation_rate = 1,
                         name = "conv2_2"))
        self.model.add(BatchNormalization())

        ### Layer 3
        self.model.add(Conv2D(256, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv3_1"))
        self.model.add(Conv2D(256, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         padding = "same",
                         dilation_rate = 1,
                         name = "conv3_2"))
        self.model.add(Conv2D(256, kernel_size = 3, strides = 2,
                         activation = 'relu',
                         padding = "same",
                         dilation_rate = 1,
                         name = "conv3_3"))
        self.model.add(BatchNormalization())

        ### Layer 4
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv4_1"))
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         padding = "same",
                         dilation_rate = 1,
                         name = "conv4_2"))
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         padding = "same",
                         dilation_rate = 1,
                         name = "conv4_3"))
        self.model.add(BatchNormalization())

        ### Layer 5
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 2,
                         padding = "same",
                         name = "conv5_1"))
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         padding = "same",
                         dilation_rate = 2,
                         name = "conv5_2"))
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         padding = "same",
                         dilation_rate = 2,
                         name = "conv5_3"))
        self.model.add(BatchNormalization())

        ### Layer 6
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 2,
                         padding = "same",
                         name = "conv6_1"))
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 2,
                         padding = "same",
                         name = "conv6_2"))
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 2,
                         padding = "same",
                         name = "conv6_3"))
        self.model.add(BatchNormalization())

        ### Layer 7
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv7_1"))
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv7_2"))
        self.model.add(Conv2D(512, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv7_3"))
        self.model.add(BatchNormalization())

        ### Layer 8
        self.model.add(Deconvolution2D(256, kernel_size = 4, strides = 2,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv8_1"))
        self.model.add(Conv2D(256, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv8_2"))
        self.model.add(Conv2D(256, kernel_size = 3, strides = 1,
                         activation = 'relu',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv8_3"))

        ### Softmax
        self.model.add(Conv2D(313, kernel_size = 1, strides = 1,
                         activation = 'softmax',
                         dilation_rate = 1,
                         name = "conv8_313"))


        ### TODO add weights
        self.model.compile(loss = multimodal_cross_entropy(np.ones(313, )),
              optimizer = "adam",
              metrics = ['accuracy'])


    def train(self, epochs, batch_size = 80):
        self.model.fit(self.x_train, self.y_train,
          batch_size = batch_size,
          epochs = epochs,
          verbose = 1,
          validation_data = (self.x_test, self.y_test),
          callbacks = [self.history])


    def predict(self, image):
        prediction = self.model.predict(image)
        pic = sni.zoom(prediction[0, :, :, :],(1. * 64 / 16, 1. * 64 / 16, 1))
        return prediction, pic
       


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))




def multimodal_cross_entropy(weights):
        
    weights = keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= keras.backend.sum(y_pred, axis = - 1, keepdims = True)
        # clip to prevent NaN's and Inf's
        y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        # calc
        loss = y_true * keras.backend.log(y_pred) * weights
        loss = - keras.backend.sum(loss, - 1)
        return loss
        
    return loss