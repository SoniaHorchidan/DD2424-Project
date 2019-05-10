import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers import InputLayer
from sklearn.neighbors import NearestNeighbors
from keras.regularizers import l2
import keras
from generate_minibatches import train_gen_minibatches, valid_gen_minibatches


l2_reg = l2(1e-3)

"""base model"""
class Network(object):

    def __init__(self):
        # self.x_train = x_train
        # self.y_train = y_train
        # self.x_test = x_test
        # self.y_test = y_test
        self.history = AccuracyHistory()

        batch_shape = (64, 64, 1)
        self.build_network(batch_shape)



    def build_network(self, input_shape):
        self.model = Sequential()

        ### Layer 1
        self.model.add(Conv2D(64, kernel_size = 3,
                         activation = 'relu',
                         padding = "same",
                         input_shape = input_shape,
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         data_format='channels_last',
                         name = "conv1_1"))
        self.model.add(Conv2D(64, kernel_size = 3, strides = 2,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv1_2"))
        self.model.add(BatchNormalization())

        ### Layer 2
        self.model.add(Conv2D(128, kernel_size = 3,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv2_1"))
        self.model.add(Conv2D(128, kernel_size = 3, strides = 2,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         dilation_rate = 1,
                         name = "conv2_2"))
        self.model.add(BatchNormalization())

        ### Layer 3
        self.model.add(Conv2D(256, kernel_size = 3, 
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv3_1"))
        self.model.add(Conv2D(256, kernel_size = 3,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv3_2"))
        self.model.add(Conv2D(256, kernel_size = 3, strides = 2,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv3_3"))
        self.model.add(BatchNormalization())

        ### Layer 4
        self.model.add(Conv2D(512, kernel_size = 3,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv4_1"))
        self.model.add(Conv2D(512, kernel_size = 3, 
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv4_2"))
        self.model.add(Conv2D(512, kernel_size = 3, 
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv4_3"))
        self.model.add(BatchNormalization())

        ### Layer 5
        self.model.add(Conv2D(512, kernel_size = 3, 
                         activation = 'relu',
                         dilation_rate = 2,
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv5_1"))
        self.model.add(Conv2D(512, kernel_size = 3,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         dilation_rate = 2,
                         name = "conv5_2"))
        self.model.add(Conv2D(512, kernel_size = 3, 
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         dilation_rate = 2,
                         name = "conv5_3"))
        self.model.add(BatchNormalization())

        ### Layer 6
        self.model.add(Conv2D(512, kernel_size = 3, 
                         activation = 'relu',
                         dilation_rate = 2,
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv6_1"))
        self.model.add(Conv2D(512, kernel_size = 3,
                         activation = 'relu',
                         dilation_rate = 2,
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv6_2"))
        self.model.add(Conv2D(512, kernel_size = 3, 
                         activation = 'relu',
                         dilation_rate = 2,
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv6_3"))
        self.model.add(BatchNormalization())

        ### Layer 7
        self.model.add(Conv2D(512, kernel_size = 3, 
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv7_1"))
        self.model.add(Conv2D(512, kernel_size = 3,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv7_2"))
        self.model.add(Conv2D(512, kernel_size = 3,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv7_3"))
        self.model.add(BatchNormalization())

        ### Layer 8
        self.model.add(Deconvolution2D(256, kernel_size = 4, strides = 2,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv8_1"))
        self.model.add(Conv2D(256, kernel_size = 3,
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv8_2"))
        self.model.add(Conv2D(256, kernel_size = 3, 
                         activation = 'relu',
                         padding = "same",
                         kernel_initializer = "he_normal",
                         kernel_regularizer = l2_reg,
                         name = "conv8_3"))
        self.model.add(BatchNormalization())

        ### Softmax
        self.model.add(Conv2D(313, kernel_size = 1, 
                         activation = 'softmax',
                         dilation_rate = 1,
                         padding = "same",
                         name = "conv8_313"))


        ### TODO add weights
        self.model.compile(loss = multimodal_cross_entropy(),
              optimizer = "adam",
              metrics = ['accuracy'])


    def train(self, epochs, batch_size = 80):
        self.model.fit_generator(train_gen_minibatches(),
                            validation_data = valid_gen_minibatches(),
                            epochs = epochs,
                            verbose = 1,
                            use_multiprocessing = True,
                            workers = 8)
        return self.model


    def predict(self, image):
        prediction = self.model.predict(image)
        return prediction

    def set_loaded_model(self, model):
        self.model = model
        self.model.compile(loss = multimodal_cross_entropy(),
              optimizer = "adam",
              metrics = ['accuracy'])



class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))



def multimodal_cross_entropy():

    classrebalance = np.load("classrebalance.npy")
    classrebalance = classrebalance.astype(np.float32)

    def loss(y_true, y_pred):
        y_pred /= keras.backend.sum(y_pred, axis = - 1, keepdims = True)
        y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())

        index = keras.backend.argmax(y_true,axis = 3) #We want a tensor of dimension 16x16x(minibatchsize) So might have to change axis
        weights = keras.backend.gather(classrebalance, index) #We want a tensor of dimension 16x16x(minibatchsize)
        loss = y_true * keras.backend.log(y_pred) * tf.expand_dims(weights, -1)
        loss = - keras.backend.sum(loss, - 1)
        return loss

    return loss
