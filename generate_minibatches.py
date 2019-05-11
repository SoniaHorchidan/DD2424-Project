import os
import random
from random import shuffle
import cv2 as cv
import numpy as np
from keras.utils import Sequence
from manipulate_data import convert_rgb_to_lab
from softencoding import softencoding


class MiniBatchesSeqGeneretor(Sequence):
    def __init__(self, usage, batch_size = 80):
        self.usage = usage
        self.batch_size = batch_size
        self.image_folder = "./Dataset"

        if usage == 'train':
            names_file = 'train_names.txt'
            self.image_folder += "/Train"
        else:
            names_file = 'valid_names.txt'
            self.image_folder += "/Validation"

        self.image_folder += "/images"

        with open(names_file, 'r') as f:
            self.names = f.read().splitlines()


    def __len__(self):
        return int(np.ceil(len(self.names) / float(self.batch_size)))

    def __getitem__(self, idx):
        i = idx * self.batch_size

        length = min(self.batch_size, (len(self.names) - i))
        batch_x = np.empty((length, 64, 64, 1), dtype=np.float32)
        batch_y = np.empty((length, 16, 16, 313), dtype=np.float32)

        for i_batch in range(length):
            name = self.names[i]
            image = os.path.join(self.image_folder, name)
            bgr = cv.imread(image)
            l, a, b = convert_rgb_to_lab(bgr)
            x = l / 255.

            out_a= cv.resize(a, (16, 16), cv.INTER_CUBIC)
            out_b= cv.resize(b, (16, 16), cv.INTER_CUBIC)

            out_a = out_a.astype(np.int32) - 128        ## / 256 * 190 - 90
            out_b = out_b.astype(np.int32) - 128        ## / 256 * 190 - 90
            
            y = softencoding(out_a, out_b)

            batch_x[i_batch, :, :, 0] = x
            batch_y[i_batch] = y

            i += 1

        return batch_x, batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.names)



def train_gen_minibatches():
    return MiniBatchesSeqGeneretor('train')


def valid_gen_minibatches():
    return MiniBatchesSeqGeneretor('valid')