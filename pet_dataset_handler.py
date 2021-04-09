################################################################
##
## Pet Dataset Handler
##
## Part of the code borrowed from Keras's official document
##
##  Author:  Peizhi Yan
##    Date:  Mar. 7, 2021
## Editted:  Mar. 28, 2021
##
################################################################

import numpy as np
import os
import cv2
import random
from tensorflow.compat.v1 import keras

""" mean and std for each channel (derived from training data set) """
mean = np.array([0.4781414, 0.44545504, 0.3940489]) # mean pixel intensity for each channel
std = np.array([0.2631686, 0.25842762, 0.2667502]) # standard deviation of pixel intensity for each channel



def rotate(image, angle=15):
    """Rotate the image"""
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

def scale(image, scale=1.0):
    """Scale the image"""
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), 0, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image

def flip(image, vertical=False, horizontal=False):
    """Flip the image (vertically or horizontally or both)"""
    if vertical or horizontal:
        if vertical and horizontal:
            fc = -1
        else:
            if vertical:
                fc = 0 
            else:
                fc = 1
        image = cv2.flip(image, flipCode=fc)
    return image




def data_augmentation(image, mask, angle_range=(-15,15), scale_range=(0.7,1.3), flip=True):
    """My custom data augmentation method"""
    # Random rotation
    random_angle = random.randrange(angle_range[0], angle_range[1]+1)
    image = rotate(image, angle=random_angle)
    mask = rotate(mask, angle=random_angle)

    # Random scaling
    random_scale = random.uniform(scale_range[0], scale_range[1])
    image = scale(image, scale=random_scale)
    mask = scale(mask, scale=random_scale)
    
    # Random flipping
    if random.randrange(0, 2) == 0:
        # no flipping
        pass
    else:
        if random.randrange(0, 2) == 0:
            # horizontal flipping
            image = cv2.flip(image, flipCode=1)
            mask = cv2.flip(mask, flipCode=1)
        else:
            # vertical flipping
            image = cv2.flip(image, flipCode=0)
            mask = cv2.flip(mask, flipCode=0)

    # Set "empty" area in mask as background
    ##mask = np.where(mask == [0,0,0], [0,1.0,0], mask)

    return image, mask





class PetDataGenerator(keras.utils.Sequence):
    """My custom overloaded data generator, for loading data and perform data augmentation"""

    def __init__(self, data_path, batch_size, augmentation=True, shuffle=True, autoencoder=False, pseudo_label=False, n_classes=2):

        """Initialization"""
        fnames = []
        for fname in os.listdir(data_path):
            if fname.endswith('.jpg'):
                fnames.append(fname[:-4])
        self.data_path = data_path
        self.dim = (224,224) # height and width
        self.batch_size = batch_size
        self.flist = np.array(fnames) # List of file names (not include suffix)
        self.n_channels = 3
        self.n_classes = n_classes
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.on_epoch_end()
        self.autoencoder = autoencoder
        self.pseudo_label = pseudo_label

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.flist) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.flist[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.flist))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """Generates data containing batch_size samples""" # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load sample
            img = cv2.imread(self.data_path+'/'+ID+'.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.

            # Load mask
            mask = np.load(self.data_path+'/_'+ID+'.npy') / 1.0
            
            # Data augmentation
            if self.augmentation:
                img, mask = data_augmentation(img, mask, angle_range=(-5,5), scale_range=(0.8,1.2), flip=True)

            mask = np.array(mask, dtype=np.int) # load label from file

            # Store data
            X[i] = img
            if self.pseudo_label == False:
                Y[i,:,:,0] = mask[:,:,1] # background (class 0)
                Y[i,:,:,1] = mask[:,:,0] + mask[:,:,2] # foreground (class 1)
            elif self.pseudo_label == True and self.n_classes == 1:
                Y[i,:,:,0] = mask[:,:]
            elif self.pseudo_label == True and self.n_classes == 2:
                Y[i,:,:,0] = 1 - mask[:,:] # background (class 0)
                Y[i,:,:,1] = mask[:,:] # foreground (class 1)

        # Z-score standardization for each color channel
        for c in range(3):
            X[:,:,:,c] = (X[:,:,:,c] - mean[c]) / std[c]

        if self.autoencoder:
            return X, X
        return X, Y


def load_data(data_path, list_ids, width, height, standardization=True):
    """Load all the data with listed IDs"""
    # Initialization
    X = np.zeros((len(list_ids), height, width, 3), dtype=np.float32)
    Y = np.zeros((len(list_ids), height, width, 2), dtype=np.float32)
    # Generate data
    for i, ID in enumerate(list_ids):
        # Load sample
        img = cv2.imread(data_path+'/'+ID+'.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.
        # Load mask
        mask = np.load(data_path+'/_'+ID+'.npy') / 1.0
        mask = np.array(mask, dtype=np.int)
        # Store data
        X[i] = img
        Y[i,:,:,0] = mask[:,:,1] # background (class 0)
        Y[i,:,:,1] = mask[:,:,0] + mask[:,:,2] # foreground (class 1)
    # Z-score standardization for each color channel
    if standardization:
        for c in range(3):
            X[:,:,:,c] = (X[:,:,:,c] - mean[c]) / std[c]
    return X, Y



