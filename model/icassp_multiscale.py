####################################################################
##
## Architecture Used for Unsupervised Semantic Segmentation
##
## Author: Peizhi Yan
##
## This code is a Tensorflow implementation of the network used in 
## ICASSP paper:
##     Kanezaki, A. (2018). Unsupervised Image Segmentation By 
##     Backpropagation. ICASSP, 2–4.
##
####################################################################

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers



def get_model(img_size, num_classes, M=4):
    inputs = keras.Input(shape=img_size + (3,))

    x = inputs
    
    # Feature Extraction Blocks
    # Down-sampling
    for component in range(M):
        x = layers.Conv2D(100, 3, strides=1, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="valid")(x) # max-pooling down sample
        x = layers.BatchNormalization()(x)
    # Up-sampling
    for component in range(M):
        x = layers.Conv2D(100, 3, strides=1, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x) # bi-linear up sample
        x = layers.BatchNormalization()(x)    

    # Add a per-pixel classification layer
    x = layers.Conv2D(num_classes, 1, strides=1, activation="linear", padding="same")(x) # conv 1x1
    x = layers.BatchNormalization()(x)
    x = layers.Activation(tf.keras.activations.sigmoid)(x)
    
    #outputs = layers.Activation(tf.keras.activations.softmax)(x)

    # Define the model
    #model = keras.Model(inputs, outputs)
    model = keras.Model(inputs, x)
    return model

## Free up RAM in case the model definition cells were run multiple times
#keras.backend.clear_session()
#
## Build model
#model = get_model(img_size, num_classes)
#model.summary()