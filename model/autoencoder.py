####################################################################
##
## Deep Autoencoder Model
##
## Author: Peizhi Yan
##
####################################################################

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers


def get_model(img_size, M=3):
    inputs = keras.Input(shape=img_size + (3,))

    x = inputs
    
    #M = 5
    filters = {
        0: 32,
        1: 32,
        2: 64,
        3: 64,
        4: 16,
    }
    # Down-sampling
    for component in range(M):
        nfilters = filters[component]
        x = layers.Conv2D(nfilters, 3, strides=1, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(nfilters, 3, strides=1, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), padding="valid")(x) # max-pooling down sample
    
    encoding = x
    
    # Up-sampling
    for component in range(M):
        nfilters = filters[M-component-1]
        x = layers.Conv2D(nfilters, 3, strides=1, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(nfilters, 3, strides=1, padding="same")(x)
        x = layers.Activation("relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x) # bi-linear up sample
            

    # Add a per-pixel classification layer
    x = layers.Conv2D(3, 1, strides=1, activation="linear", padding="same")(x) # conv 1x1
    
    # Define the model
    model = keras.Model(inputs, x)
    return model, encoding


## Free up RAM in case the model definition cells were run multiple times
#keras.backend.clear_session()
#
## Build model
#model = get_model(img_size, num_classes)
#model.summary()