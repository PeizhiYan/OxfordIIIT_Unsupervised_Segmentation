###################################################
##
## Loss Functions
##
## Some loss function codes were borrowed and the
## sources will be stated in the function. 
##
## Author:  Peizhi Yan
##   Date:  Mar. 8, 2021
## Update:  Mar. 27, 2021
##
###################################################

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np


def cce_loss(targ, pred):
    """Loss function"""
    cce = tf.keras.losses.CategoricalCrossentropy()
    l_cce = cce(targ, pred) # cross-entropy loss
    return l_cce


def dice_loss(targ, pred, ignore_background=False):
    """Dice loss function"""
    # Original code from:
    #  https://gchlebus.github.io/2018/02/18/semantic-segmentation-loss-functions.html
    # Modified by: 
    #  Peizhi Yan
    if ignore_background:
        targ = targ[..., 1:]
        pred = pred[..., 1:]
    axis = (0,1,2,3) # NxHxWxC
    eps = 1e-7
    nom = (2 * tf.reduce_sum(targ * pred, axis=axis) + eps)
    targ = tf.square(targ)
    pred = tf.square(pred)
    denom = tf.reduce_sum(targ, axis=axis) + tf.reduce_sum(pred, axis=axis) + eps
    return 1 - tf.reduce_mean(nom / denom)


def icassp_loss(feature_layer, batch_size, mu=100):
    """
    This loss function is a re-implemented and modified version.
    -------------------------------------------------------------------
    Totorial on how to define custon loss in Keras:
        https://towardsdatascience.com/advanced-keras-constructing-
        complex-custom-losses-and-metrics-c07ca130a618
    ===================================================================
    Originally from the ICASSP paper (therefore I named it icassp loss):
         Kanezaki, A. (2018). Unsupervised Image Segmentation By 
         Backpropagation. ICASSP, 2–4.
    Original code (implemented through PyTorch):
        https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
        /blob/master/demo.py
    -------------------------------------------------------------------
    The loss function has two terms:
        1) feature similarity: pixels with similar deep feature should have
                               higher probability to have the same label;
        2) spatial continuity: pixels that are spatially close should have
                               higher probability to have the same label.
    """
    # - targ: the label. shape [N,H,W,C]
    # - pred: the predicted mask. shape [N,H,W,C]
    # - mu: weighting factor
    # N: number of samples (batch_size)
    # H: height
    # W: width
    # C: number of classes

    def loss(targ, pred):
        # in fact, targ is not used, because we are doing unsupervised learning
        # the reason for idicating targ as an input parameter is to comply with
        # Keras.
        
        n_classes = int(pred.shape[3]) # number of classes
        n_features = int(feature_layer.shape[3]) # number of features
        
        # c is the pseudo-label (don't forget argmax is not differentiable)
        c = tf.math.argmax(pred, axis=3) # get predicted label
        c = tf.one_hot(indices = c, depth=n_classes) # convert predicted label to pixel-wise one-hot encoding

        """feature similariry loss"""
        loss_sim = 0
        loss_sim = tf.math.reduce_mean(-tf.math.multiply(c, tf.math.log(feature_layer)))

        
        """continuity loss"""
        loss_con = 0
        fea_y = tf.math.reduce_mean(feature_layer[:,1:,:,:] - feature_layer[:,0:-1,:,:])
        fea_x = tf.math.reduce_mean(feature_layer[:,:,1:,:] - feature_layer[:,:,0:-1,:])
        loss_con_y = tf.norm(fea_y, ord=1) # L1 norm
        loss_con_x = tf.norm(fea_x, ord=1) # L1 norm
        loss_con = loss_con_y + loss_con_x
    
        return loss_sim + mu*loss_con
    return loss
    




