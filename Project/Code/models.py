from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np

from cleverhans.attacks import FastGradientMethod

Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Dropout = keras.layers.Dropout
Dense = keras.layers.Dense
Activation = keras.layers.Activation
Flatten = keras.layers.Flatten
KerasModel = keras.models.Model
Reshape = keras.layers.Reshape
Input = keras.layers.Input
Conv2DTranspose = keras.layers.Conv2DTranspose

import keras.backend as K

# def ConvNet(x):
#     """ConvNet builds the graph for a deep net for classifying digits.
#     Args:
#         x: an input tensor with the dimensions (N_examples, 784), where 784 is the number of pixels in a standard MNIST image.
#     Returns:
#         A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values equal to the logits of classifying the digit into one of 10 classes (the digits 0-9). keep_prob is a scalar placeholder for the probability of dropout.
#     """
#     with tf.variable_scope("ConvNet", reuse=reuse):
#
#         # 28 x 28 x 1
#         x = tf.reshape(x, shape=[-1, 28, 28, 1])
#
#         conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
#         conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
#         conv2 = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
#         conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
#         fc1 = tf.contrib.layers.flatten(conv2)
#         fc1 = tf.layers.dense(fc1, 1024)
#         fc1 = tf.layers(dropout(fc1, rate=dropout, training=is_training))
#         out = tf.layers.dense(fc1, n_classes)
#
#
#     return out

NUM_EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# Define ConvNet model
def ConvNet(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     padding="same",
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32,
                     kernel_size=(5, 5),
                     strides=(1, 1),
                     padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def get_adversarial_acc_metric(model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc

def get_adversarial_acc_with_preprocess_metric(model, p_model, fgsm, fgsm_params):
    def adv_acc(y, _):
        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Accuracy on the adversarial examples
        x_adv = p_model(x_adv)
        preds_adv = model(x_adv)
        return keras.metrics.categorical_accuracy(y, preds_adv)

    return adv_acc

def get_adversarial_loss(model, fgsm, fgsm_params):
    def adv_loss(y, preds):
        # Cross-entropy on the legitimate examples
        cross_ent = keras.losses.categorical_crossentropy(y, preds)

        # Generate adversarial examples
        x_adv = fgsm.generate(model.input, **fgsm_params)
        # Consider the attack to be constant
        x_adv = tf.stop_gradient(x_adv)

        # Cross-entropy on the adversarial examples
        preds_adv = model(x_adv)
        cross_ent_adv = keras.losses.categorical_crossentropy(y, preds_adv)

        return 0.5 * cross_ent + 0.5 * cross_ent_adv

    return adv_loss

def DenoisingAutoencoder(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding="same",
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     strides=(1, 1),
                     padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2DTranspose(filters=64,
                              kernel_size=(3, 3),
                              padding="same",
                              strides=(2,2)))
    model.add(Activation('relu'))
    # model.add(Conv2DTranspose(filters=32,
    #                           kernel_size=(3, 3),
    #                           padding="same",
    #                           strides=(2,2)))
    # model.add(Activation('relu'))
    model.add(Conv2DTranspose(filters=1,
                              kernel_size=(3, 3),
                              padding="same",
                              strides=(2,2)))
    model.add(Activation('sigmoid'))
    return model

def StackedDenoisingAutoencoder(input_shape, num_stacks=3):
    model = Sequential()
    for i in range(num_stacks):
        model.add(Conv2D(filters=32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding="same",
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=32,
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         padding="same"))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2DTranspose(filters=64,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  strides=(2,2)))
        model.add(Activation('relu'))
        model.add(Conv2DTranspose(filters=1,
                                  kernel_size=(3, 3),
                                  padding="same",
                                  strides=(2,2)))
        model.add(Activation('sigmoid'))
    return model

def transfer_weights_stacked_dae(stacked_dae, autoencoders):
    num_layers = len(autoencoders[0].layers)
    layer_idx = 0
    for autoencoder in autoencoders:
        for i in range(num_layers):
            stacked_dae.layers[layer_idx + i].set_weights(autoencoder.layers[i].get_weights())
        layer_idx += num_layers

    return stacked_dae

# def ContractiveAutoencoder(input_shape):
#     model = Sequential()
#     model.add(Dense())
#     model.add(Activation('sigmoid'))
#     model.add(Dense(np.product(input_shape)))
#     model.add(Activation('sigmoid'))
#     return model
#
# def contractive_loss(y_pred, y_true):
#     mse = K.mean(K.square(y_true - y_pred), axis=1)
#
#     W = K.variable(value=model.get_layer('encoded').get_weights()[0])  # N x N_hidden
#     W = K.transpose(W)  # N_hidden x N
#     h = model.get_layer('encoded').output
#     dh = h * (1 - h)  # N_batch x N_hidden
#
#     # N_batch x N_hidden * N_hidden x 1 = N_batch x 1
#     contractive = lam * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
#
#     return mse + contractive
