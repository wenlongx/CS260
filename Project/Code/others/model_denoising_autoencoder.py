from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper

Sequential = keras.models.Sequential
Conv2D = keras.layers.Conv2D
MaxPooling2D = keras.layers.MaxPooling2D
Dropout = keras.layers.Dropout
Dense = keras.layers.Dense
Activation = keras.layers.Activation
Flatten = keras.layers.Flatten
KerasModel = keras.models.Model
Reshape = keras.layers.Reshape
Conv2DTranspose = keras.Conv2DTranspose
Input = keras.layers.Input
Conv2DTranspose = keras.layers.Conv2DTranspose

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
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    return model

def DenoisingAutoencoder(input_shape):
    model = Sequential()
    model.add(Conv2D(filters=,
                     kernel_size=,
                     strides=,
                     padding=))


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

def run_mnist_adv(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                  testing=False, learning_rate=LEARNING_RATE):

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # set random seed
    tf.set_random_seed(42)

    # can use gpu
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )

    # Create TF session and set Keras backend session as TF
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Get MNIST test data
    mnist = MNIST()
    x_train, y_train = mnist.get_set("train")
    x_test, y_test = mnist.get_set("test")

    # Obtain image params
    n_rows, n_cols, n_channels = x_train.shape[1:4]
    n_classes = y_train.shape[1]

    # define TF model graph
    model = ConvNet((n_rows, n_cols, n_channels), n_classes)
    model(model.input)

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {
        'eps': 0.3,
        'clip_min': 0.,
        'clip_max': 1.
    }
    adv_acc_metric = get_adversarial_acc_metric(model, fgsm, fgsm_params)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', adv_acc_metric]
    )

    # Train an MNIST model
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(x_test, y_test),
              verbose=1)

    # Evaluate the accuracy on legitimate and adversarial test examples
    _, acc, adv_acc = model.evaluate(x_test, y_test,
                                     batch_size=batch_size,
                                     verbose=0)
    report.clean_train_clean_eval = acc
    report.clean_train_adv_eval = adv_acc
    print('Test accuracy on legitimate examples: %0.4f' % acc)
    print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

    # Calculate training error
    if testing:
        _, train_acc, train_adv_acc = model.evaluate(x_train, y_train,
                                                     batch_size=batch_size,
                                                     verbose=0)
        report.train_clean_train_clean_eval = train_acc
        report.train_clean_train_adv_eval = train_adv_acc

    print("Repeating the process, using adversarial training")
    # Redefine Keras model
    model_2 = ConvNet((n_rows, n_cols, n_channels), n_classes)
    model_2(model_2.input)
    wrap_2 = KerasModelWrapper(model_2)
    fgsm_2 = FastGradientMethod(wrap_2, sess=sess)

    # Use a loss function based on legitimate and adversarial examples
    adv_loss_2 = get_adversarial_loss(model_2, fgsm_2, fgsm_params)
    adv_acc_metric_2 = get_adversarial_acc_metric(model_2, fgsm_2, fgsm_params)
    model_2.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=adv_loss_2,
        metrics=['accuracy', adv_acc_metric_2]
    )

    # Train an MNIST model
    model_2.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=num_epochs,
                validation_data=(x_test, y_test),
                verbose=1)

    # Evaluate the accuracy on legitimate and adversarial test examples
    _, acc, adv_acc = model_2.evaluate(x_test, y_test,
                                       batch_size=batch_size,
                                       verbose=0)
    report.adv_train_clean_eval = acc
    report.adv_train_adv_eval = adv_acc
    print('Test accuracy on legitimate examples: %0.4f' % acc)
    print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

    # Calculate training error
    if testing:
        _, train_acc, train_adv_acc = model_2.evaluate(x_train, y_train,
                                                       batch_size=batch_size,
                                                       verbose=0)
        report.train_adv_train_clean_eval = train_acc
        report.train_adv_train_adv_eval = train_adv_acc

    return report

if __name__ == "__main__":
    run_mnist_adv(testing=False)
