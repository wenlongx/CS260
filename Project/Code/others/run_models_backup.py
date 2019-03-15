from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow import keras
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

from cleverhans.attacks import FastGradientMethod
from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import KerasModelWrapper

from models import *

NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def train_dae(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, testing=False, learning_rate=LEARNING_RATE, v_noise=0.3):

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

    print(x_test.shape)

    # define TF model graph
    model_l1 = DenoisingAutoencoder((n_rows, n_cols, n_channels))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse'
    )

    # corrupt inputs
    noise_train = v_noise * np.random.normal(size=x_train.shape)
    noise_test = v_noise * np.random.normal(size=x_test.shape)
    x_train_noisy = np.clip(x_train + noise_train, 0.0, 1.0)
    x_test_noisy = np.clip(x_test + noise_test, 0.0, 1.0)

    # Train an MNIST model
    model.fit(x_train_noisy, x_train,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(x_test_noisy, x_test),
              verbose=1)

    model.summary()


    # Evaluate the accuracy on legitimate and adversarial test examples
    x_test_denoised = model.predict(x_test_noisy,
                                    batch_size=batch_size,
                                    verbose=0)

    print(x_test_denoised.shape)
    print(x_test_denoised[0].shape)
    first_img = np.reshape(x_test_denoised[0], (28, 28))
    plt.imsave("temp.png", first_img)

    # Display the 1st 8 corrupted and denoised images
    rows, cols = 10, 30
    num = rows * cols
    imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_test_denoised[:num]])
    imgs = imgs.reshape((rows * 3, cols, n_rows, n_cols))
    imgs = np.vstack(np.split(imgs, rows, axis=1))
    imgs = imgs.reshape((rows * 3, -1, n_rows, n_cols))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    imgs = (imgs * 255).astype(np.uint8)
    # plt.figure()
    # plt.axis('off')
    # plt.title('Original images: top rows, '
    #           'Corrupted Input: middle rows, '
    #           'Denoised Input:  third rows')
    # plt.imshow(imgs, interpolation='none', cmap='gray')
    Image.fromarray(imgs).save('corrupted_and_denoised.png')
    # plt.show()

    # Save model locally
    keras.models.save_model(
        model,
        "models/denoising_autoencoder.hdf5",
        overwrite=True,
        include_optimizer=True
    )


def run_mnist_adv(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, testing=False, learning_rate=LEARNING_RATE):

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

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

    # ======================================================================
    # Generate Adversarial examples
    # ======================================================================

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

    # print("Repeating the process, using adversarial training")
    # # Redefine Keras model
    # model_2 = ConvNet((n_rows, n_cols, n_channels), n_classes)
    # model_2(model_2.input)
    # wrap_2 = KerasModelWrapper(model_2)
    # fgsm_2 = FastGradientMethod(wrap_2, sess=sess)
    #
    # # Use a loss function based on legitimate and adversarial examples
    # adv_loss_2 = get_adversarial_loss(model_2, fgsm_2, fgsm_params)
    # adv_acc_metric_2 = get_adversarial_acc_metric(model_2, fgsm_2, fgsm_params)
    # model_2.compile(
    #     optimizer=keras.optimizers.Adam(learning_rate),
    #     loss=adv_loss_2,
    #     metrics=['accuracy', adv_acc_metric_2]
    # )
    #
    # # Train an MNIST model
    # model_2.fit(x_train, y_train,
    #             batch_size=batch_size,
    #             epochs=num_epochs,
    #             validation_data=(x_test, y_test),
    #             verbose=1)
    #
    # # Evaluate the accuracy on legitimate and adversarial test examples
    # _, acc, adv_acc = model_2.evaluate(x_test, y_test,
    #                                    batch_size=batch_size,
    #                                    verbose=0)
    # report.adv_train_clean_eval = acc
    # report.adv_train_adv_eval = adv_acc
    # print('Test accuracy on legitimate examples: %0.4f' % acc)
    # print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)
    #
    # # Calculate training error
    # if testing:
    #     _, train_acc, train_adv_acc = model_2.evaluate(x_train, y_train,
    #                                                    batch_size=batch_size,
    #                                                    verbose=0)
    #     report.train_adv_train_clean_eval = train_acc
    #     report.train_adv_train_adv_eval = train_adv_acc

    return report

if __name__ == "__main__":
    # set random seed
    tf.set_random_seed(42)

    # run MNIST model
    run_mnist_adv(num_epochs=50, testing=False)
    train_dae(testing=False)
