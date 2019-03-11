from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
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
BATCH_SIZE = 32
LEARNING_RATE = 0.001

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_mnist_adv(num_epochs=NUM_EPOCHS,
                  batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  run_cnn=True,
                  test_dae=False, # test CNN with DAE preprocessing
                  test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
                  v_noises=[0.1, 0.2, 0.3, 0.4, 0.5]):

    # ======================================================================
    # General Setup
    # ======================================================================

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
    # Train CNN and calculate Adv error for normal case
    # ======================================================================

    if run_cnn:
        # define TF model graph
        cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
        cnn_model(cnn_model.input)

        wrap = KerasModelWrapper(cnn_model)
        fgsm = FastGradientMethod(wrap, sess=sess)
        fgsm_params = {
            'eps': 0.3,
            'clip_min': 0.,
            'clip_max': 1.
        }
        adv_acc_metric = get_adversarial_acc_metric(cnn_model, fgsm, fgsm_params)
        cnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', adv_acc_metric]
        )

        # Train an MNIST model
        cnn_model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=num_epochs,
                  validation_data=(x_test, y_test),
                  verbose=1)

        # Calculate training error
        _, train_acc, train_adv_acc = cnn_model.evaluate(x_train, y_train,
                                                     batch_size=batch_size,
                                                     verbose=0)
        report.cnn_train_clean_eval = train_acc
        report.cnn_train_adv_eval = train_adv_acc

        # Calculate test error
        # Evaluate the accuracy on legitimate and adversarial test examples
        _, acc, adv_acc = cnn_model.evaluate(x_test, y_test,
                                         batch_size=batch_size,
                                         verbose=0)
        report.cnn_test_clean_eval = acc
        report.cnn_test_adv_eval = adv_acc
        print('Test accuracy on legitimate examples: %0.4f' % acc)
        print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

        # Save trained CNN model
        keras.models.save_model(
            cnn_model,
            "models/cnn.hdf5",
            overwrite=True,
            include_optimizer=True
        )

    # ======================================================================
    # Generate adversarial examples for both train and test
    # ======================================================================

    # if generate_adv_images:
    #     print("Generate Adversarial images")
    #
    #     # Perform adversarial evaluation
    #     wrap = KerasModelWrapper(cnn_model)
    #     fgsm = FastGradientMethod(wrap, sess=sess)
    #     fgsm_params = {
    #         'eps': 0.3,
    #         'clip_min': 0.,
    #         'clip_max': 1.
    #     }
    #
    #     adv_x_train = fgsm.generate_np(x_train, **fgsm_params)
    #     adv_x_test = fgsm.generate_np(x_test, **fgsm_params)
    #     np.save("data/x_train_adv.npy", adv_x_train)
    #     np.save("data/x_test_adv.npy", adv_x_test)

    # ======================================================================
    # Test with DAE
    # ======================================================================

    if test_dae:
        dae_adv_accuracies = []
        dae_accuracies = []
        for v_noise in v_noises:
            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights("models/cnn.hdf5", by_name=False)
            cnn_model(cnn_model.input)

            wrap = KerasModelWrapper(cnn_model)
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
            }

            dae_model = keras.models.load_model(f"models/denoising_autoencoder_{v_noise}.hdf5")

            dae_metric = get_adversarial_acc_with_preprocess_metric(cnn_model, dae_model, fgsm, fgsm_params)

            cnn_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=[dae_metric]
            )

            # Calculate test error
            _, adv_acc = cnn_model.evaluate(x_test, y_test,
                                             batch_size=batch_size,
                                             verbose=0)
            print(f"Adv Test Accuracy for DAE vnoise:\t{v_noise:.1f}: {adv_acc:.5f}")
            dae_adv_accuracies.append(adv_acc)

        for v_noise in v_noises:

            dae_model = keras.models.load_model(f"models/denoising_autoencoder_{v_noise}.hdf5")

            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights("models/cnn.hdf5", by_name=False)

            cnn_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=["accuracy"]
            )

            # Calculate test error
            x_denoised = dae_model.predict(x_test)
            _, acc = cnn_model.evaluate(x_denoised, y_test,
                                             batch_size=batch_size,
                                             verbose=0)
            print(f"Test Accuracy for DAE vnoise:\t{v_noise:.1f}: {acc:.5f}")
            dae_accuracies.append(acc)

        np.savetxt("dae_accuracies.npy", np.array([dae_accuracies, dae_adv_accuracies]))
    # ======================================================================
    # Test with Stacked DAE
    # ======================================================================

    if test_stacked_dae:
        stacked_dae_adv_accuracies = []
        stacked_dae_accuracies = []
        for v_noise in v_noises:
            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights("models/cnn.hdf5", by_name=False)
            cnn_model(cnn_model.input)

            wrap = KerasModelWrapper(cnn_model)
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
            }

            stacked_dae_model = keras.models.load_model(f"models/stacked_denoising_autoencoder_{v_noise}.hdf5")

            stacked_dae_metric = get_adversarial_acc_with_preprocess_metric(cnn_model, stacked_dae_model, fgsm, fgsm_params)

            cnn_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=[stacked_dae_metric]
            )

            # Calculate test error
            _, adv_acc = cnn_model.evaluate(x_test, y_test,
                                             batch_size=batch_size,
                                             verbose=0)
            print(f"Adv Test Accuracy for Stacked DAE vnoise:\t{v_noise:.1f}: {adv_acc:.5f}")
            stacked_dae_adv_accuracies.append(adv_acc)

        for v_noise in v_noises:

            stacked_dae_model = keras.models.load_model(f"models/stacked_denoising_autoencoder_{v_noise}.hdf5")

            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights("models/cnn.hdf5", by_name=False)

            cnn_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=["accuracy"]
            )

            # Calculate test error
            x_denoised = stacked_dae_model.predict(x_test)
            _, acc = cnn_model.evaluate(x_denoised, y_test,
                                             batch_size=batch_size,
                                             verbose=0)
            print(f"Test Accuracy for Stacked DAE vnoise:\t{v_noise:.1f}: {acc:.5f}")
            stacked_dae_accuracies.append(acc)

        np.savetxt("stacked_dae_accuracies.npy", np.array([stacked_dae_accuracies, stacked_dae_adv_accuracies]))

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

    # # Train CNN, and save model
    # run_mnist_adv(num_epochs=1,
    #               batch_size=BATCH_SIZE,
    #               learning_rate=LEARNING_RATE,
    #               run_cnn=True,
    #               test_dae=False,
    #               test_stacked_dae=False)

    # Generate adversarial results / accuracies with
    # Denoising Autoencoder preprocessing
    run_mnist_adv(num_epochs=1,
                  batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  run_cnn=False, # whether to use train CNN and generate adv examples as you go
                  test_dae=True, # test CNN with DAE preprocessing
                  test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
                  #v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])
                  v_noises=[0.5])

    # # Generate adversarial results / accuracies with
    # # Stacked Denoising Autoencoder preprocessing
    # run_mnist_adv(num_epochs=1,
    #               batch_size=BATCH_SIZE,
    #               learning_rate=LEARNING_RATE,
    #               run_cnn=False, # whether to use train CNN and generate adv examples as you go
    #               test_dae=False, # test CNN with DAE preprocessing
    #               test_stacked_dae=True, # test CNN with Stacked DAE preprocessing
    #               v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])
