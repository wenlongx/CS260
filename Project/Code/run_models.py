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

NUM_EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# MODEL_PATH = "models"
MODEL_PATH = "20_epochs"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_mnist_adv(num_epochs=NUM_EPOCHS,
                  batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  run_cnn=False,
                  test_cnn=False,
                  adversarial_training=False,
                  test_ae=False, # test CNN with AE preprocessing
                  test_cae=False, # test CNN with DAE preprocessing
                  test_dae=False, # test CNN with DAE preprocessing
                  test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
                  v_noises=[0.1, 0.2, 0.3, 0.4, 0.5],
                  lambdas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                  num_stacks=3):

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
    # Train CNN and calculate Adv error for adversarial case
    # ======================================================================

    if run_cnn and adversarial_training:

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
        print('Test accuracy on legitimate examples: %0.4f' % acc)
        print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

        keras.models.save_model(
            cnn_model,
            "models/cnn_adversarial.hdf5",
            overwrite=True,
            include_optimizer=True
        )

    # ======================================================================
    # Run CNN
    # ======================================================================
    if test_cnn:
        cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
        cnn_model.load_weights(f"{MODEL_PATH}/cnn_backup.hdf5", by_name=False)
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

        cnn_model.summary()

        _, acc, adv_acc = cnn_model.evaluate(x_test, y_test,
                                         batch_size=batch_size,
                                         verbose=0)
        print('Test accuracy on legitimate examples: %0.4f' % acc)
        print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)


    if adversarial_training:
        cnn_name = "cnn_adversarial"
    else:
        cnn_name = "cnn"

    # ======================================================================
    # Test with AE
    # ======================================================================
    if test_ae:
        # define TF model graph
        cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
        cnn_model.load_weights(f"{MODEL_PATH}/{cnn_name}.hdf5", by_name=False)
        cnn_model(cnn_model.input)

        wrap = KerasModelWrapper(cnn_model)
        fgsm = FastGradientMethod(wrap, sess=sess)
        fgsm_params = {
            'eps': 0.3,
            'clip_min': 0.,
            'clip_max': 1.
        }

        ae_model = keras.models.load_model(f"{MODEL_PATH}/autoencoder.hdf5")

        ae_metric = get_adversarial_acc_with_preprocess_metric(cnn_model, ae_model, fgsm, fgsm_params)

        cnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=[ae_metric]
        )

        # Calculate test error
        _, adv_acc = cnn_model.evaluate(x_test, y_test,
                                         batch_size=batch_size,
                                         verbose=0)
        print(f"Adv Test Accuracy for AE: {adv_acc:.5f}")


        ae_model = keras.models.load_model(f"{MODEL_PATH}/autoencoder.hdf5")

        # define TF model graph
        cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
        cnn_model.load_weights(f"{MODEL_PATH}/{cnn_name}.hdf5", by_name=False)

        cnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=["accuracy"]
        )

        # Calculate test error
        x_recon = ae_model.predict(x_test)
        _, acc = cnn_model.evaluate(x_recon, y_test,
                                         batch_size=batch_size,
                                         verbose=0)
        print(f"Test Accuracy for AE: {acc:.5f}")

        np.savetxt("dae_accuracies.npy", np.array([acc, adv_acc]))

    # ======================================================================
    # Test with CAE
    # ======================================================================
    elif test_cae:
        print("test cae")
        cae_adv_accuracies = []
        cae_accuracies = []
        for lam in lambdas:
            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights(f"{MODEL_PATH}/{cnn_name}.hdf5", by_name=False)
            cnn_model(cnn_model.input)

            wrap = KerasModelWrapper(cnn_model)
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
            }

            cae_model = keras.models.load_model(f"{MODEL_PATH}/contractive_autoencoder_{lam}.hdf5")

            cae_metric = get_adversarial_acc_with_preprocess_metric(cnn_model, cae_model, fgsm, fgsm_params)
            cnn_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=[cae_metric]
            )

            # Calculate test error
            _, adv_acc = cnn_model.evaluate(x_test, y_test,
                                             batch_size=batch_size,
                                             verbose=0)

            cae_adv_accuracies.append(adv_acc)
            print(f"Adv Test Accuracy for CAE:\t {adv_acc:.5f}")

        for lam in lambdas:
            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights(f"{MODEL_PATH}/{cnn_name}.hdf5", by_name=False)
            cnn_model(cnn_model.input)

            wrap = KerasModelWrapper(cnn_model)
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
            }

            cae_model = keras.models.load_model(f"{MODEL_PATH}/contractive_autoencoder_{lam}.hdf5")

            cnn_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=["accuracy"]
            )

            # Calculate test error
            x_test_reconstruction = cae_model.predict(x_test)
            _, acc = cnn_model.evaluate(x_test_reconstruction, y_test,
                                             batch_size=batch_size,
                                             verbose=0)
            cae_accuracies.append(acc)

            print(f"Test Accuracy for CAE:\t {acc:.5f}")

        np.savetxt("cae_accuracies.npy", np.array([cae_accuracies, cae_adv_accuracies]))

    # ======================================================================
    # Test with DAE
    # ======================================================================

    elif test_dae:
        dae_adv_accuracies = []
        dae_accuracies = []
        for v_noise in v_noises:
            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights(f"{MODEL_PATH}/{cnn_name}.hdf5", by_name=False)
            cnn_model(cnn_model.input)

            wrap = KerasModelWrapper(cnn_model)
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
            }

            dae_model = keras.models.load_model(f"{MODEL_PATH}/denoising_autoencoder_{v_noise}.hdf5")

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

            dae_model = keras.models.load_model(f"{MODEL_PATH}/denoising_autoencoder_{v_noise}.hdf5")

            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights(f"{MODEL_PATH}/{cnn_name}.hdf5", by_name=False)

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

    elif test_stacked_dae:
        stacked_dae_adv_accuracies = []
        stacked_dae_accuracies = []
        for v_noise in v_noises:
            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights(f"{MODEL_PATH}/{cnn_name}.hdf5", by_name=False)
            cnn_model(cnn_model.input)

            wrap = KerasModelWrapper(cnn_model)
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
            }

            stacked_dae_model = keras.models.load_model(f"{MODEL_PATH}/stacked_denoising_autoencoder_{num_stacks}_{v_noise}.hdf5")

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
            print(f"Adv Test Accuracy for {num_stacks} Stacked DAE vnoise:\t{v_noise:.1f}: {adv_acc:.5f}")
            stacked_dae_adv_accuracies.append(adv_acc)

        for v_noise in v_noises:

            stacked_dae_model = keras.models.load_model(f"{MODEL_PATH}/stacked_denoising_autoencoder_{num_stacks}_{v_noise}.hdf5")

            # define TF model graph
            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights(f"{MODEL_PATH}/{cnn_name}.hdf5", by_name=False)

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
            print(f"Test Accuracy for {num_stacks} Stacked DAE vnoise:\t{v_noise:.1f}: {acc:.5f}")
            stacked_dae_accuracies.append(acc)

        np.savetxt("stacked_dae_accuracies.npy", np.array([stacked_dae_accuracies, stacked_dae_adv_accuracies]))

    return report

if __name__ == "__main__":
    # set random seed
    tf.set_random_seed(42)

    # # Train CNN, and save model
    # # Also train CNN with adversarial training, and save model
    # run_mnist_adv(num_epochs=20,
    #               batch_size=BATCH_SIZE,
    #               learning_rate=LEARNING_RATE,
    #               run_cnn=True,
    #               adversarial_training=True,)

    # # Run CNN without adversarial training
    # run_mnist_adv(num_epochs=20,
    #               batch_size=BATCH_SIZE,
    #               learning_rate=LEARNING_RATE,
    #               test_cnn=True)

    # WITHOUT ADVERSARIAL TRAINING ================================

    # Generate adversarial results / accuracies with
    # Autoencoder preprocessing
    run_mnist_adv(run_cnn=False, # train CNN
                  adversarial_training=False,
                  test_ae=True, # test CNN with AE preprocessing
                  test_cae=False, # test CNN with CAE preprocessing
                  test_dae=False, # test CNN with DAE preprocessing
                  test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
                  v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])

    # Generate adversarial results / accuracies with
    # Contractive Autoencoder preprocessing
    run_mnist_adv(run_cnn=False, # train CNN
                  adversarial_training=False,
                  test_ae=False, # test CNN with AE preprocessing
                  test_cae=True, # test CNN with CAE preprocessing
                  test_dae=False, # test CNN with DAE preprocessing
                  test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
                  lambdas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

    # Generate adversarial results / accuracies with
    # Denoising Autoencoder preprocessing
    run_mnist_adv(run_cnn=False, # train CNN
                  adversarial_training=False,
                  test_ae=False, # test CNN with AE preprocessing
                  test_cae=False, # test CNN with CAE preprocessing
                  test_dae=True, # test CNN with DAE preprocessing
                  test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
                  v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])

    # Generate adversarial results / accuracies with
    # Stacked Denoising Autoencoder preprocessing (stack of 2)
    run_mnist_adv(run_cnn=False, # train CNN
                  adversarial_training=False,
                  test_ae=False, # test CNN with AE preprocessing
                  test_cae=False, # test CNN with CAE preprocessing
                  test_dae=False, # test CNN with DAE preprocessing
                  test_stacked_dae=True, # test CNN with Stacked DAE preprocessing
                  num_stacks=2,
                  v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])

    # Generate adversarial results / accuracies with
    # Stacked Denoising Autoencoder preprocessing (stack of 3)
    run_mnist_adv(run_cnn=False, # train CNN
                  adversarial_training=False,
                  test_ae=False, # test CNN with AE preprocessing
                  test_cae=False, # test CNN with CAE preprocessing
                  test_dae=False, # test CNN with DAE preprocessing
                  test_stacked_dae=True, # test CNN with Stacked DAE preprocessing
                  num_stacks=3,
                  v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])

    # WITH ADVERSARIAL TRAINING ================================

    # # Generate adversarial results / accuracies with
    # # Contractive Autoencoder preprocessing
    # run_mnist_adv(run_cnn=False, # train CNN
    #               adversarial_training=True,
    #               test_cae=True, # test CNN with CAE preprocessing
    #               test_dae=False, # test CNN with DAE preprocessing
    #               test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
    #               v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])
    #
    # # Generate adversarial results / accuracies with
    # # Denoising Autoencoder preprocessing
    # run_mnist_adv(run_cnn=False, # train CNN
    #               adversarial_training=True,
    #               test_cae=False, # test CNN with CAE preprocessing
    #               test_dae=True, # test CNN with DAE preprocessing
    #               test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
    #               v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])
    #
    # # Generate adversarial results / accuracies with
    # # Stacked Denoising Autoencoder preprocessing
    # run_mnist_adv(run_cnn=False, # train CNN
    #               adversarial_training=True,
    #               test_cae=False, # test CNN with CAE preprocessing
    #               test_dae=False, # test CNN with DAE preprocessing
    #               test_stacked_dae=True, # test CNN with Stacked DAE preprocessing
    #               v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])
