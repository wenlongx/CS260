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
LEARNING_RATE = 0.002

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_mnist_adv(num_epochs=NUM_EPOCHS,
                  batch_size=BATCH_SIZE,
                  learning_rate=LEARNING_RATE,
                  test_cae=False, # test CNN with CAE preprocessing
                  test_dae=False, # test CNN with DAE preprocessing
                  test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
                  v_noises=[0.1, 0.2, 0.3, 0.4, 0.5],
                  lambdas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]):

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


    cnn_name = "cnn"

    # ======================================================================
    # Test with CAE
    # ======================================================================
    if test_cae:
        cae_adv_accuracies = []
        cae_accuracies = []
        for lam in lambdas:
            cae_model = ContractiveAutoencoder((n_rows, n_cols, n_channels))
            cae_model.load_weights(f"models/contractive_autoencoder_{lam}.hdf5", by_name=False)

            final_out = ConvNet((n_rows, n_cols, n_channels), n_classes, concat=True, concat_layer = cae_model.output)

            combined_model = Model(inputs=cae_model.input,
                                   outputs=final_out)

            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights(f"models/{cnn_name}.hdf5", by_name=False)

            num_cae_layers = len(cae_model.layers)
            num_cnn_layers = len(cnn_model.layers)
            for i in range(len(combined_model.layers)):
                if i < num_cae_layers:
                    weights = cae_model.layers[i].get_weights()
                    combined_model.layers[i].set_weights(weights)
                else:
                    weights = cnn_model.layers[i - num_cae_layers + 1].get_weights()
                    combined_model.layers[i].set_weights(weights)

            combined_model(combined_model.input)
            wrap = KerasModelWrapper(combined_model)
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
            }
            adv_acc_metric = get_adversarial_acc_metric(combined_model, fgsm, fgsm_params)
            combined_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', adv_acc_metric]
            )

            _, acc, adv_acc = combined_model.evaluate(x_test, y_test,
                                             batch_size=batch_size,
                                             verbose=0)

            cae_accuracies.append(acc)
            cae_adv_accuracies.append(adv_acc)
            print(f"Lambda = {lam}")
            print('Test accuracy on legitimate examples: %0.4f' % acc)
            print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

        np.savetxt("cae_accuracies_whitebox.npy", np.array([cae_accuracies, cae_adv_accuracies]))


    # ======================================================================
    # Test with DAE
    # ======================================================================
    if test_dae:
        dae_adv_accuracies = []
        dae_accuracies = []
        for v_noise in v_noises:
            dae_model = DenoisingAutoencoder((n_rows, n_cols, n_channels))
            dae_model.load_weights(f"models/denoising_autoencoder_{v_noise}.hdf5", by_name=False)

            final_out = ConvNet((n_rows, n_cols, n_channels), n_classes, concat=True, concat_layer = dae_model.output)

            combined_model = Model(inputs=dae_model.input,
                                   outputs=final_out)

            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights(f"models/{cnn_name}.hdf5", by_name=False)

            num_dae_layers = len(dae_model.layers)
            num_cnn_layers = len(cnn_model.layers)
            for i in range(len(combined_model.layers)):
                if i < num_dae_layers:
                    weights = dae_model.layers[i].get_weights()
                    combined_model.layers[i].set_weights(weights)
                else:
                    weights = cnn_model.layers[i - num_dae_layers + 1].get_weights()
                    combined_model.layers[i].set_weights(weights)

            combined_model(combined_model.input)
            wrap = KerasModelWrapper(combined_model)
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
            }
            adv_acc_metric = get_adversarial_acc_metric(combined_model, fgsm, fgsm_params)
            combined_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', adv_acc_metric]
            )

            _, acc, adv_acc = combined_model.evaluate(x_test, y_test,
                                             batch_size=batch_size,
                                             verbose=0)

            dae_accuracies.append(acc)
            dae_adv_accuracies.append(adv_acc)
            print(f"V_noise = {v_noise}")
            print('Test accuracy on legitimate examples: %0.4f' % acc)
            print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

        np.savetxt("dae_accuracies_whitebox.npy", np.array([dae_accuracies, dae_adv_accuracies]))


    # ======================================================================
    # Test with Stacked DAE
    # ======================================================================
    if test_stacked_dae:
        stacked_dae_adv_accuracies = []
        stacked_dae_accuracies = []
        for v_noise in v_noises:
            stacked_dae_model = StackedDenoisingAutoencoder((n_rows, n_cols, n_channels), 3)
            stacked_dae_model.load_weights(f"models/stacked_denoising_autoencoder_{v_noise}.hdf5", by_name=False)

            final_out = ConvNet((n_rows, n_cols, n_channels), n_classes, concat=True, concat_layer = stacked_dae_model.output)

            combined_model = Model(inputs=stacked_dae_model.input,
                                   outputs=final_out)

            cnn_model = ConvNet((n_rows, n_cols, n_channels), n_classes)
            cnn_model.load_weights(f"models/{cnn_name}.hdf5", by_name=False)

            num_stacked_dae_layers = len(stacked_dae_model.layers)
            num_cnn_layers = len(cnn_model.layers)
            for i in range(len(combined_model.layers)):
                if i < num_stacked_dae_layers:
                    weights = stacked_dae_model.layers[i].get_weights()
                    combined_model.layers[i].set_weights(weights)
                else:
                    weights = cnn_model.layers[i - num_stacked_dae_layers + 1].get_weights()
                    combined_model.layers[i].set_weights(weights)

            combined_model(combined_model.input)
            wrap = KerasModelWrapper(combined_model)
            fgsm = FastGradientMethod(wrap, sess=sess)
            fgsm_params = {
                'eps': 0.3,
                'clip_min': 0.,
                'clip_max': 1.
            }
            adv_acc_metric = get_adversarial_acc_metric(combined_model, fgsm, fgsm_params)
            combined_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy', adv_acc_metric]
            )

            _, acc, adv_acc = combined_model.evaluate(x_test, y_test,
                                             batch_size=batch_size,
                                             verbose=0)

            stacked_dae_accuracies.append(acc)
            stacked_dae_adv_accuracies.append(adv_acc)
            print(f"V_noise = {v_noise}")
            print('Test accuracy on legitimate examples: %0.4f' % acc)
            print('Test accuracy on adversarial examples: %0.4f\n' % adv_acc)

        np.savetxt("stacked_dae_accuracies_whitebox.npy", np.array([stacked_dae_accuracies, stacked_dae_adv_accuracies]))


    return report

if __name__ == "__main__":
    # set random seed
    tf.set_random_seed(42)

    # WITHOUT ADVERSARIAL TRAINING ================================

    # Generate adversarial results / accuracies with
    # Contractive Autoencoder preprocessing
    run_mnist_adv(test_cae=True, # test CNN with CAE preprocessing
                  test_dae=False, # test CNN with DAE preprocessing
                  test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
                  lambdas=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

    # Generate adversarial results / accuracies with
    # Denoising Autoencoder preprocessing
    run_mnist_adv(test_cae=False, # test CNN with CAE preprocessing
                  test_dae=True, # test CNN with DAE preprocessing
                  test_stacked_dae=False, # test CNN with Stacked DAE preprocessing
                  v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])

    # Generate adversarial results / accuracies with
    # Stacked Denoising Autoencoder preprocessing
    run_mnist_adv(test_cae=False, # test CNN with CAE preprocessing
                  test_dae=False, # test CNN with DAE preprocessing
                  test_stacked_dae=True, # test CNN with Stacked DAE preprocessing
                  v_noises=[0.1, 0.2, 0.3, 0.4, 0.5])
