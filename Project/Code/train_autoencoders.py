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

from cleverhans.dataset import MNIST

from models import *

NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001

MODEL_PATH = "models"

def train_cae(num_epochs=NUM_EPOCHS, num_pretrain_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, testing=False, learning_rate=LEARNING_RATE, lam=1e-4):

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
    model = ContractiveAutoencoder((n_rows, n_cols, n_channels))
    contractive_loss = get_contractive_loss(model, lam)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=contractive_loss,
        metrics=['mse']
    )
    model.summary()
    sess.run(tf.global_variables_initializer())

    # Train an MNIST model
    model.fit(x_train, x_train,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(x_test, x_test),
              verbose=1)

    # Evaluate the accuracy on legitimate and adversarial test examples
    x_test_reconstruction = model.predict(x_test,
                                    batch_size=batch_size,
                                    verbose=0)

    first_img = np.reshape(x_test_reconstruction[0], (28, 28))
    plt.imsave("temp.png", first_img)

    # # Display the 1st 8 corrupted and denoised images
    # rows, cols = 10, 30
    # num = rows * cols
    # imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_test_denoised[:num]])
    # imgs = imgs.reshape((rows * 3, cols, n_rows, n_cols))
    # imgs = np.vstack(np.split(imgs, rows, axis=1))
    # imgs = imgs.reshape((rows * 3, -1, n_rows, n_cols))
    # imgs = np.vstack([np.hstack(i) for i in imgs])
    # imgs = (imgs * 255).astype(np.uint8)
    # Image.fromarray(imgs).save('corrupted_and_denoised.png')

    # Save model locally
    keras.models.save_model(
        model,
        f"{MODEL_PATH}/contractive_autoencoder_{lam}.hdf5",
        overwrite=True,
        include_optimizer=False
    )

def train_stacked_dae(num_epochs=NUM_EPOCHS, num_pretrain_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, testing=False, learning_rate=LEARNING_RATE, v_noise=0.3, num_stacks=3):

    # can use gpu
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )

    # Create TF session and set Keras backend session as TF
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Get MNIST test data
    mnist = MNIST()
    x_train, y_train = mnist.get_set("train")
    x_test, y_test = mnist.get_set("test")

    def corrupt(x):
        noisy_x = v_noise * np.random.normal(size=x.shape)
        return np.clip(x + noisy_x, 0.0, 1.0)

    # Obtain image params
    n_rows, n_cols, n_channels = x_train.shape[1:4]
    n_classes = y_train.shape[1]


    # Pretrain the autoencoders
    models = []
    x_trains = [x_train]
    x_tests = [x_test]
    for i in range(num_stacks):
        # generate corrupted inputs
        x_train_noisy = corrupt(x_trains[i])
        x_test_noisy = corrupt(x_tests[i])

        pretrain_layer_path = f"{MODEL_PATH}/pretrain_sdae_layer_{i}_{v_noise}.hdf5"

        if tf.gfile.Exists(pretrain_layer_path):
            model = keras.models.load(pretrain_layer_path)

        else:
            # Pretrain autoencoder
            # define TF model graph
            model = DenoisingAutoencoder((n_rows, n_cols, n_channels))

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss='mse'
            )

            # pretrain on MNIST
            model.fit(x_train_noisy, x_trains[i],
                        batch_size=batch_size,
                        epochs=num_pretrain_epochs,
                        validation_data=(x_test_noisy, x_tests[i]),
                        verbose=1)

            # save the pretrained model for use in later ones
            keras.models.save_model(
                model,
                pretrain_layer_path,
                overwrite=True,
                include_optimizer=True,
            )

        models.append(model)

        # Evaluate the accuracy on legitimate and adversarial test examples
        x_train_denoised = models[i].predict(x_train_noisy,
                                        batch_size=batch_size,
                                        verbose=0)
        # Evaluate the accuracy on legitimate and adversarial test examples
        x_test_denoised = models[i].predict(x_test_noisy,
                                        batch_size=batch_size,
                                        verbose=0)

        x_trains.append(x_train_denoised)
        x_tests.append(x_test_denoised)

    # ================================================================
    # Create Stacked Denoising Autoencoder
    # define TF model graph
    model = StackedDenoisingAutoencoder((n_rows, n_cols, n_channels), num_stacks=num_stacks)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse'
    )
    model = transfer_weights_stacked_dae(model, models)

    # Train an MNIST model
    model.fit(x_train_noisy, x_train,
                batch_size=batch_size,
                epochs=num_epochs,
                validation_data=(x_test_noisy, x_test),
                verbose=1)

    x_train_noisy = corrupt(x_train)
    x_test_noisy = corrupt(x_test)

    # Evaluate the accuracy on legitimate and adversarial test examples
    x_train_denoised = model.predict(x_train_noisy,
                                    batch_size=batch_size,
                                    verbose=0)
    # Evaluate the accuracy on legitimate and adversarial test examples
    x_test_denoised = model.predict(x_test_noisy,
                                    batch_size=batch_size,
                                    verbose=0)

    first_img = np.reshape(x_test_denoised[0], (28, 28))
    plt.imsave("temp.png", first_img)

    # # Display the 1st 8 corrupted and denoised images
    # rows, cols = 10, 30
    # num = rows * cols
    # imgs = np.concatenate([x_test[:num], x_test_noisy[:num], x_test_denoised[:num]])
    # imgs = imgs.reshape((rows * 3, cols, n_rows, n_cols))
    # imgs = np.vstack(np.split(imgs, rows, axis=1))
    # imgs = imgs.reshape((rows * 3, -1, n_rows, n_cols))
    # imgs = np.vstack([np.hstack(i) for i in imgs])
    # imgs = (imgs * 255).astype(np.uint8)
    # Image.fromarray(imgs).save('corrupted_and_denoised.png')

    # Save model locally
    keras.models.save_model(
        model,
        f"{MODEL_PATH}/stacked_denoising_autoencoder_{num_stacks}_{v_noise}.hdf5",
        overwrite=True,
        include_optimizer=True
    )


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

    # define TF model graph
    model = DenoisingAutoencoder((n_rows, n_cols, n_channels))
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

    # Evaluate the accuracy on legitimate and adversarial test examples
    x_test_denoised = model.predict(x_test_noisy,
                                    batch_size=batch_size,
                                    verbose=0)

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
    Image.fromarray(imgs).save('corrupted_and_denoised.png')

    # Save model locally
    keras.models.save_model(
        model,
        f"{MODEL_PATH}/denoising_autoencoder_{v_noise}.hdf5",
        overwrite=True,
        include_optimizer=True
    )


def train_ae(num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, testing=False, learning_rate=LEARNING_RATE):

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
    model = DenoisingAutoencoder((n_rows, n_cols, n_channels))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse'
    )

    # Train an MNIST model
    model.fit(x_train, x_train,
              batch_size=batch_size,
              epochs=num_epochs,
              validation_data=(x_test, x_test),
              verbose=1)

    # Evaluate the accuracy on legitimate and adversarial test examples
    x_test_recon = model.predict(x_test,
                                batch_size=batch_size,
                                verbose=0)

    # Save model locally
    keras.models.save_model(
        model,
        f"{MODEL_PATH}/autoencoder.hdf5",
        overwrite=True,
        include_optimizer=True
    )


if __name__ == "__main__":
    # set random seed
    tf.random.set_random_seed(1234)

    # Train Conv Autoencoder
    train_ae(num_epochs=20, testing=False)

    # # Train Contractive Autoencoder
    for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        train_cae(num_epochs=20, testing=False, lam=lam)

    # # Train Denoising Autoencoder Model
    # for v_noise in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     train_dae(num_epochs=20, testing=False, v_noise=v_noise)

    # # Train Stacked Denoising Autoencoder Models
    # for num_stacks in [2, 3]:
    #     for v_noise in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #         train_stacked_dae(num_epochs=20, num_pretrain_epochs=10, testing=False, v_noise=v_noise, num_stacks=num_stacks)
