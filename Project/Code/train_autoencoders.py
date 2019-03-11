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

def train_stacked_dae(num_epochs=NUM_EPOCHS, num_pretrain_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, testing=False, learning_rate=LEARNING_RATE, v_noise=0.3):

    # can use gpu
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )

    # Create TF session and set Keras backend session as TF
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Get MNIST test data
    mnist = MNIST()
    x_train, y_train = mnist.get_set("train")
    x_test, y_test = mnist.get_set("test")

    # corrupt inputs
    noise_train = v_noise * np.random.normal(size=x_train.shape)
    noise_test = v_noise * np.random.normal(size=x_test.shape)
    x_train_noisy = np.clip(x_train + noise_train, 0.0, 1.0)
    x_test_noisy = np.clip(x_test + noise_test, 0.0, 1.0)

    # Obtain image params
    n_rows, n_cols, n_channels = x_train.shape[1:4]
    n_classes = y_train.shape[1]

    # ================================================================
    # Pretrain first autoencoder later
    # define TF model graph
    model_1 = DenoisingAutoencoder((n_rows, n_cols, n_channels))
    model_1.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse'
    )
    # Train an MNIST model
    model_1.fit(x_train_noisy, x_train,
                batch_size=batch_size,
                epochs=num_pretrain_epochs,
                validation_data=(x_test_noisy, x_test),
                verbose=1)

    # Evaluate the accuracy on legitimate and adversarial test examples
    x_train_denoised_1 = model_1.predict(x_train_noisy,
                                    batch_size=batch_size,
                                    verbose=0)
    # Evaluate the accuracy on legitimate and adversarial test examples
    x_test_denoised_1 = model_1.predict(x_test_noisy,
                                    batch_size=batch_size,
                                    verbose=0)

    # ================================================================
    # Pretrain second autoencoder later

    # corrupt inputs
    noise_train_2 = v_noise * np.random.normal(size=x_train.shape)
    noise_test_2 = v_noise * np.random.normal(size=x_test.shape)
    x_train_noisy_2 = np.clip(x_train_denoised_1 + noise_train_2, 0.0, 1.0)
    x_test_noisy_2 = np.clip(x_test_denoised_1 + noise_test_2, 0.0, 1.0)

    # define TF model graph
    model_2 = DenoisingAutoencoder((n_rows, n_cols, n_channels))
    model_2.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse'
    )
    # Train an MNIST model
    model_2.fit(x_train_noisy_2, x_train_denoised_1,
                batch_size=batch_size,
                epochs=num_pretrain_epochs,
                validation_data=(x_test_noisy_2, x_test_denoised_1),
                verbose=1)

    # Evaluate the accuracy on legitimate and adversarial test examples
    x_train_denoised_2 = model_2.predict(x_train_noisy_2,
                                    batch_size=batch_size,
                                    verbose=0)
    # Evaluate the accuracy on legitimate and adversarial test examples
    x_test_denoised_2 = model_2.predict(x_test_noisy_2,
                                    batch_size=batch_size,
                                    verbose=0)

    # ================================================================
    # Pretrain third autoencoder later

    # corrupt inputs
    noise_train_3 = v_noise * np.random.normal(size=x_train.shape)
    noise_test_3 = v_noise * np.random.normal(size=x_test.shape)
    x_train_noisy_3 = np.clip(x_train_denoised_2 + noise_train_3, 0.0, 1.0)
    x_test_noisy_3 = np.clip(x_test_denoised_2 + noise_test_3, 0.0, 1.0)

    # define TF model graph
    model_3 = DenoisingAutoencoder((n_rows, n_cols, n_channels))
    model_3.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse'
    )
    # Train an MNIST model
    model_3.fit(x_train_noisy_3, x_train_denoised_2,
                batch_size=batch_size,
                epochs=num_pretrain_epochs,
                validation_data=(x_test_noisy_3, x_test_denoised_2),
                verbose=1)

    # Evaluate the accuracy on legitimate and adversarial test examples
    x_train_denoised_3 = model_3.predict(x_train_noisy_3,
                                    batch_size=batch_size,
                                    verbose=0)
    # Evaluate the accuracy on legitimate and adversarial test examples
    x_test_denoised_3 = model_3.predict(x_test_noisy_3,
                                    batch_size=batch_size,
                                    verbose=0)

    # ================================================================
    # Create Stacked Denoising Autoencoder
    # define TF model graph
    model = StackedDenoisingAutoencoder((n_rows, n_cols, n_channels), 3)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='mse'
    )
    model = transfer_weights_stacked_dae(model, [model_1, model_2, model_3])

    # Train an MNIST model
    model.fit(x_train_noisy, x_train,
                batch_size=batch_size,
                epochs=num_epochs,
                validation_data=(x_test_noisy, x_test),
                verbose=1)

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
        f"models/stacked_denoising_autoencoder_{v_noise}.hdf5",
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
        f"models/denoising_autoencoder_{v_noise}.hdf5",
        overwrite=True,
        include_optimizer=True
    )


if __name__ == "__main__":
    # set random seed
    tf.set_random_seed(42)

    # Train Denoising Autoencoder Model
    train_stacked_dae(num_epochs=30, num_pretrain_epochs=10, testing=False, v_noise=0.3)

    # Train Denoising Autoencoder Model
    train_dae(num_epochs=30, testing=False, v_noise=0.3)
