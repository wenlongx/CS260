# README

## Overview
For this project, I implemented several autoencoders and a classifier using Keras and Tensorflow. I then used the cleverhans library to generate adversarial examples to attack them.

The code consists of 4 files: `models.py`, `train_autoencoders.py`, `run_models.py`, `run_models_whitebox.py`. `models.py` includes all the code to generate the keras models, while `train_autoencoders.py` includes all the code to train and save the various autoencoder models that I tested, for various hyperparameters. `run_models.py` and `run_models_whitebox.py` include the code that loads these trained classifiers and autoencoders, and runs an attack using FGSM on them for the grey box and white box setting, respectively.

## Running the Code
To generate and train the models needed to run the attack, first create the directory `models` in the same directory that houses the code, and run the following:
```
python3 train_autoencoders.py
```
This will generate the following models in the `models` directory.
```
cnn.hdf5
autoencoder.hdf5
contractive_autoencoder_#.hdf5
denoising_autoencoder_*.hdf5
stacked_denoising_autoencoder_$_*.hdf5
```
Where `#` takes any of the values from `[0.1, 0.01, 0.001, 0.0001, 1e-05]`,  `*` takes any of the values from `[0.1, 0.2, 0.3, 0.4, 0.5]`, and `$` takes any of the values from `[2, 3]`.

Then, after you have generated the models, to run the grey-box adversarial attack:
```
python3 run_models.py
```
Then, after you have generated the models, to run the white-box adversarial attack:
```
python3 run_models_whitebox.py
```
