#! -*- coding: utf-8 -*-


'''
VAE implemented by using keras (TensorFlow as backend)

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import to_categorical

batch_size = 100
original_dim = 784
latent_dim = 2 # Set to 2 in order to visualize the distributions of NUMBERS
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0
num_classes = 10

# mnist_model_path = ''

# Load MNIST dataset
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
# If you want to use local mnist model
# (x_train, y_train_), (x_test, y_test_) = mnist.load_data(path=mnist_model_path)
x_train = x_train.astype('float32') / 255. # (60000, 28, 28)
x_test = x_test.astype('float32') / 255. # (10000, 28, 28)
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  # (60000, 784)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) # (10000, 784)
y_train = to_categorical(y_train_, num_classes) # (60000, 10)
y_test = to_categorical(y_test_, num_classes) # (10000, 10)

x = Input(shape=(original_dim,)) # Tensor("input_1:0", shape=(?, 784), dtype=float32)
h = Dense(intermediate_dim, activation='relu')(x) # Tensor("dense_1/Relu:0", shape=(?, 256), dtype=float32), Densely-connected layer

# Compute the mean and std of p(Z|X)
z_mean = Dense(latent_dim)(h)  # (?, 2)
z_log_var = Dense(latent_dim)(h) # (?, 2)
print('z_mean is {}'.format(z_log_var)) 

# reparameter trick 
# refer to (https://kexue.fm/archives/5253#%E9%87%8D%E5%8F%82%E6%95%B0%E6%8A%80%E5%B7%A7)
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std) # (?, 2)
    print('epsilon_shape is {}, {}'.format(epsilon.shape, epsilon))

    return z_mean + K.exp(z_log_var / 2) * epsilon

# reparameter layer, equals to add noise to input data
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var]) # (?, 2)

# decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# create model
vae = Model(x, x_decoded_mean)

# xent_loss is reconstruct loss，kl_loss is KL loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss is designed to add other kinds of loss more easily
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

# Construct encoder, then navigate the NUMBER distributions at latent space
encoder = Model(x, z_mean)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
plt.colorbar()
plt.show()

# Construct decoder
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# See how the dim change of latent vector influence the output
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

# Construct pairs of latent vector by using the medium of norm-dist
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = generator.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
