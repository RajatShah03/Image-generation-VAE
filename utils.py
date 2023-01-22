import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Preprocess image
def preprocess(data):
    return data.astype('float32') / 255.

# Preprocess with staticaly binarization
def preprocess_binarize(images):
    images = images[..., tf.newaxis]
    images = images.astype('float32') / 255.
    binarized = np.where(images > 0.5, 1.0, 0.0).astype('float32')
    return binarized


# Add random noise
def add_noise(data, noise_factor=0.2):
    # Expand dimension to add 1 channel
    data = data[..., tf.newaxis]
    
    # Add random gaussian noise
    data_noisy = data + noise_factor * tf.random.normal(shape=data.shape)
    
    # Clip result
    data_noisy = tf.clip_by_value(data_noisy, clip_value_min=0., clip_value_max=1.)
    
    return data_noisy
   
# Plot loss    
def plot_loss(history):
    plt.plot(range(len(history['loss'])), history['loss'], label='Training loss')
    plt.plot(range(len(history['val_loss'])), history['val_loss'], label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()