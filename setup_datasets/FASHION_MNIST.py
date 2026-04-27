print("Importing Fashion MNIST data...")
from time import time
s = time()

import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.datasets import fashion_mnist#type: ignore
import numpy as np

# Load Fashion MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values to 0-1 range

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# mean = np.mean(x_train)
# std = np.std(x_train)
# x_train = (x_train - mean) / std
# x_test = (x_test.astype('float32') - mean) / std

# Flatten images from 28x28 to 784 features
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# print(x_train.shape, x_test.shape)
# One-hot encode labels
def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

y_train_one_hot = np.array(to_one_hot(y_train))
y_test_one_hot = np.array(to_one_hot(y_test))

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(f"Done, took {round(time()-s, 3)}s to run.\n")
