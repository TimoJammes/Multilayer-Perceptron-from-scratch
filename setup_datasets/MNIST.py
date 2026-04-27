print("Importing MNIST data...")
from time import time
s = time()

import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.datasets import mnist#type: ignore
import numpy as np
# from copy import deepcopy
# from scipy.ndimage import rotate

# from rotate import rotate_batch
# from shift import augment_with_shifts

import numpy as np

# def get_margins(img):
#     nz = img > 0
#     top = np.argmax(np.any(nz, axis=1))
#     bottom = np.argmax(np.any(nz[::-1], axis=1))
#     left = np.argmax(np.any(nz, axis=0))
#     right = np.argmax(np.any(nz[:, ::-1], axis=0))
#     return top, bottom, left, right

# def get_all_margins(imgs):
#     nz = imgs > 0  # (N,28,28)
#     top = np.argmax(np.any(nz, axis=2), axis=1)
#     bottom = np.argmax(np.any(nz[:, ::-1, :], axis=2), axis=1)
#     left = np.argmax(np.any(nz, axis=1), axis=1)
#     right = np.argmax(np.any(nz[:, :, ::-1], axis=1), axis=1)

#     return np.stack([top, bottom, left, right], axis=1)

# def shift_each_image(imgs, shifts):
#     out = np.empty_like(imgs)
#     for i, (dy, dx) in enumerate(shifts):
#         out[i] = np.roll(imgs[i], shift=(dy, dx), axis=(0, 1))
#     return out

def sample_first_axis(arr, n, replace=False, seed=None):
    """
    Sample n elements along the first axis of an ND array.
    
    Parameters:
        arr: np.ndarray, shape (N, ...)
        n: int, number of elements to sample along axis 0
        replace: bool, allow repeated rows
        seed: int, optional random seed
    
    Returns:
        np.ndarray of shape (n, ...) containing the sampled rows
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(arr.shape[0], size=n, replace=replace)
    return arr[indices]

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to 0-1 range
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# x_train_unflattened = deepcopy(x_train)
# x_test_unflattened = deepcopy(x_test)

# Flatten images from 28x28 to 784 features
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)


# rots = [0, -10, 10]
# x_train_rotations = rotate_batch(x_train_unflattened, rots)
# x_test_rotations = rotate_batch(x_test_unflattened, rots)

# y_train_rotations = np.tile(y_train, len(rots))
# y_test_rotations = np.tile(y_test, len(rots))

# x_train_rotations_shifts = augment_with_shifts(sample_first_axis(x_train_rotations.reshape(-1, 28, 28), int(60000*len(rots)/6)))
# x_test_rotations_shifts = augment_with_shifts(sample_first_axis(x_test_rotations.reshape(-1, 28, 28), int(60000*len(rots)/6)))

# y_train_rotations_shifts = np.tile(y_train_rotations, 9)
# y_test_rotations_shifts = np.tile(y_test_rotations, 9)

# # print(np.shape(x_train_rotations_shifts))#, np.shape(y_train_rotations_shifts))

# # np.random.shuffle(x_train_rotations_shifts)


# indices_train = np.random.permutation(x_train_rotations.shape[0])
# indices_test = np.random.permutation(x_test_rotations.shape[0])

# x_train_rotations = x_train_rotations[indices_train]
# y_train_rotations = y_train_rotations[indices_train]
# x_test_rotations = x_test_rotations[indices_test]
# y_test_rotations = y_test_rotations[indices_test]


# indices_train = np.random.permutation(x_train_rotations_shifts.shape[0])
# indices_test = np.random.permutation(x_test_rotations_shifts.shape[0])

# x_train_rotations_shifts = x_train_rotations_shifts[indices_train]
# y_train_rotations_shifts = y_train_rotations_shifts[indices_train]
# x_test_rotations_shifts = x_test_rotations_shifts[indices_test]
# y_test_rotations_shifts = y_test_rotations_shifts[indices_test]


def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

y_train_one_hot = np.array(to_one_hot(y_train))
# y_train_rotations_one_hot = to_one_hot(y_train_rotations)
y_test_one_hot = np.array(to_one_hot(y_test))
# y_test_rotations_one_hot = to_one_hot(y_test_rotations)

# y_train_rotations_shifts_one_hot = to_one_hot(y_test_rotations_shifts)
# y_test_rotations_shifts_one_hot = to_one_hot(y_test_rotations_shifts)


print(f"Done, took {round(time()-s, 3)}s to run.\n")
