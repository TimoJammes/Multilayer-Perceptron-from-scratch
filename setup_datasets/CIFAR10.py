import numpy as np

from pathlib import Path
import warnings

SCRIPT_DIR = Path(__file__).parent

def unpickle(file):
    import pickle
    file = SCRIPT_DIR / file
    with open(file, 'rb') as fo:
        with warnings.catch_warnings():
            # warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
            warnings.filterwarnings("ignore", message=".*align.*")
            dict = pickle.load(fo, encoding='bytes')
    return dict

def to_one_hot(labels, num_classes=10):
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

dic_train_1:dict = unpickle("cifar-10-batches-py/data_batch_1")
dic_test:dict = unpickle("cifar-10-batches-py/test_batch")

x_train = dic_train_1[b"data"]
y_train = dic_train_1[b"labels"]

for i in range(2, 6):    
    train_dic = unpickle("cifar-10-batches-py/data_batch_"+str(i))
    x_train = np.vstack([x_train, train_dic[b"data"]])
    y_train += train_dic[b"labels"]

x_train = x_train.astype('float32') / 255
x_test = dic_test[b"data"].astype('float32') / 255

y_test = dic_test[b"labels"]

y_train = np.array(y_train)
y_test = np.array(y_test)

# print(x_train.shape)
# print(len(y_train))
y_train_one_hot = np.array(to_one_hot(y_train))
y_test_one_hot = np.array(to_one_hot(y_test))


class_names = unpickle("cifar-10-batches-py/batches.meta")[b"label_names"]

class_names = [n.decode() for n in class_names]
