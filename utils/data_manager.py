import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
import numpy as np
import tensorflow.keras.utils

# test.npz and not_robust.npz were generated from train_mnist using 28 pixels perturbations (sqrt(number_pixels)), which were assigned random values
def load_data(name = "mnist", filename_robust = "test.npz", filename_not_robust = "not_robust.npz"):
    if(name == "cifar10"):
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    elif(name == "mnist"):
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = np.array(tf.expand_dims(x_train, 3))
        x_test = np.array(tf.expand_dims(x_test, 3))

        y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    elif(name == "mnist_robust"):
        num_classes = 10
        _, (x_test, y_test) = mnist.load_data()
        npz = np.load(filename_robust)
        
        x_train = npz["arr_0"]
        y_train = npz["arr_1"]
        
        x_test = tf.expand_dims(x_test, 3)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    elif(name == "mnist_not_robust"):
        num_classes = 10
        _, (x_test, y_test) = mnist.load_data()
        npz = np.load(filename_not_robust)
        
        x_train = npz["arr_0"]
        y_train = npz["arr_1"]
        
        x_test = tf.expand_dims(x_test, 3)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    elif(name == "ordered_robust_not_robust"):
        num_classes = 10
        _, (x_test, y_test) = mnist.load_data()
        npz_robust = np.load(filename_robust)
        
        x_train_robust = npz_robust["arr_0"]
        y_train_robust = npz_robust["arr_1"]
        
        npz_not_robust = np.load(filename_not_robust)
        
        x_train_not_robust = npz_not_robust["arr_0"]
        y_train_not_robust = npz_not_robust["arr_1"]
        
        x_train = np.append(x_train_robust, x_train_not_robust, axis = 0)
        y_train = np.append(y_train_robust, y_train_not_robust, axis = 0)
        
        x_test = tf.expand_dims(x_test, 3)
        y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
    else:
        raise Exception("Invalid data name")
        
    return (x_train, y_train), (x_test, y_test)