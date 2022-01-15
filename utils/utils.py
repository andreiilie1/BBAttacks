from matplotlib import pyplot as plt
import numpy as np


def cap(value, inf, sup):
    if value <= inf:
        return inf
    if value >= sup:
        return sup
    return value


def plot_digits(X, Y):
    for i in range(20):
        plt.subplot(5, 4, i+1)
        plt.tight_layout()
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title('Digit:{}'.format(Y[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
    
def top_k_accuracy(y_true, y_pred, k=1):
    '''From: https://github.com/chainer/chainer/issues/606
    
    Expects both y_true and y_pred to be one-hot encoded.
    '''
    argsorted_y = np.argsort(y_pred)[:,-k:]
    return np.any(argsorted_y.T == y_true.argmax(axis=1), axis=0).mean()


def generate_metrics(ega, unperturbed_images, sample_size):
    l0_dists_succ = []
    queries_succ = []
    samples_succ = []
    samples_fail = []
    for i in range(sample_size):
        if ega[i].is_perturbed():
            l0_dist = (ega[i].img != unperturbed_images[i]).sum()
            l0_dists_succ.append(l0_dist)
            queries_succ.append(ega[i].count_explorations)
            samples_succ.append(i)
        else:
            samples_fail.append(i)
    return {
        "l0_dists_succ": l0_dists_succ,
        "queries_succ": queries_succ,
        "samples_succ": samples_succ,
        "samples_fail": samples_fail
    }
