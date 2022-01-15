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


def get_evoba_stats(adv_evo_strategy):
    count_succ = 0
    queries_succ = []
    l0_dists_succ = []
    indices_succ = []

    count_fail = 0
    indices_fails = []

    for i in tqdm(range(len(adv_evo_strategy))):
        img = adv_evo_strategy[i].img

        if adv_evo_strategy[i].stop_criterion():
            count_succ += 1
            queries_succ.append(adv_evo_strategy[i].queries)
            l0_dists_succ.append(np.sum(adv_evo_strategy[i].get_best_candidate() != img))
            index_succ.append(i)
        else:
            count_fail +=1
            index_fail.append(i)
        
    return {
        "count_succ": count_succ,
        "queries_succ": queries_succ,
        "l0_dists_succ": l0_dists_succ, 
        "indices_succ": indices_succ,
        "count_fail": count_fail,
        "indices_fails": indices_fails
    }
