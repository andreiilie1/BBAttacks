from matplotlib import pyplot as plt
import numpy as np
from tqdm.notebook import tqdm
import json


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

    for i in range(len(adv_evo_strategy)):
        img = adv_evo_strategy[i].img

        if adv_evo_strategy[i].stop_criterion():
            count_succ += 1
            queries_succ.append(adv_evo_strategy[i].queries)
            l0_dists_succ.append(np.sum(adv_evo_strategy[i].get_best_candidate() != img))
            indices_succ.append(i)
        else:
            count_fail +=1
            index_fail.append(i)

    return {
        "count_succ": int(count_succ),
        "queries_succ": queries_succ,
        "l0_dists_succ": l0_dists_succ,
        "indices_succ": indices_succ,
        "count_fail": int(count_fail),
        "indices_fails": indices_fails,
        "queries_succ_mean": np.mean(queries_succ),
        "l0_dists_succ_mean": np.mean(l0_dists_succ)
    }


def print_evoba_stats(evoba_stats):
    SEP = "_" * 20
    count_succ = evoba_stats["count_succ"]
    count_fail = evoba_stats["count_fail"]
    count_total = count_succ + count_fail
    
    queries_succ_mean = evoba_stats["queries_succ_mean"]
    l0_dists_succ_mean = evoba_stats["l0_dists_succ_mean"]
    
    queries_succ = evoba_stats["queries_succ"]
    l0_dists_succ = evoba_stats["l0_dists_succ"]
    
    print()
    print("EvoBA STATS (L0 attack)")
    print(SEP)
    
    print(f"Perturbed successfully {count_succ}/{count_total} images")
    print(f"Average query count: {queries_succ_mean}")
    print(f"Average l0 distance: {l0_dists_succ_mean}")

    print()
    print(f"Median query count: {np.median(queries_succ)}")
    print(f"Median l0 dist: {np.median(l0_dists_succ)}")

    print()
    print(f"Max query count: {max(queries_succ)}")
    print(f"Max l0 dist: {max(l0_dists_succ)}")
    print(SEP)
    print()
    

def get_epsgreedy_stats(ega, unperturbed_images):
    l0_dists_succ = []
    queries_succ = []
    indices_succ = []
    indices_fails = []
    sample_size = len(unperturbed_images)
    for i in range(sample_size):
        if ega[i].is_perturbed():
            dist = (ega[i].img != unperturbed_images[i]).sum()
            l0_dists_succ.append(dist)
            queries_succ.append(ega[i].count_explorations)
            indices_succ.append(i)
        else:
            indices_fails.append(i)

    return {
        "count_succ": len(queries_succ),
        "queries_succ": queries_succ,
        "l0_dists_succ": l0_dists_succ,
        "indices_succ": indices_succ,
        "count_fail": len(indices_fails),
        "indices_fails": indices_fails,
        "queries_succ_mean": np.mean(queries_succ),
        "l0_dists_succ_mean": np.mean(l0_dists_succ)
    }
    
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_evoba_artifacts(evoba_stats, run_output_folder):
    with open(run_output_folder+"/evoba_l0_stats.json", 'w') as outfile:
        json.dump(dict(evoba_stats), outfile, cls=NpEncoder)
        
    np.save(run_output_folder+"/evoba_l0_stats.npy", evoba_stats)
    
    fig = plt.figure(figsize=(20, 14))
    plt.hist(evoba_stats["l0_dists_succ"])
    plt.title("EvoBA L0 distances histogram", fontsize=26)
    plt.xlabel("L0 distance", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder+"/evoba_l0_hist.png")
    
    fig = plt.figure(figsize=(20, 14))
    plt.hist(evoba_stats["queries_succ"])
    plt.title("EvoBA queries histogram", fontsize=26)
    plt.xlabel("Queries", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder+"/evoba_l0_queries_hist.png")


def save_original_perturbed_image_pairs(adv_evo_strategy):
    # TODO: add implementation that saves pairs of original and
    #  perturbed images, together with individual stats
    #  Decide whether some sampling should happen here, or in
    #  the caller - maybe in the caller is a better idea.
    return "NOT IMPLEMENTED"


def get_simba_stats(simba_wrapper):
    count_succ = sum(simba_wrapper.perturbed)
    cont_fail = len(simba_wrapper.perturbed) - count_succ

    queries_succ = list(np.array(simba_wrapper.queries)[simba_wrapper.perturbed])
    l2_dists_succ = list(np.array(simba_wrapper.l2_distances)[simba_wrapper.perturbed])

    return {
        "count_succ": int(count_succ),
        "count_fail": int(cont_fail),
        "queries_succ": queries_succ,
        "l2_dists_succ": l2_dists_succ,
        "queries_succ_mean": np.mean(queries_succ),
        "l2_dists_succ_mean": np.mean(l2_dists_succ)
    }


def print_simba_stats(simba_stats):
    SEP = "_" * 20
    count_succ = simba_stats["count_succ"]
    count_fail = simba_stats["count_fail"]
    count_total = count_succ + count_fail

    queries_succ_mean = simba_stats["queries_succ_mean"]
    l2_dists_succ_mean = simba_stats["l2_dists_succ_mean"]

    queries_succ = simba_stats["queries_succ"]
    l2_dists_succ = simba_stats["l2_dists_succ"]

    print()
    print("SimBA STATS (L2 attack)")
    print(SEP)

    print(f"Perturbed successfully {count_succ}/{count_total} images")
    print(f"Average query count: {queries_succ_mean}")
    print(f"Average l2 distance: {l2_dists_succ_mean}")

    print()
    print(f"Median query count: {np.median(queries_succ)}")
    print(f"Median l2 dist: {np.median(l2_dists_succ)}")

    print()
    print(f"Max query count: {max(queries_succ)}")
    print(f"Max l2 dist: {max(l2_dists_succ)}")
    print(SEP)
    print()


def save_simba_artifacts(simba_stats, run_output_folder):
    with open(run_output_folder + "/simba_l2_stats.json", 'w') as outfile:
        json.dump(dict(simba_stats), outfile, cls=NpEncoder)

    np.save(run_output_folder + "/simba_l2_stats.npy", simba_stats)

    fig = plt.figure(figsize=(20, 14))
    plt.hist(simba_stats["l2_dists_succ"])
    plt.title("SimBA L2 distances histogram", fontsize=26)
    plt.xlabel("L2 distance", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder + "/simba_l2_hist.png")

    fig = plt.figure(figsize=(20, 14))
    plt.hist(simba_stats["queries_succ"])
    plt.title("SimBA queries histogram", fontsize=26)
    plt.xlabel("Queries", fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.ylabel("Count images", fontsize=24)
    plt.savefig(run_output_folder + "/simba_l2_queries_hist.png")
