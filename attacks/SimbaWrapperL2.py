import numpy as np
import random

import sys
sys.path.append("../utils/")
import utils

import matplotlib.pyplot as plt
import json
import math

from tqdm import tqdm

# A wrapper class that performs the SimBA attack - pixel version
# It will try to perturb the 'model' classification of images 'X' with 
# labels 'y', by modifying random pixels with +/- epsilon steps. 
# Given an image and its label, it stops unsuccesfully after one of the following conditions is satisfied:
#     - max_queries unsuccessful calls to the model
#     - max_iteration iterations of the model (set by default to some large value, 
#                                  this condition shouldn't hold, it's more like a safety net)
#     - the image has sufferred modications of max_l0_distance pixels
# It stops succesfully for an image as soon as it is not longer classified to have label y.
class SimbaWrapper():
    def __init__(self, model, X, y, epsilon, max_queries, max_iterations=100, max_l2_distance=1000, checkpoints=True,
                 folder="saved_experiments_simba", verbose=False, reshape_flag=False, reshape=(28, 28), max_value=1):
        self.model = model
        self.X = X
        self.y = y
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.max_queries = max_queries
        self.max_l2_distance = max_l2_distance
        self.folder = folder
        self.X_modified = []
        self.verbose = verbose
        self.reshape_flag = reshape_flag
        self.reshape = reshape
        self.max_value = max_value
        
        # At the end of a SimbaWrapper.run_simba run, self.queries[i] will contain the number of queries
        # to the model for image X[i] until the attack stopped (with or without success); 
        # self.perturbed[i] will contain a True/False flag, stating whether X[i] was succesfully perturbed
        # within the max_itertaions, max_queries, and max_l0_distance limits;
        # and self.l2_distances[i] will contain the l2 distance at which the attack stopped for image X[i]
        # (be it with or without success).
        self.queries = []
        self.perturbed = []
        self.l2_distances = []

    @staticmethod
    def stop_condition_success(prob_distribution, label):
        if np.argmax(prob_distribution) != label:
            return True
        return False

    def run_simba(self):
        verbose = self.verbose
        for index in tqdm(range(len(self.X))):
            if verbose:
                print()
                print()
                print("Index:", index)
            queries_count = 0
            img_clean = self.X[index]
            img = self.X[index].copy()

            if verbose:
                print("INITIAL IMAGE")
            if self.reshape_flag:
                new_shape = self.reshape
            else:
                new_shape = np.shape(img)
            if verbose:
                plt.imshow(np.reshape(img, new_shape))
            plt.show()
            label = np.argmax(self.y[index])

            shape = np.shape(img)
            selected = set()

            init_probs_distribution = self.model.predict(np.array([img]))[0]
            queries_count += 1
            curr_probs_distribution = init_probs_distribution.copy()
            init_prob = init_probs_distribution[label]
            curr_prob = init_prob
            perturbed = False
            l2_distance_sq = 0

            for step in range(self.max_iterations):
                if(l2_distance_sq > self.max_l2_distance ** 2):
                    if verbose:
                        print("l2_distance exceeded")
                        print()
                    break
                if(step % 100 == 0):
                    if verbose:
                        print(" Step:", step)
                        print(" Prob:", curr_prob)
                        print()
                if SimbaWrapper.stop_condition_success(curr_probs_distribution, label):
                    if verbose:
                        print("Perturbation successful")
                        print("Classified as: ", np.argmax(curr_probs_distribution))
                        print("Perturbed image:")
                        if self.reshape_flag:
                            new_shape = self.reshape
                        else:
                            new_shape = np.shape(img)
                        plt.imshow(np.reshape(img, new_shape))
                        plt.show()
                    perturbed = True
                    break

                if queries_count >= self.max_queries:
                    if verbose:
                        print("max queries exceeded")
                        print()
                    break

                if len(selected) >= np.shape(self.X[0])[0] * np.shape(self.X[0])[1]:
                    if verbose:
                        print("all pixels consumed")
                        print()
                    break
                
                # Pick next pixel to alter. This can't be part of the pixels already expored, as per the SimBA paper.
                i = random.randint(0, shape[0] - 1)
                j = random.randint(0, shape[1] - 1)
                while (i,j) in selected:
                    i = random.randint(0, shape[0] - 1)
                    j = random.randint(0, shape[1] - 1)
                selected.add((i,j))

                img_pos = img.copy()
                img_pos[i][j][0] = img_pos[i][j][0] + self.epsilon * self.max_value
                img_pos[i][j][0] = utils.cap(img_pos[i][j][0], 0, self.max_value)

                res_pos_distribution = self.model.predict(np.array([img_pos]))[0]
                queries_count += 1
                res_pos = res_pos_distribution[label]

                if(res_pos < curr_prob):
                    curr_probs_distribution = res_pos_distribution
                    curr_prob = res_pos
                    l2_distance_sq += \
                        (img_pos[i][j][0] - img_clean[i][j][0])**2 - (img[i][j][0] - img_clean[i][j][0])**2
                    img = img_pos
                else:
                    img_neg = img.copy()
                    img_neg[i][j][0] = img_neg[i][j][0] - self.epsilon * self.max_value
                    img_neg[i][j][0] = utils.cap(img_neg[i][j][0], 0, self.max_value)

                    res_neg_distribution = self.model.predict(np.array([img_neg]))[0]
                    queries_count += 1
                    res_neg = res_neg_distribution[label]

                    if(res_neg < curr_prob):
                        curr_probs_distribution = res_neg_distribution
                        curr_prob = res_neg
                        l2_distance_sq += \
                            (img_neg[i][j][0] - img_clean[i][j][0])**2 - (img[i][j][0] - img_clean[i][j][0])**2
                        img = img_neg

            if verbose:
                print(" Queries:", queries_count)
                print("__________________")

            self.queries.append(queries_count)
            self.perturbed.append(perturbed)
            self.l2_distances.append(math.sqrt(l2_distance_sq))
            self.X_modified.append(img)
            
        save_data = {
            "queries":self.queries,
            "perturbed":self.perturbed,
            "l0_distances":self.l2_distances
        }
        with open(self.folder + "/simba_stats.json", "w") as fp:
            json.dump(save_data, fp)
