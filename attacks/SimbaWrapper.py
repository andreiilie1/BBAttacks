import numpy as np
import random

import sys
sys.path.append("../utils/")
import utils

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
    def __init__(self, model, X, y, epsilon, max_queries, max_iterations=1000, max_l0_distance=28):
        self.model = model
        self.X = X
        self.y = y
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.max_queries = max_queries
        self.max_l0_distance = max_l0_distance
        
        # At the end of a SimbaWrapper.run_simba run, self.queries[i] will contain the number of queries
        # to the model for image X[i] until the attack stopped (with or without success); 
        # self.perturbed[i] will contain a True/False flag, stating whether X[i] was succesfully perturbed
        # within the max_itertaions, max_queries, and max_l0_distance limits;
        # and self.l0_distances[i] will contain the l0 distance at which the attack stopped for image X[i]
        # (be it with or without success).
        self.queries = []
        self.perturbed = []
        self.l0_distances = []

    @staticmethod
    def stop_condition_success(prob_distribution, label):
        if np.argmax(prob_distribution) != label:
            return True
        return False

    def run_simba(self, verbose=True):
        for index in range(len(self.X)):
            if verbose:
                print("Index:", index)
            queries_count = 0
            img = self.X[index].copy()
            label = np.argmax(self.y[index])

            shape = np.shape(img)
            selected = set()

            init_probs_distribution = self.model.predict(np.array([img]))[0]
            queries_count += 1
            curr_probs_distribution = init_probs_distribution.copy()
            init_prob = init_probs_distribution[label]
            curr_prob = init_prob
            perturbed = False
            l0_distance = 0

            for step in range(self.max_iterations):
                if(l0_distance > self.max_l0_distance):
                    if verbose:
                        print("l0_distance exceeded")
                        print()
                    break
                if(step % 25 == 0):
                    if verbose:
                        print(" Step:", step)
                        print(" Prob:", curr_prob)
                        print()
                if SimbaWrapper.stop_condition_success(curr_probs_distribution, label):
                    if verbose:
                        print("image perturbed")
                        print()
                    perturbed = True
                    break

                if queries_count >= self.max_queries:
                    if verbose:
                        print("max queries exceeded")
                        print()
                    break

                if len(selected) == np.shape(self.X[0])[0] * np.shape(self.X[0])[1]:
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
                img_pos[i][j][0] = img_pos[i][j][0] + self.epsilon
                img_pos[i][j][0] = utils.cap(img_pos[i][j][0], 0, 255)

                res_pos_distribution = self.model.predict(np.array([img_pos]))[0]
                queries_count += 1
                res_pos = res_pos_distribution[label]

                if(res_pos < curr_prob):
                    curr_probs_distribution = res_pos_distribution
                    curr_prob = res_pos
                    img = img_pos
                    l0_distance += 1
                else:
                    img_neg = img.copy()
                    img_neg[i][j][0] = img_neg[i][j][0] - self.epsilon
                    img_neg[i][j][0] = utils.cap(img_neg[i][j][0], 0, 255)

                    res_neg_distribution = self.model.predict(np.array([img_neg]))[0]
                    queries_count += 1
                    res_neg = res_neg_distribution[label]

                    if(res_neg < curr_prob):
                        curr_probs_distribution = res_neg_distribution
                        curr_prob = res_neg
                        img = img_neg
                        l0_distance += 1

            print(" Queries:", queries_count)
            print("__________________")

            self.queries.append(queries_count)
            self.perturbed.append(perturbed)
            self.l0_distances.append(l0_distance)
