from abc import ABC, abstractmethod
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
    
    
class EpsilonGreedyAttackTargeted:
    def __init__(self, model, img, one_hot_label, pixel_groups, target_label, epsilon=0.1, preprocess=lambda x: x):
        self.model = model
        self.preprocess = preprocess

        self.img = np.copy(img)
        copy_self_img = np.copy(self.img)
        self.model_prediction = self.model.predict(np.array([self.preprocess(copy_self_img)]))[0]

        self.one_hot_label = one_hot_label
        self.label = np.argmax(self.one_hot_label)
        self.target_label = target_label
        
        self.pixel_groups = pixel_groups
        self.number_groups = len(self.pixel_groups)
        
        assert 0 <= epsilon <= 1
        self.epsilon = epsilon

        self.values = [0.0 for x in range(self.number_groups)]
        self.counts = [0 for x in range(self.number_groups)]
        self.count_explorations = 0

    def select_group(self):
        if random.random() > self.epsilon:
            max_group_index = np.random.choice(np.flatnonzero(np.array(self.values) == np.array(self.values).max()))
            return max_group_index
        else:
            random_group_index = random.randrange(self.number_groups)
            return  random_group_index


    def update(self, chosen_group, reward):
        self.counts[chosen_group] = self.counts[chosen_group] + 1

        n = self.counts[chosen_group]
        value = self.values[chosen_group]

        new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_group] = new_value
        return new_value


    def explore_attack_group(self, group_index):
        self.count_explorations += 1
        attack_group = self.pixel_groups[group_index]
        count_pixels_group = len(attack_group)
        attack_pixel_index = random.randrange(count_pixels_group)
        attack_pixel = attack_group[attack_pixel_index]

        copy_img = self.img.copy()
    
        for ch in range(3):
            value = random.randint(0,255)
            copy_img[attack_pixel[0]][attack_pixel[1]][ch] = value

        copy_self_img = np.copy(self.img)
        copy_img_clean = np.copy(copy_img)

        target_class_prob_before = self.model_prediction[self.target_label]

        pred_after = self.model.predict(np.array([self.preprocess(copy_img)]))[0]
        target_class_prob_after = pred_after[self.target_label]
        potential_reward = target_class_prob_after - target_class_prob_before

        return {
            "potential_reward": potential_reward,
            "altered_image": copy_img_clean,
            "prob_before": target_class_prob_before,
            "prob_after": target_class_prob_after,
            "pred_after": pred_after
        }


    def run_attack(self):
        NUM_STEPS = 5000

        for i in tqdm(range(NUM_STEPS)):
            attack_group = self.select_group()
            attack_result = self.explore_attack_group(attack_group)
            potential_reward = attack_result["potential_reward"]
            altered_image = attack_result["altered_image"]
            if potential_reward > 0.0001:
                self.img = altered_image
                self.model_prediction = attack_result["pred_after"]
            # potential_reward = max(potential_reward, 0)
            self.update(attack_group, potential_reward)
            # print("PROB BEFORE:", attack_result["prob_before"])
            # print("PROB AFTER:", attack_result["prob_after"])
            if self.is_perturbed():
                print("Image succesfully perturbed")
                print("Correct label:", self.label)
                print("Predicted label:", np.argmax(self.model_prediction))
                break
            # print()


    def is_perturbed(self):
        copy_self_img = np.copy(self.img)
        pred_label = np.argmax(self.model_prediction)
        return pred_label == self.target_label