from abc import ABC, abstractmethod
from utils import cap
import numpy as np
import random
import matplotlib.pyplot as plt

# Base class for simple evolutionary strategy attacks
class EvoStrategy(ABC):
    def __init__(self):
        self.generation_count = 0
        self.active_generation = []
        self.fitness_scores = []
        self.queries = 0
        pass
    
    def get_active_generation(self):
        return self.active_generation
    
    def get_best_candidate(self):
        fitnesses = self.fitness_scores
        best_candidate_index = np.argmax(fitnesses)
        return self.active_generation[best_candidate_index]
    
    @abstractmethod
    def get_next_generation(self):
        pass
    
    @abstractmethod
    def get_fitness_scores(self):
        pass
    
    def generate_next_generation(self, queries = -1):
        new_generation = self.get_next_generation()
        self.active_generation = new_generation
        self.generation_count += 1
        self.fitness_scores = self.get_fitness_scores()
        if queries == -1:
            self.queries += len(new_generation)
        else:
            self.queries += queries

    
# Implementation of the EvoStrategy base class, which will generate random 
# individuals near a parent, and proceeds by only selecting the best (lowest correct probability) 
# individual from the generation as the parent of the next one.
# Main parameters to change are self.generation_size (how many individuals per each generation),
# and, when calling the actual attack, the maximum number of rounds, passed as parameter 'steps' to
# the method AdversarialPerturbationEvoStraegy.run_adversarial_attack(steps)

# TODO: using (28x28) images below, customise it for any image sizes to be able to test on ImageNet
class AdversarialPerturbationEvoStraegy(EvoStrategy):
    # We do not abstract model as an objective_function to be able
    # to use batch prediction easier.
    def __init__(self, model, img, label, generation_size, one_step_perturbation_pixel_count, 
                 verbose, reshape_flag=False, reshape=(28,28), zero_one_scale=True, range_scale_int=False,
                 min_rand=0, max_rand=255):
        EvoStrategy.__init__(self)
        self.model = model
        self.img = img
        self.active_generation = [img]
        self.fitness_scores = [1-self.model.predict(np.expand_dims(img, axis=0))[0][label]]
        self.generation_size = generation_size
        self.label = label
        self.one_step_perturbation_pixel_count = one_step_perturbation_pixel_count
        self.queries += 1
        self.verbose = verbose
        self.reshape_flag = reshape_flag
        self.reshape = reshape
        self.zero_one_scale = zero_one_scale
        self.range_scale_int = range_scale_int
        self.min_rand = min_rand
        self.max_rand = max_rand
        if self.verbose:
            print()
            print("___________________")
            print("Correct label:", self.label)
            print("Initial class:", 
                  np.argmax(self.model.predict(np.expand_dims(img, axis=0))[0]))
            print("Initial probability to be classified correctly:", 
                  self.model.predict(np.expand_dims(img, axis=0))[0][label])
    
    def get_next_generation(self):
        best_candidate = self.get_best_candidate()
        new_generation = []
        for i in range(self.generation_size):
            offspring = self.get_offspring(best_candidate)
            new_generation.append(offspring)
        return new_generation
    
    def get_fitness_scores(self):
        # We definte fitness as probability to be anything else than the correct classs (self.label),
        # which is 1 - correct_class_probability. We do batch predictions for entire generations.
        fitnesses = 1 - self.model.predict(np.array(self.active_generation))
        fitnesses = np.array(list(map (lambda x: x[self.label], fitnesses)))
        return fitnesses
    
    def get_offspring(self, candidate):
        # Offspring are within one pixel distance from their parent, with gaussian noise being added.
        shape = np.shape(candidate)
#         NOTE: need so many queries because we previously ony modified pixel by pixel! modify batches of pixels per new generation
        candidate_copy = candidate.copy()
        for perturb_count in range(self.one_step_perturbation_pixel_count):
            i = random.randint(0, shape[0] - 1)
            j = random.randint(0, shape[1] - 1)
            for c in range(np.shape(self.img)[2]):
                if self.zero_one_scale:
                    value = random.randint(0,255)/255
                elif self.range_scale_int:
                    value = random.randint(self.min_rand, self.max_rand)
                else:
                    value = random.randint(0,255)
                candidate_copy[i][j][c] = value
        return candidate_copy
    
    def generate_next_generation(self):
        EvoStrategy.generate_next_generation(self)
    
    def stop_criterion(self):
        best_candidate = self.get_best_candidate()
        if np.argmax(self.model.predict(np.expand_dims(best_candidate, axis=0))[0]) != self.label:
            return True
        return False
    
    def run_adversarial_attack(self, steps=100):
        i = 0
        while i < steps and not self.stop_criterion():
            self.generate_next_generation()
            i += 1
        if self.stop_criterion() and i > 0:
            if self.verbose:
                print("After", i, "generations")
                print("Label:", self.label, "; Prediction:", np.argmax(self.model.predict(np.expand_dims(self.get_best_candidate(), axis=0))))
                print("Fitness:", max(self.fitness_scores))
                try:              
                    plt.subplot(121)
                    if self.reshape_flag:
                        plt.imshow(np.reshape(self.img, self.shape))
                    else:
                        plt.imshow(self.img)
                        
                    plt.subplot(122)
                    if self.reshape_flag:
                        plt.imshow(np.reshape(self.get_best_candidate(), self.shape))
                    else:
                        plt.imshow(self.get_best_candidate())
                        
                    plt.show()
                except Exception as e:
                    if self.verbose:
                        print("error displaying")
                        print(e)
                if self.verbose:
                    print()
        if self.verbose:
            print("Final probability to be classified correctly:", 
                  self.model.predict(np.expand_dims(self.get_best_candidate(), axis=0))[0][self.label])
            print("Final probability to be classified as:",
                  np.argmax(self.model.predict(np.expand_dims(self.get_best_candidate(), axis=0))[0]),
                  " is ",
                  np.max(self.model.predict(np.expand_dims(self.get_best_candidate(), axis=0))[0]))
            print("Queries: ", self.queries)
            print("_________________________")
            print()
        return i
    
    
# class AdversarialPerturbationBFStraegy(EvoStrategy):
#     # We do not abstract model as an objective_function to be able
#     # to use batch prediction easier.
#     def __init__(self, model, img, label, generation_size, one_step_perturbation_pixel_count, verbose):
#         if verbose:
#             print()
#             print("___________________")
#         EvoStrategy.__init__(self)
#         self.model = model
#         self.img = img
#         self.active_generation = [img]
#         self.fitness_scores = [1-self.model.predict(np.expand_dims(img, axis=0))[0][label]]
#         self.generation_size = generation_size
#         self.label = label
#         self.one_step_perturbation_pixel_count = one_step_perturbation_pixel_count
#         self.queries += 1
#         self.verbose = verbose
#         if verbose:
#             print("Correct label:", self.label)
#             print("Initial class:", 
#                   np.argmax(self.model.predict(np.expand_dims(img, axis=0))[0]))
#             print("Initial probability to be classified correctly:", 
#                   self.model.predict(np.expand_dims(img, axis=0))[0][label])
    
#     def get_next_generation(self):
#         best_candidate = self.get_best_candidate()
#         new_generation = []
#         for i in range(len(self.active_generation)):
#             new_generation.append(self.active_generation[i])
#         for i in range(self.generation_size):
#             offspring = self.get_offspring(best_candidate)
#             new_generation.append(offspring)
#         return new_generation
    
#     def get_fitness_scores(self):
#         # We definte fitness as probability to be anything else than the correct classs (self.label),
#         # which is 1 - correct_class_probability. We do batch predictions for entire generations.
#         new_generation_individuals = np.array(self.active_generation)[-self.generation_size:]
#         fitnesses = 1 - self.model.predict(new_generation_individuals)
#         fitnesses = (self.fitness_scores + list(map (lambda x: x[self.label], fitnesses)))
#         return fitnesses
    
#     def get_offspring(self, candidate):
#         # Offspring are within one pixel distance from their parent, with gaussian noise being added.
#         shape = np.shape(candidate)
# #         NOTE: need so many queries because we previously ony modified pixel by pixel! modify batches of pixels per new generation
#         candidate_copy = candidate.copy()
#         for perturb_count in range(self.one_step_perturbation_pixel_count):
#             i = random.randint(0, shape[0] - 1)
#             j = random.randint(0, shape[1] - 1)
#             for c in range(np.shape(self.img)[2]):
#                 value = random.randint(0,255)/255
#                 candidate_copy[i][j][c] = value
#         return candidate_copy
    
#     def generate_next_generation(self):
#         EvoStrategy.generate_next_generation(self, self.generation_size)
    
#     def stop_criterion(self):
#         best_candidate = self.get_best_candidate()
#         if np.argmax(self.model.predict(np.expand_dims(best_candidate, axis=0))[0]) != self.label:
#             return True
#         return False
    
#     def run_adversarial_attack(self, steps=100):
#         i = 0
#         while i < steps and not self.stop_criterion():
#             self.generate_next_generation()
#             i += 1
#         if self.stop_criterion() and i > 0:
#             if self.verbose:
#                 print("After", i, "generations")
#                 print("Label:", self.label, "; Prediction:", np.argmax(self.model.predict(np.expand_dims(self.get_best_candidate(), axis=0))))
#                 print("Fitness:", max(self.fitness_scores))
#             try:
#                 if self.verbose:
#                     plt.title("Before")
#                     plt.imshow(self.img)
#                     plt.show()
#                     plt.title("After")
#                     plt.imshow(self.get_best_candidate())
#                     plt.show()
#             except:
#                 1+1
#             print()
#         if self.verbose:
#             print("Final probability to be classified correctly:", 
#                   self.model.predict(np.expand_dims(self.get_best_candidate(), axis=0))[0][self.label])
#             print("Final probability to be classified as:",
#                   np.argmax(self.model.predict(np.expand_dims(self.get_best_candidate(), axis=0))[0]),
#                   " is ",
#                   np.max(self.model.predict(np.expand_dims(self.get_best_candidate(), axis=0))[0]))
#             print("Queries", self.queries)
#             print("_________________________")
#             print()
#         return i