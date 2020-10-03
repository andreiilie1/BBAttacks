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
    
    def generate_next_generation(self):
        new_generation = self.get_next_generation()
        self.active_generation = new_generation
        self.generation_count += 1
        self.fitness_scores = self.get_fitness_scores()
        self.queries += len(new_generation)

    
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
    def __init__(self, model, img, label, generation_size):
        EvoStrategy.__init__(self)
        self.model = model
        self.active_generation = [img]
        self.fitness_scores = [-self.model.predict(np.expand_dims(img, axis=0))[0][label]]
        self.generation_size = generation_size
        self.label = label
        self.queries += 1
    
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
    
    @staticmethod
    def get_offspring(candidate):
        # Offspring are within one pixel distance from their parent, with gaussian noise being added.
        shape = np.shape(candidate)
        i = random.randint(0, shape[0] - 1)
        j = random.randint(0, shape[1] - 1)
        value = random.gauss(0,128)
        candidate_copy = candidate.copy()
        candidate_copy[i][j][0] = cap(value, 0, 255)
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
        if i < steps and i > 0:
            print("After", i, "generations")
            print("Label:", self.label, "; Prediction:", np.argmax(self.model.predict(np.expand_dims(self.get_best_candidate(), axis=0))))
            print("Fitness:", max(self.fitness_scores))
            plt.imshow(np.reshape(self.get_best_candidate(), (28,28)))
            plt.show()
            print()
        return i