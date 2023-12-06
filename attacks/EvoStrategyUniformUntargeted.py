import numpy as np
import random
import matplotlib.pyplot as plt
from EvoStrategy import EvoStrategy


class EvoStrategyUniformUntargeted(EvoStrategy):
    """Black-box, untargeted adversarial attack against image classifiers.

    This is provided as an implementation of the evolutionary strategy EvoStrategy abstract base class.
    It encapsulates the target model and image and provides a method to run the adversarial attack. Fitness of
    individuals is implemented as their probability to not be classified correctly. The attack works by generating
    random individuals (offspring) near a parent, and proceeds by only selecting the fittest individual from the
    generation as the next parent of the next generation.

    Attributes:
        model: Target model to be attacked. This has to expose a predict method that returns the
            output probability distributions when provided a batch of images as input.
        img: An array (HxWxC) representing the target image to be perturbed.
        label: An integer representing the correct class index of the image.
        generation_size: An integer parameter of the attack representing how many perturbations are attempted
            per generation. The larger generation size leads to more exploration, more queries per generation,
            and success achieved in fewer generations. Usual values are in the range 10..100.
        one_step_perturbation_pixel_count: An integer parameter of the attack representing how many pixels to perturb
            in one evolution step. Smaller values lead to finding a successful perturbation slower, but at smaller
            perturbation norms. Larger values lead to finding a successful perturbation faster, but at larger
            perturbation norms. This can be seen as an equivalent of learning rates when training deep models: one
            trades off the accuracy in picking the right optimisation path with the speed of doing it.
        verbose: A boolean flag which, when set to True, enables printing info on the attack results.
        reshape_flag: A boolean flag which, when set to True, enables reshaping the target image img and the
            final perturbed image produced by the adversarial attack only for visualisation purposes. This does not
            change the way the attack works in any way, but only enables smoother visualisations when verbose is True.
            Does nothing when verbose is False.
        reshape_dims: A tuple of two or three integers representing the shape to which images will be reshaped for
            visualisation purposes. Only used when verbose and reshape_flag are both set to True. Can use a tuple of
            two integers (H, W) in the case of single-channel images. Otherwise, use tuples of 3 integers (H, W, C).
        pixel_space_int_flag: A boolean flag indicating whether the image pixel values (and hence the perturbed image
            pixel values) are integers. True means they are integers, False means they are floats.
        pixel_space_min: A number (integer or float) representing the minimum value pixels can take in the image space.
        pixel_space_max: A number (integer or float) representing the maximum value pixels can take in the image space.

    Methods:
        get_best_candidate(): Returns the fittest individual in the active generation.
        is_perturbed(): Returns a boolean representing whether a successful adversarial perturbation has been achieved
            in the active generation.
        run_adversarial_attack(self, steps=100): Runs the adversarial attack based on the evolutionary strategy until a
            successful adversarial perturbation was found or until steps generations were explored. Returns the total
            number of generations before the stopping condition was reached.
    """

    def __init__(
        self,
        model,
        img,
        label,
        generation_size,
        one_step_perturbation_pixel_count,
        verbose,
        reshape_flag=False,
        reshape_dims=(28, 28),
        pixel_space_int_flag=False,
        pixel_space_min=0.0,
        pixel_space_max=1.0,
    ):
        EvoStrategy.__init__(self)

        # Each instance encapsulates the model and image to perturb
        self.model = model
        self.img = img
        self.label = label

        # Set active generation to the unperturbed image
        self.active_generation = [img]
        self.queries += 1  # One query is used for calling predict on the unperturbed image
        self.fitness_scores = [
            1 - self.model.predict(np.expand_dims(img, axis=0), verbose=False)[0][label]
        ]

        self.generation_size = generation_size
        self.one_step_perturbation_pixel_count = one_step_perturbation_pixel_count

        self.verbose = verbose

        self.reshape_flag = reshape_flag
        self.reshape_dims = reshape_dims

        self.pixel_space_int_flag = pixel_space_int_flag
        self.pixel_space_min = pixel_space_min
        self.pixel_space_max = pixel_space_max

        if self.verbose:
            print("___________________")
            print(f"Instantiated new {type(self).__name__} attack")
            print("Correct label:", self.label)
            print(
                "Initial class:",
                np.argmax(
                    self.model.predict(np.expand_dims(img, axis=0), verbose=False)[0]
                ),
            )
            print(
                "Initial probability to be classified correctly:",
                self.model.predict(np.expand_dims(img, axis=0), verbose=False)[0][
                    label
                ],
            )

    def _get_next_generation(self):
        best_candidate = self.get_best_candidate()
        new_generation = []
        for i in range(self.generation_size):
            offspring = self._get_offspring(best_candidate)
            new_generation.append(offspring)
        return new_generation

    def _get_fitness_scores(self):
        # We define fitness as probability to be anything else than the correct class (self.label),
        # which is 1 - correct_class_probability. We do batch predictions for entire generations.
        fitness_scores = 1 - self.model.predict(
            np.array(self.active_generation), verbose=False
        )
        fitness_scores = np.array(list(map(lambda x: x[self.label], fitness_scores)))
        queries = len(fitness_scores)
        return fitness_scores, queries

    def _get_offspring(self, candidate):
        # Offspring are within one pixel distance from their parent, with gaussian noise being added.
        shape = np.shape(candidate)
        candidate_copy = candidate.copy()
        for perturb_count in range(self.one_step_perturbation_pixel_count):
            i = random.randint(0, shape[0] - 1)
            j = random.randint(0, shape[1] - 1)
            for c in range(np.shape(self.img)[2]):
                sampling_method = random.randint if self.pixel_space_int_flag else random.uniform
                value = sampling_method(self.pixel_space_min, self.pixel_space_max)
                candidate_copy[i][j][c] = value
        return candidate_copy

    def _generate_next_generation(self):
        EvoStrategy._generate_next_generation(self)

    def get_best_candidate(self):
        return super().get_best_candidate()

    def is_perturbed(self):
        best_candidate = self.get_best_candidate()
        if (
            np.argmax(
                self.model.predict(
                    np.expand_dims(best_candidate, axis=0), verbose=False
                )[0]
            )
            != self.label
        ):
            return True
        return False

    def run_adversarial_attack(self, steps=100):
        generation_idx = 0

        while generation_idx < steps and not self.is_perturbed():
            self._generate_next_generation()
            generation_idx += 1

        if self.is_perturbed() and generation_idx > 0:
            if self.verbose:
                print("After", generation_idx, "generations")
                print(
                    "Label:",
                    self.label,
                    "; Prediction:",
                    np.argmax(
                        self.model.predict(
                            np.expand_dims(self.get_best_candidate(), axis=0),
                            verbose=False,
                        )
                    ),
                )
                print("Fitness:", max(self.fitness_scores))
                try:
                    plt.subplot(121)
                    if self.reshape_flag:
                        plt.imshow(np.reshape(self.img, self.reshape_dims))
                    else:
                        plt.imshow(self.img)

                    plt.subplot(122)
                    if self.reshape_flag:
                        plt.imshow(np.reshape(self.get_best_candidate(), self.reshape_dims))
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
            print(
                "Final probability to be classified correctly:",
                self.model.predict(
                    np.expand_dims(self.get_best_candidate(), axis=0), verbose=False
                )[0][self.label],
            )
            print(
                "Final probability to be classified as:",
                np.argmax(
                    self.model.predict(
                        np.expand_dims(self.get_best_candidate(), axis=0), verbose=False
                    )[0]
                ),
                " is ",
                np.max(
                    self.model.predict(
                        np.expand_dims(self.get_best_candidate(), axis=0)
                    )[0]
                ),
            )
            print("Queries: ", self.queries)
            print("_________________________")
            print()
        return generation_idx
