import argparse

import os
import sys
sys.path.append("/home/ailie/Repos/BBAttacks/attacks/")
sys.path.append("/home/ailie/Repos/BBAttacks/utils/")

import numpy as np
import utils

import importlib
from tqdm import tqdm
from datetime import datetime

import EvoStrategy
import SimbaWrapperL2 as swl2


date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

# Python args logic
parser = argparse.ArgumentParser(
    description='Benchmark the robustness of a model using the EvoBA attack'
)

parser.add_argument(
    '--mp', '--model_path',
    type=str,
    help='Path to a python package containing a model class with a .predict function',
    required=True
)

parser.add_argument(
    '--mcn', '--model_class_name',
    type=str,
    help='Name of the python class containing a .predict function that calls the target model inference',
    required=True
)

parser.add_argument(
    '--out', '--out_path',
    type=str,
    help='Output path of the attack artifacts',
    default='results'
)

parser.add_argument(
    '--run', '--run_name',
    type=str,
    help='Name of the EvoBA run, used for grouping artifacts in out_path under the same folder',
    default=date
)

parser.add_argument(
    '--task', '--task_name',
    type=str,
    help="Name of the model's classification task. For the moment, only cifar100. \
          Will be cifar10/mnist/imagenet/custom, where custom allows for user's own input dataset and labels",
    default="cifar100"
)

parser.add_argument(
    '--egs', '--evoba_generation_size',
    type=int,
    help="Generation size to be used by EvoBA",
    default=30
)

parser.add_argument(
    '--epc', '--evoba_pixel_count',
    type=int,
    help="Count of pixels perturbed in one offspring generation step by EvoBA",
    default=1
)

parser.add_argument(
    '--es', '--evoba_steps',
    type=int,
    help="Max count of steps (number of generations) of performing EvoBA.",
    default=80
)

parser.add_argument(
    '--seps', '--simba_epsilon',
    type=int,
    help="Epsilon to be used in SimBA (perturbation of each individual step as a ratio of the max pixel value).",
    default=0.2
)

parser.add_argument(
    '--ss', '--sample_size',
    type=int,
    help="Size of the image sample that EvoBA will try to perturb (-1 for all available data)",
    default=80
)

parser.add_argument('--l0', default=True, action='store_true')
parser.add_argument('--no-l0', dest='l0', action='store_false')

parser.add_argument('--l2', default=True, action='store_true')
parser.add_argument('--no-l2', dest='l2', action='store_false')

args = parser.parse_args()


# Importing the module containing the targe model (and the model)
model_path = args.mp
module_spec = importlib.util.spec_from_file_location("cifar100vgg", model_path)

module_model = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module_model)

model_name = args.mcn
model_class = getattr(module_model, model_name)
print("Imported model module")

model_instance = model_class()
print("Instantiated model\n")


# Create the output folder structure
OUT_PATH = args.out
RUN_NAME = args.run
run_output_folder = os.path.join(OUT_PATH, RUN_NAME)
os.mkdir(run_output_folder)

print(f"Results will be stored in {run_output_folder}\n")


#TODO: get task from argument
task = args.task

print(f"Task: {task}\n")

# Task specific data loading and preprocessing
if task == "cifar100":
    from tensorflow.keras.datasets import cifar100
    from tensorflow import keras
    NUM_CLASSES = 100

    _, (x_test, y_test) = cifar100.load_data()

    x_test = x_test.astype('int')
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    print("Loaded CIFAR100 test data")
else:
    print(f"Task {task} not supported yet")
    sys.exit()


# Evaluate model accuracy on sample set
preds_test = model_instance.predict(x_test)
preds_test_labels = np.argmax(preds_test, axis=1)
true_test_labels = np.argmax(y_test, axis=1)

acc_test = np.mean(true_test_labels == preds_test_labels)
print(f"Test accuracy: {acc_test}\n")


# We will only try to perturb the correctly classified images
x_test_correct = x_test[preds_test_labels == true_test_labels]
y_test_correct = y_test[preds_test_labels == true_test_labels]
print(f"Will save for perturbation only the correctly "
      f"classified images: {len(x_test_correct)} samples out of {len(x_test)} images")


# Select the sample to attack below
sample_size = args.ss
LEFT_IDX_ATTACK_IMAGES = 0
# If the specified sample size is <0, then use the entire test set for running the attack
if sample_size > 0:
    RIGHT_IDX_ATTACK_IMAGES = sample_size
else:
    RIGHT_IDX_ATTACK_IMAGES = len(x_test_correct)
SAMPLE_IMAGES = x_test_correct[LEFT_IDX_ATTACK_IMAGES: RIGHT_IDX_ATTACK_IMAGES]
SAMPLE_Y = y_test_correct[LEFT_IDX_ATTACK_IMAGES: RIGHT_IDX_ATTACK_IMAGES]
print("Selected data sample to attack\n")

RUN_L0_ATTACK = args.l0
RUN_L2_ATTACK = args.l2

if RUN_L0_ATTACK:
    print("STARTING THE L0 ATTACK (EvoBA)")

    GENERATION_SIZE = args.egs
    PIXEL_COUNT = args.epc
    STEPS = args.es

    print(f"Will perform EvoBA with")
    print(f"- GENERATION_SIZE={GENERATION_SIZE}")
    print(f"- PIXEL_COUNT={PIXEL_COUNT}")
    print(f"- STEPS={STEPS}\n")

    adv_evo_strategy = {}
    VERBOSE = False

    for index in tqdm(range(len(SAMPLE_IMAGES))):
        img = SAMPLE_IMAGES[index]
        label = SAMPLE_Y[index]

        true_label = np.argmax(label)

        adv_evo_strategy[index] = EvoStrategy.AdversarialPerturbationEvoStraegy(
            model=model_instance,
            img=img,
            label=true_label,
            generation_size=GENERATION_SIZE,
            one_step_perturbation_pixel_count=PIXEL_COUNT,
            verbose=VERBOSE,
            zero_one_scale=False
        )

        no_steps = adv_evo_strategy[index].run_adversarial_attack(steps=STEPS)

    evoba_stats = utils.get_evoba_stats(adv_evo_strategy)
    utils.print_evoba_stats(evoba_stats)

    utils.save_evoba_artifacts(evoba_stats, run_output_folder)
    print(f"EvoBA Artifacts saved at path {run_output_folder}")
    print()


if RUN_L2_ATTACK:
    print("STARTING THE L2 ATTACK (SIMBA)")

    EPSILON = args.seps

    print(f"Will perform SimBA with")
    print(f"- EPSILON={EPSILON}")

    simba_wrapper = swl2.SimbaWrapper(
        model=model_instance,
        X=SAMPLE_IMAGES,
        y=SAMPLE_Y,
        epsilon=EPSILON,
        max_queries=4000,
        max_l2_distance=1000,
        max_iterations = 2000,
        folder=run_output_folder,
        verbose=False,
        max_value=255.0
    )

    simba_wrapper.run_simba()
    print(simba_wrapper.perturbed)
    print(simba_wrapper.queries)

    import math

    l2_dists_simba = []
    for index_diff in tqdm(range(len(SAMPLE_IMAGES))):
        diff = np.abs(simba_wrapper.X_modified[index_diff] - simba_wrapper.X[index_diff])
        #     diff = np.reshape(diff, (32, 32, 3))
        l2_dist = math.sqrt(np.sum(np.reshape(diff, (-1)) ** 2))
        l2_dists_simba.append(l2_dist)
    #     print("L2 distance:", math.sqrt(np.sum(np.reshape(diff, (-1))**2)))
    #     plt.imshow(np.reshape(adv_evo_strategy[index_diff].get_best_candidate(), (28, 28)))
    #     plt.show()
    #     print("Prediction:", model.predict(np.array([adv_evo_strategy[index_diff].get_best_candidate()])))

    print(l2_dists_simba)