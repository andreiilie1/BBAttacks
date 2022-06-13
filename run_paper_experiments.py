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
import EpsilonGreedyAttack
import EpsilonGreedyAttackThreshold
import SimbaWrapperL2 as swl2

from utils import AttackType

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

# parser.add_argument(
#     '--egs', '--evoba_generation_size',
#     type=int,
#     help="Generation size to be used by EvoBA",
#     default=30
# )

# parser.add_argument(
#     '--epc', '--evoba_pixel_count',
#     type=int,
#     help="Count of pixels perturbed in one offspring generation step by EvoBA",
#     default=1
# )

# parser.add_argument(
#     '--es', '--evoba_steps',
#     type=int,
#     help="Max count of steps (number of generations) of performing EvoBA.",
#     default=80
# )

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

# parser.add_argument(
#     '--egeps', '--eps_greedy_eps',
#     type=float,
#     help="Value to be used as exploration epsilon in the epsilon greedy attack.",
#     default=0.1
# )

parser.add_argument('--run-evoba', default=True, action='store_true')
parser.add_argument('--no-run-evoba', dest='run_evoba', action='store_false')

parser.add_argument('--run-eps-greedy', default=True, action='store_true')
parser.add_argument('--no-run-eps-greedy', dest='run_eps_greedy', action='store_false')

parser.add_argument('--run-eps-greedy-thresh', default=True, action='store_true')
parser.add_argument('--no-run-eps-greedy-thresh', dest='run_eps_greedy_thresh', action='store_false')

parser.add_argument('--run-simba', default=False, action='store_true')
parser.add_argument('--no-run-simba', dest='run_simba', action='store_false')

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
    
    EXPERIMENT_NAME_MLFLOW = "CIFAR100"
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

RUN_EVOBA = args.run_evoba
RUN_SIMBA = args.run_simba
RUN_EPS_GREEDY = args.run_eps_greedy
RUN_EPS_GREEDY_THRESH = args.run_eps_greedy_thresh


def run_and_log_evoba(generation_size, pixel_count, steps, model_instance, images, labels):
    print("STARTING EvoBA")

    print(f"Will perform EvoBA with")
    print(f"- GENERATION_SIZE={generation_size}")
    print(f"- PIXEL_COUNT={pixel_count}")
    print(f"- STEPS={steps}\n")

    adv_evo_strategy = {}
    VERBOSE = False

    for index in tqdm(range(len(images))):
        img = images[index]
        label = labels[index]

        true_label = np.argmax(label)

        adv_evo_strategy[index] = EvoStrategy.AdversarialPerturbationEvoStraegy(
            model=model_instance,
            img=img,
            label=true_label,
            generation_size=generation_size,
            one_step_perturbation_pixel_count=pixel_count,
            verbose=VERBOSE,
            zero_one_scale=False
        )

        no_steps = adv_evo_strategy[index].run_adversarial_attack(steps=steps)

    evoba_stats = utils.get_evoba_stats(adv_evo_strategy)
    utils.print_evoba_stats(evoba_stats)
    utils.save_evoba_artifacts(evoba_stats, run_output_folder)
    
    additional_params = {
        "gen_size": generation_size,
        "px_count": pixel_count,
        "steps": steps
    }
    
    utils.generate_mlflow_logs(
        strategy_objects=adv_evo_strategy, 
        attack_type=AttackType.EVOBA, 
        unperturbed_images=images, 
        run_name="EVOBA", 
        experiment_name=EXPERIMENT_NAME_MLFLOW,
        additional_params=additional_params
    )


if RUN_EVOBA:
    
#     GENERATION_SIZE = args.egs
#     PIXEL_COUNT = args.epc
#     STEPS = args.es
    
    evoba_configs = [
        {"generation_size": gs, "pixel_count": pc, "steps": 80} for gs in [20, 30, 40] for pc in [1, 2, 4]
    ]
    
    for config in evoba_configs:
        run_and_log_evoba(
            generation_size=config["generation_size"],
            pixel_count=config["pixel_count"],
            steps=config["steps"],
            model_instance=model_instance,
            images=SAMPLE_IMAGES,
            labels=SAMPLE_Y
        )

    
def run_and_log_eps_greedy(eps, patch_size, model_instance, images, labels):
    
    print(f"RUNNING EPS_GREEDY WITH EPSILON {eps} AND PATCH_SIZE {patch_size}")
    
    EPS = eps
    
    pixel_groups = utils.get_grid_pixel_groups(patch_size=patch_size, image_size=np.shape(images[0])[0])
    adv_evo_strategy_epsilon = {}
        
    for index in tqdm(range(len(images))):
        img = images[index]
        label = labels[index]
        
        true_label = np.argmax(label)

        adv_evo_strategy_epsilon[index] = EpsilonGreedyAttack.EpsilonGreedyAttack(
            model=model_instance,
            img=img,
            one_hot_label=label,
            pixel_groups=pixel_groups,
            epsilon=EPS
            # generation_size=GENERATION_SIZE, 
            # one_step_perturbation_pixel_count=PIXEL_COUNT,
            # verbose=VERBOSE,
            # zero_one_scale=False
        )
    
        no_steps = adv_evo_strategy_epsilon[index].run_attack()
    
        
    eps_greedy_stats = utils.get_epsgreedy_stats(adv_evo_strategy_epsilon, images)
    utils.print_evoba_stats(eps_greedy_stats)
    utils.save_evoba_artifacts(eps_greedy_stats, run_output_folder)
    
    params_eps_greedy = {"epsilon": EPS}
    
    utils.generate_mlflow_logs(
        strategy_objects=adv_evo_strategy_epsilon, 
        attack_type=AttackType.EPSGREEDY, 
        unperturbed_images=images, 
        run_name="EPSGREEDY", 
        experiment_name=EXPERIMENT_NAME_MLFLOW,
        additional_params=params_eps_greedy
    )


if RUN_EPS_GREEDY:
    
    eps_greedy_configs = [
        {"epsilon": eps, "patch_size": ps} for eps in [0, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8] for ps in [4]
    ]
    
    for config in eps_greedy_configs:
        run_and_log_eps_greedy(
            eps=config["epsilon"], 
            patch_size=config["patch_size"], 
            model_instance=model_instance, 
            images=SAMPLE_IMAGES, 
            labels=SAMPLE_Y
        )
    

def run_and_log_eps_greedy_thresh(eps, threshold, increase_factor, decay_factor, max_rounds_until_decay, 
                                  patch_size, model_instance, images, labels):
    
    print("Running EpsGreedyThreshold")
    print(f"Params: \n eps {eps} \n increase_factor {increase_factor} \n decay_factor \
        {decay_factor} \n max_rounds_until_decay {max_rounds_until_decay}")
    
    pixel_groups = utils.get_grid_pixel_groups(patch_size=patch_size, image_size=np.shape(images[0])[0])
    
    adv_evo_strategy_eps_threshold = {}
    
    for index in tqdm(range(len(images))):
        img = images[index]
        label = labels[index]
        
        true_label = np.argmax(label)

        adv_evo_strategy_eps_threshold[index] = EpsilonGreedyAttackThreshold.EpsilonGreedyAttackThreshold(
            model=model_instance,
            img=img,
            one_hot_label=label,
            pixel_groups=pixel_groups,
            threshold=threshold,
            increase_factor=increase_factor,
            decay_factor=decay_factor,
            epsilon=eps,
            max_rounds_until_decay=max_rounds_until_decay
            # generation_size=GENERATION_SIZE, 
            # one_step_perturbation_pixel_count=PIXEL_COUNT,
            # verbose=VERBOSE,
            # zero_one_scale=False
        )

        no_steps = adv_evo_strategy_eps_threshold[index].run_attack()

    params_eps_threshold = {
        "epsilon": eps,
        "increase_fact": increase_factor,
        "decay_fact": decay_factor,
        "rounds_decay": max_rounds_until_decay,
        "patch_size": patch_size
    }
    
    eps_threshold_stats = utils.get_epsgreedy_stats(adv_evo_strategy_eps_threshold, images)
    utils.print_evoba_stats(eps_threshold_stats)
    utils.generate_mlflow_logs(
        strategy_objects=adv_evo_strategy_eps_threshold, 
        attack_type=AttackType.EPSGREEDY, 
        unperturbed_images=images, 
        run_name="EPSGREEDY_THRESHOLD", 
        experiment_name=EXPERIMENT_NAME_MLFLOW,
        additional_params=params_eps_threshold
    )

    
if RUN_EPS_GREEDY_THRESH:
    
    configs_eps_greedy_thresh = [
        {"epsilon": eps, "threshold": thr, "increase_factor": ifr, "decay_factor": dfr, "max_rounds_until_decay": mrud, "patch_size": ps} 
        for eps in [0.1, 0.2, 0.4] for thr in [0.05] for ifr in [1.01, 1.05] for dfr in [0.5, 0.9] for mrud in [20, 40] for ps in [4]
    ]
    
    for config in configs_eps_greedy_thresh:
        run_and_log_eps_greedy_thresh(
            eps=config["epsilon"],
            threshold=config["threshold"],
            increase_factor=config["increase_factor"],
            decay_factor=config["decay_factor"],
            max_rounds_until_decay=config["max_rounds_until_decay"],
            patch_size=config["patch_size"],
            model_instance=model_instance,
            images=SAMPLE_IMAGES,
            labels=SAMPLE_Y
        )
    
    
if RUN_SIMBA:
    print("STARTING THE L2 ATTACK (SIMBA)")

    EPSILON = args.seps

    print(f"Will perform SimBA with")
    print(f"- EPSILON={EPSILON}\n")

    simba_wrapper = swl2.SimbaWrapper(
        model=model_instance,
        X=SAMPLE_IMAGES,
        y=SAMPLE_Y,
        epsilon=EPSILON,
        max_queries=4000,
        max_l2_distance=1000,
        max_iterations=1000,
        folder=run_output_folder,
        verbose=False,
        max_value=255.0
    )

    simba_wrapper.run_simba()

    simba_stats = utils.get_simba_stats(simba_wrapper)
    utils.print_simba_stats(simba_stats)
    utils.save_simba_artifacts(simba_stats, run_output_folder)


print(f"Artifacts saved at path {run_output_folder}")
