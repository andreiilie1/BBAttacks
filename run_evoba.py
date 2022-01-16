import os
import sys
sys.path.append("/home/ailie/Repos/BBAttacks/attacks/")
sys.path.append("/home/ailie/Repos/BBAttacks/utils/")

import numpy as np
import utils

import importlib
import EvoStrategy
from tqdm import tqdm


# TODO: get model_path from argument
model_path = "/home/ailie/Repos/BBAttacks/models/cifar100vgg/cifar100vgg.py"
module_spec = importlib.util.spec_from_file_location("cifar100vgg", model_path)

module_model = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(module_model)

# TODO: get model_name from argument
model_name = "cifar100vgg"
model_class = getattr(module_model, "cifar100vgg")
print("Imported model module")

weights_path = "/home/ailie/Repos/BBAttacks/models/cifar100vgg/cifar100vgg.h5"

#TODO: get params from argument
params = {
    "model_class_params":
        {
            "train": False,
            "weights_path": weights_path
        }
}

model_instance = model_class(**params["model_class_params"])
print("Instantiated model\n")

#TODO: get attack params from arguments
GENERATION_SIZE = 30
PIXEL_COUNT = 1
STEPS = 80

#TODO: get outpath from arguments
OUT_PATH = "/home/ailie/Repos/BBAttacks/results"

#TODO: get outpath from arguments
RUN_NAME = "dummy_run"

run_output_folder = os.path.join(OUT_PATH, RUN_NAME)
os.mkdir(run_output_folder)

print(f"Results will be stored in {run_output_folder}\n")

#TODO: get task from argument
task = "cifar100"

print(f"Task: {task}\n")

if task == "cifar100":
    from tensorflow.keras.datasets import cifar100
    from tensorflow import keras
    NUM_CLASSES = 100
    
    _, (x_test, y_test) = cifar100.load_data()

    x_test = x_test.astype('int')
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
    print("Loaded test data")
    
    preds_test = model_instance.predict(x_test)
    preds_test_labels = np.argmax(preds_test, axis=1)
    true_test_labels = np.argmax(y_test, axis=1)
    
    acc_test = np.mean(true_test_labels == preds_test_labels)
    print(f"Test accuracy: {acc_test}\n")
    
    print("Saving only the correctly classified images to be perturbed")
    x_test_correct = x_test[preds_test_labels == true_test_labels]
    y_test_correct = y_test[preds_test_labels == true_test_labels]
          
    print(f"Saved {len(x_test_correct)} samples out of {len(x_test)} samples")

    LEFT_IDX_ATTACK_IMAGES = 0
    RIGHT_IDX_ATTACK_IMAGES = 10
    SAMPLE_IMAGES = x_test_correct[LEFT_IDX_ATTACK_IMAGES: RIGHT_IDX_ATTACK_IMAGES]
    SAMPLE_Y = y_test_correct[LEFT_IDX_ATTACK_IMAGES: RIGHT_IDX_ATTACK_IMAGES]
    print("Selected data sample to attack\n")
    
    print("STARTING THE ATTACK")
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
else:
    print(f"Task {task} not supported")
    sys.exit()

evoba_stats = utils.get_evoba_stats(adv_evo_strategy)
utils.print_evoba_stats(evoba_stats)

utils.save_evoba_artifacts(evoba_stats, run_output_folder)

print(f"Artifacts saved at path {run_output_folder}")