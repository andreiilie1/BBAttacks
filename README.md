# BBAttacks
Black-Box Adversarial Attacks for Image Classifiers

## Structure
- **attacks** contains all the classes encapsulating various BB attacks: MAB, epsilon-greedy, EvoBA, (our version of) SimBA, etc
- **models** contains models artifacts (usually, most will be left out because of their big dimensions)
- **notebooks** contains multiple experiments we've been running (in the ipynb format). Most notably, the 
**cifar_10_experiments.ipynb** and **imagenet_experiments.ipynb** were used in the **EvoBA: An Evolution Strategy as a Strong Baseline for
Black-Box Adversarial Attacks** paper experiments
- **utils** contains various helper functions

## Important note about EvoBA
As there are multiple experiments and attacks in this repo, we provide a pack with the minimal necessary scripts and notebooks to replicate the EvoBA results from 
**EvoBA: An Evolution Strategy as a Strong Baseline for Black-Box Adversarial Attacks**. This pack includes the used data and models as well. It can be found here: https://drive.google.com/file/d/1OVn3w0VBtW5x84LTsJUymgqYlnJKwCtA/view?usp=sharing
