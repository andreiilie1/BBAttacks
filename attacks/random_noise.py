import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Noisy applies the given type of noise to an image
def noisy(image, noise_type="gauss", args={}):
    row, col, ch = image.shape
    
    if 'normalize' in args:
        normalize = args['normalize']
    else:
        normalize = 255
    
    if noise_type == "gauss":
        mean = 0
        var = 126
        sigma = var ** 0.5
        
        gauss = np.random.normal(mean,sigma,(row,col,ch)) / normalize
        gauss = gauss.reshape(row,col,ch)
        
        noisy = image + gauss
        noisy = np.minimum(noisy, 255)
        noisy = np.maximum(noisy, 0)
    elif noise_type == "cells":
        noisy = np.copy(image)
        
        if "cell_count" in args:
            cell_count = args["cell_count"]
        else:
            cell_count = 32
            
        for i in range(cell_count):
            cell_x = np.random.randint(row)
            cell_y = np.random.randint(col)
            for channel in range(ch):
                noisy[cell_x][cell_y][channel] = np.random.randint(255) / normalize

    return noisy

# Apply_noise_to_data applies the given type of noise to an array of images
def apply_noise_to_data(x_test, noise_type="gauss", args = {}):
    apply_noise_to_image = lambda x : noisy(x, noise_type, args)
    noisy_imgs = np.array(list(map(apply_noise_to_image, x_test)))
    return noisy_imgs

# Check_noise_robustness applies the given type of noise to an array of images
# and measures the accuracy of a model on top of the resulting noisy images.
# X_test is the array of images. Y_test is the array of ground truths, containing
# for each corresponding x_test entry a one-hot encoding with the correct class.
# X_test and y_test use the types of the MNIST data loaded through TF datasets.
def check_noise_robustness(model, x_test, y_test, noise_type="gauss", args = {}):
    noisy_imgs = apply_noise_to_data(x_test, noise_type, args)
    
    noisy_pred = model.predict(np.array(noisy_imgs))
    noisy_labels = np.argmax(noisy_pred, axis=1)
    
    correct_labels = np.argmax(y_test, axis = 1)
    agreements = noisy_labels == correct_labels
    
    accuracy = agreements.sum() / len(x_test)
    return accuracy, agreements, noisy_imgs

# Check_noise_robustness_multiple_rounds performs independent series of noising
# on the sample_x set of images. Adding noise on top of an image is stopped either
# when the model does not predict the label of the image correctly, which is saved
# in sample_y, or when the maximum number of steps has been reached.
def check_noise_robustness_multiple_rounds(model, sample_x, sample_y, steps = 5, noise_type="gauss", verbose = True, args = {}):
    # TODO: add early stopping
    K = len(sample_x)
    number_queries = {}
    print("K:",K)
    
    # Robustness_progress[i] measures the percentage of images from sample_x which
    robustness_progress = []
    
    # Saved_noisy_imgs[i] will either be a noisy version of the image sample_x[i] if 
    # the model labels it incorrectly in at most 'steps' maximum steps, or [] otherwise.
    saved_noisy_imgs = [[]] * K
    
    if verbose:
        clear_output(wait=True)
        print("Step", 0)
    
    # Applying one noise step to all the images
    accuracy, agreements, noisy_imgs = check_noise_robustness(model, sample_x, sample_y, noise_type, args)
    robustness_progress.append(np.sum(agreements)/len(agreements))
    
    for i in range(K):
        if(agreements[i] == False and saved_noisy_imgs[i] == []):
            saved_noisy_imgs[i] = noisy_imgs[i]
    
    for i in range(steps - 1):
#         print("\n\n\n")
#         print("Round", i)
#         print("agreements:", agreements)
        if verbose:
            clear_output(wait=True)
            print("Step", i + 1, "/", steps)
            print("Previous robustness: ", np.sum(agreements) / len(agreements))
            
            plt.plot(robustness_progress)
            plt.show()
            
        accuracy_local, agreements_local, noisy_imgs_local = check_noise_robustness(model, sample_x, sample_y, noise_type, args)
#         print("agreements_local:", agreements_local)
        
        # Agreements[j] is true if the image sample_x[j] was correctly classified by 
        # all the attempts of adding noise up to the current (i) step
        agreements = np.logical_and(agreements, agreements_local)
        robustness_progress.append(np.sum(agreements)/len(agreements))
#         print("agreements:", agreements)

        # If we manage to add noise to an image and get it misclassified for the first
        # time, save the noisy image in saved_noisy_imgs
        for j in range(K):
            if(agreements[j] == False and saved_noisy_imgs[j] == []):
                print("Updated sample", j)
                saved_noisy_imgs[j] = noisy_imgs_local[j]
                number_queries[j] = i
    if verbose:
        clear_output(wait=True)
        
    return agreements, saved_noisy_imgs, number_queries