import cv2
import numpy as np
from tensorflow.keras import models

from GaslightEngine import run

# ===================================================================================================
# REQUIRED PARAMETERS
# These parameters decide which image to perturb and which classifier to attack.
# They must be filled out so that the algorithm understands what its objective is.
# Use the example below for MNIST to get started
# ===================================================================================================

#Wrapper function that takes in the current perturbed image, the victim model, and any associated data.
#The victim model should predict the class of the image, then return the outcome.
#Can return either a list of numbers (for standard classifications) or an object (for total black-box)
def predict_wrapper(image, victim_data):
    victim = victim_data["model"]
    image_input = image.reshape((1,) + image.shape) / 255.0
    return victim.predict(image_input, verbose=0)[0]

#Data that predict_wrapper will use that contains victim model and other data-processing variables.
victim_data = {
    "model": models.load_model('Classifiers/mnist')
}

#Numpy array to be morphed (Will not affect the original file).
attack_array = cv2.imread("Inputs/MNIST.png", 0)

#A 2-length tuple that stores the minimum and maximum values for the attack array.
array_range = (0, 255)

# The intended outcome for perturbation.
# If predict_wrapper returns a list of numbers, this is the index to maximize (use zero-based indexing)
# If predict_wrapper returns an object, this is the intended value
# If new_class is None, then it will perform an untargeted attack (i.e, it does not matter what the final outcome is, as long as its different from the original)
new_class = 5

#Minimum similarity needed for a successful morph [0-1]
similarity = 0.9

#Name of the .npy file used to save the final results when successfully perturbed.
result_file = "Outputs/MNIST-Soft-Targeted-Reset.npy"

#Which RL framework to use (A2C, PPO, TD3)
framework = "PPO"

# ===================================================================================================
# ADVANCED PARAMETERS
# These parameters are used for debugging purposes.
# They are used to either display progress or save models/images/params for future use.
# Assume these are the default settings, they can be left alone.
# ===================================================================================================

#Display progress after how many steps. Set to 0 for no render.
render_interval = 10

#Save model, checkpoint, and graph after how many steps. Set to 0 for no save.
save_interval = 100

#Checkpoint file used to start (if it exists) and save perturbation (Should be a .npy file with the same shape as attack_array). 
#Set to None to disable checkpoints. (Note: Checkpoints can also be turned off by save_interval)
checkpoint_file = None

#File to store graphing information (Perturbance, Similarity, Reward). Set to None for no graphing. (Note: Graphing can also be turned off by save_interval).
graph_file = None

#Which RL model to use/save to (If it doesn't exist, it will be created). Stable-Baselines3 uses a .zip file. Set to None for no model.
rl_model = None

#Which hyperparameter pickle file to use from Optuna.py (Make sure it matches the framework). Set to None to use default hyperparameters
param_file = None

run(predict_wrapper, victim_data, attack_array, array_range, new_class, similarity, result_file, framework, render_interval, save_interval, checkpoint_file, graph_file, rl_model, param_file)

