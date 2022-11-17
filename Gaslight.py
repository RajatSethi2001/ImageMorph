import cv2
import numpy as np
from tensorflow.keras import models

from MorphEngine import run

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
    "model": models.load_model('mnist')
}

#Numpy array to be morphed (Will not affect the original file).
attack_array = cv2.imread("MNIST.png")

#A 2-length tuple that stores the minimum and maximum values for the attack array.
array_range = (0, 255)

# The intended outcome for perturbation.
# If predict_wrapper returns a list of numbers, this is the index to maximize (use zero-based indexing)
# If predict_wrapper returns an object, this is the intended value
# If new_class is None, then it will perform an untargeted attack (i.e, it does not matter what the final outcome is, as long as its different from the original)
new_class = 2

#Minimum similarity needed for a successful morph [0-1]
similarity = 0.9

#Which RL framework to use (A2C, PPO, TD3)
framework = "PPO"

# ===================================================================================================
# ADVANCED PARAMETERS
# These parameters are used for debugging purposes.
# They are used to either display progress or save models/images/params for future use.
# Assume these are the default settings, they can be left alone.
# ===================================================================================================

#0 for no render, 1 to print best perturbance + similarity
render_level = 1

#0 for only saving the final result, 1 for also saving results that beat the classifier but not similarity, 2 for saving results with the best reward
checkpoint_level = 2

#Checkpoint File to start and save perturbation (Should be a .npy file with the same shape as attack_array). Set to None to use original.
# Note: If checkpoint_file is None, checkpoints and final results will be stored in "Checkpoint.npy"
checkpoint_file = "CheckpointMNIST.npy"

#File to store graphing information (Perturbance, Similarity, Reward). Set to None for no graphing. (Note: This is independent from render_level).
graph_file = None

#Which RL model to use/save to (If it doesn't exist, it will be created). Stable-Baselines3 uses a .zip file. Set to None for no model.
rl_model = None

#Save model after how many steps. Set to 0 for no save.
save_interval = 0

#Which hyperparameter pickle file to use from Optuna.py (Make sure it matches the framework). Set to None to use default hyperparameters
param_file = None

run(predict_wrapper, victim_data, attack_array, array_range, new_class, similarity, framework, render_level, checkpoint_level, checkpoint_file, graph_file, rl_model, save_interval, param_file)

