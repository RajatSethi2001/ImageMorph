from MorphEngine import run
from tensorflow.keras import models

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
    image_input = image.reshape((1,) + image.shape)  / 255.0
    return victim.predict(image_input, verbose=0)[0]

#Filename of image to be morphed (Will not affect the original image)
image_file = "CIFAR10.png"

#Is the image grayscale? True for Grayscale, False for RGB.
grayscale = False

#Data that predict_wrapper will use that contains victim model and other data-processing variables.
victim_data = {
    "model": models.load_model("cifar10")
}

# The intended outcome for perturbation.
# If predict_wrapper returns a list of numbers, this is the index to maximize (use zero-based indexing)
# If predict_wrapper returns an object, this is the intended value
new_class = 9

#Which action space to use
#Action 0 - Edit one pixel at a time by -255 or +255 (This might be a bit broken, still testing)
#Action 1 - Edit one pixel at a time by changing it to a value between [0-255]
action = 1

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

#0 for no render, 1 for light render (only print), 2 for full render (pyplot graph)
render_level = 1

#0 for saving the final result, 1 for also saving results that beat the classifier but not similarity, 2 for saving results with the best reward
checkpoint_level = 2

#Checkpoint image to start perturbation and save checkpoint files. Set to None to use original.
#If checkpoint_level > 0 but checkpoint_file is None, then checkpoints will be saved to "Checkpoint{image_file}"
checkpoint_file = "CheckpointCIFAR.png"

#File to store graphing information (Perturbance, Similarity, Reward). Set to None for no graphing. (Note: This is independent from render_level). Set to None for no graphing.
graph_file = "GraphCIFAR.png"

#Which RL model to use/save to (If it doesn't exist, it will be created). Stable-Baselines3 uses a .zip file. Set to None for no model.
rl_model = None

#Save model after how many steps. Set to 0 for no save.
save_interval = 0

#Which hyperparameter pickle file to use from Optuna.py (Make sure it matches the framework). Set to None to use default hyperparameters
param_file = None

run(predict_wrapper, image_file, grayscale, victim_data, new_class, action, similarity, framework, render_level, checkpoint_level, checkpoint_file, graph_file, rl_model, save_interval, param_file)

