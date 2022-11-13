from MorphEngine import run
from tensorflow.keras import models

#Wrapper function that takes in the current perturbed image, the victim model, and any associated data.
#The victim model should predict the class of the image, then return the outcome.
#Can return either a list of numbers (for standard classifications) or an object (for total black-box)
def predict_wrapper(image, victim_data):
    victim = victim_data["model"]
    image_input = image.reshape((1,) + image.shape)  / 255.0
    return victim.predict(image_input, verbose=0)[0]

#Filename of image to be morphed
image_file = "MNIST.png"

#Is the image grayscale? True for Grayscale, False for RGB.
grayscale = True

#Data that predict_wrapper will use that contains victim model and other data-processing variables.
victim_data = {
    "model": models.load_model("mnist")
}

# The intended outcome for perturbation.
# If predict_wrapper returns a list of numbers, this is the index to maximize
# If predict_wrapper returns an object, this is the intended value
new_class = 5

#Which action space to use (0-3)
#Action 0 - Edit one pixel at a time by -255 or +255
#Action 1 - Edit one pixel at a time by changing it to a value between [0-255]
action = 1

#Minimum similarity needed for a successful morph [0-1]
similarity = 0.9

#0 for no render, 1 for light render (only print), 2 for full render (pyplot graph)
render_level = 1

#0 for saving the final result, 1 for also saving results that beat the classifier but not similarity, 2 for saving results with the best reward
checkpoint_level = 2

#Checkpoint image to start perturbation. Set to None to use original
checkpoint_file = "CheckpointMNIST.png"

#Which RL framework to use (A2C, PPO, TD3)
framework = "PPO"

#Which RL model to use/save to (If it doesn't exist, it will be created)
rl_model = None

#Save model after how many steps. Set to 0 for no save.
save_interval = 0

#Which hyperparameter pickle file to use (Make sure it matches the framework)
param_file = None

run(predict_wrapper, image_file, grayscale, victim_data, new_class, action, similarity, render_level, checkpoint_level, checkpoint_file, framework, rl_model, save_interval, param_file)

