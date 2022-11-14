import cv2
import numpy as np
import optuna
import pickle
import tensorflow as tf

from MorphEnv import MorphEnv
from os.path import exists
from stable_baselines3 import PPO, TD3, A2C
from stable_baselines3.common.callbacks import BaseCallback
from tensorflow.keras import models
from torch import nn as nn

#Wrapper function that takes in the current perturbed image, the victim model, and any associated data.
#The victim model should predict the class of the image, then return the outcome.
#Can return either a list of numbers (for standard classifications) or an object (for total black-box)
def predict_wrapper(image, victim_data):
    victim = victim_data["model"]
    image_input = image.reshape((1,) + image.shape)  / 255.0
    return victim.predict(image_input, verbose=0)[0]

#Filename of image to be morphed (Will not affect the original image)
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

#Which RL framework to use (Currently only supports A2C, will add the other frameworks soon)
framework = "A2C"

#Where to save the optimal hyperparameters (This is a .pkl file). Can also be set to an existing file to continue trials.
param_file = "A2C-Params.pkl"

#How many trials to run for this iteration.
trials = 20

#How many timesteps to run through per trial.
timesteps = 2000

class ParamFinder:
    def __init__(self, predict_wrapper, image_file, grayscale, victim_data, new_class, action, framework, param_file, trials, timesteps):
        self.predict_wrapper = predict_wrapper
        self.image_file = image_file
        self.grayscale = grayscale
        self.victim_data = victim_data
        self.new_class = new_class
        self.action = action
        self.framework = framework
        self.param_file = param_file
        self.trials = trials
        self.timesteps = timesteps

        #Retrieve existing parameters if they exist.
        if exists(param_file):
            self.study = pickle.load(open(self.param_file, 'rb'))
        else:
            self.study = optuna.create_study(direction="maximize")

    def run(self):
        if self.framework == "A2C":
            self.study.optimize(self.optimize_a2c, n_trials=self.trials)
        else:
            raise Exception(f"Unknown Framework: {self.framework} - Available Frameworks: (A2C)")
        
        pickle.dump(self.study, open(self.param_file, 'wb'))
    
    def get_a2c(self, trial):
        #These are the hyperparameters that Stable-Baselines3 uses.
        #In particular, these are the hyperparameters suggested by rl-zoo.
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        normalize_advantage = trial.suggest_categorical("normalize_advantage", [True])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 0.6, 0.7, 0.8, 0.9, 1])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2)
        ent_coef = trial.suggest_float("ent_coef", 0.000001, 0.1)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1)

        return {
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "normalize_advantage": normalize_advantage,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
        }

    def optimize_a2c(self, trial):
        #Save the current study in the pickle file.
        pickle.dump(self.study, open(self.param_file, 'wb'))

        #Guess the optimal hyperparameters for testing in this trial.
        hyperparams = self.get_a2c(trial)

        #Create an environment and model to test out the hyperparameters. 
        env = MorphEnv(self.predict_wrapper, self.image_file, self.grayscale, self.victim_data, self.new_class, 1)
        model = A2C("MlpPolicy", env, **hyperparams)

        #Run the trial for the designated number of timesteps.
        model.learn(self.timesteps, progress_bar=True)

        #Return the best reward as the score for this trial.
        reward = env.get_best_reward()
        return reward

if __name__=='__main__':
    param_finder = ParamFinder(predict_wrapper, image_file, grayscale, victim_data, new_class, framework, param_file, trials, timesteps)
    param_finder.run()


