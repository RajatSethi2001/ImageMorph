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

#Data that predict_wrapper will use that contains victim model and other data-processing variables.
victim_data = {
    "model": models.load_model("mnist")
}

#Numpy array to be morphed (Will not affect the original file).
attack_array = cv2.imread("MNIST.png", 0)

#A 2-length tuple that stores the minimum and maximum values for the attack array.
array_range = (0, 255)

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
    def __init__(self, predict_wrapper, victim_data, attack_array, array_range, new_class, framework, param_file, trials, timesteps):
        self.predict_wrapper = predict_wrapper
        self.victim_data = victim_data
        self.attack_array = attack_array
        self.array_range = array_range
        self.new_class = new_class
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
        self.study.optimize(self.optimize_a2c, n_trials=self.trials)
        pickle.dump(self.study, open(self.param_file, 'wb'))
    
    def get_a2c(self, trial):
        #These are the hyperparameters that Stable-Baselines3 uses.
        #In particular, these are the hyperparameters suggested by rl-zoo.
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64])
        gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995, 0.999])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 0.6, 0.7, 0.8, 0.9, 1])
        
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2)
        ent_coef = trial.suggest_float("ent_coef", 0.000001, 0.1)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1)

        return {
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
        }
    
    def get_ppo(self, trial):
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995, 0.999])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2)
        ent_coef = trial.suggest_float("ent_coef", 0.000001, 0.1)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1)

        if batch_size > n_steps:
            batch_size = n_steps

        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef
        }
    
    def get_td3(self, trial):
        buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        gamma = trial.suggest_categorical("gamma", [0.95, 0.98, 0.99, 0.995, 0.999])
        tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05])
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128])
        gradient_steps = train_freq

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-2)

        return {
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "tau": tau,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "learning_rate": learning_rate
        }

    def optimize_framework(self, trial):
        #Save the current study in the pickle file.
        pickle.dump(self.study, open(self.param_file, 'wb'))

        #Create an environment and model to test out the hyperparameters. 
        env = MorphEnv(self.predict_wrapper, self.victim_data, self.attack_array, self.array_range, self.new_class, similarity=1)

        hyperparams = {}
        #Guess the optimal hyperparameters for testing in this trial.
        if self.framework == "A2C":
            hyperparams = self.get_a2c(trial)
            model = A2C("MlpPolicy", env, **hyperparams)
        elif self.framework == "PPO":
            hyperparams = self.get_ppo(trial)
            model = PPO("MlpPolicy", env, **hyperparams)
        elif self.framework == "TD3":
            hyperparams = self.get_td3(trial)
            model = TD3("MlpPolicy", env, **hyperparams)
        else:
            raise Exception(f"Unknown Framework: {self.framework} - Available Frameworks: (A2C)")

        #Run the trial for the designated number of timesteps.
        model.learn(self.timesteps, progress_bar=True)

        #Return the best reward as the score for this trial.
        reward = env.get_best_reward()
        return reward

if __name__=='__main__':
    param_finder = ParamFinder(predict_wrapper, victim_data, attack_array, array_range, new_class, framework, param_file, trials, timesteps)
    param_finder.run()


