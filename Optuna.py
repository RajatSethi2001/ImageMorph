import cv2
import numpy as np
import optuna
import pickle
import tensorflow as tf

from MorphEnv import MorphEnv
from os.path import exists
from stable_baselines3 import PPO, TD3, A2C
from tensorflow.keras import models
from torch import nn as nn

def predict_wrapper(image, victim_data):
    victim = victim_data["model"]
    image_input = image.reshape((1,) + image.shape)  / 255.0
    return victim.predict(image_input, verbose=0)[0]

image_file = "MNIST.png"
grayscale = True
victim_data = {
    "model": models.load_model("mnist")
}
new_class = 5
action = 1
similarity = 0.7
framework = "A2C"
param_file = "A2C-Params.pkl"
trials = 20
timesteps = 1000
episodes = 4
steps_per_episode = 250

class ParamFinder:
    def __init__(self, predict_wrapper, image_file, grayscale, victim_data, new_class, action, similarity, framework, param_file, trials, timesteps, episodes, steps_per_episode):
        self.predict_wrapper = predict_wrapper
        self.image_file = image_file
        self.grayscale = grayscale
        self.victim_data = victim_data
        self.new_class = new_class
        self.action = action
        self.similarity = similarity
        self.framework = framework
        self.param_file = param_file
        self.trials = trials
        self.timesteps = timesteps
        self.episodes = episodes
        self.steps_per_episode = steps_per_episode

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
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        normalize_advantage = trial.suggest_categorical("normalize_advantage", [True])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 0.6, 0.7, 0.8, 0.9, 1])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-1)
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
        pickle.dump(self.study, open(self.param_file, 'wb'))
        hyperparams = self.get_a2c(trial)
        env = MorphEnv(self.predict_wrapper, self.image_file, self.grayscale, self.victim_data, self.new_class, self.action, self.similarity)
        model = A2C("CnnPolicy", env, **hyperparams)
        model.learn(self.timesteps, progress_bar=True)

        rewards = []
        for episode in range(self.episodes):
            obs = env.reset()
            reward_total = 0
            for step in range(self.steps_per_episode):
                action, _ = model.predict(obs)
                obs, reward, _, info = env.step(action)
                reward_total += reward

            rewards.append(reward_total)
        
        print(rewards)
        return np.mean(rewards)

if __name__=='__main__':
    param_finder = ParamFinder(predict_wrapper, image_file, grayscale, victim_data, new_class, action, similarity, framework, param_file, trials, timesteps, episodes, steps_per_episode)
    param_finder.run()


