import cv2
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pickle
import tensorflow as tf

from MorphEnv import MorphEnv
from os.path import exists
from stable_baselines3 import PPO, TD3, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from torch import nn as nn

#Callback class that saves the model after a set interval of steps.
class MorphCheckpoint(CheckpointCallback):
    def __init__(self, save_interval, rl_model):
        super().__init__(save_interval, ".", name_prefix=rl_model)
        self.save_interval = save_interval
        self.rl_model = rl_model
    
    def _on_step(self) -> bool:
        if self.save_interval > 0 and self.n_calls % self.save_interval == 0:            
            if self.rl_model is not None:
                self.model.save(self.rl_model)
        return True

def run(predict_wrapper, victim_data, attack_array, array_range, new_class, similarity=0.7, framework="PPO", render_level=0, checkpoint_level=0, checkpoint=None, graph_file=None, rl_model=None, save_interval=1000, param_file=None):
    #Hyperparameters collected from Optuna.py
    hyperparams = {}
    if param_file is not None:
        study = pickle.load(open(param_file, 'rb'))
        hyperparams = study.best_params

    #Environment that will conduct the attack.
    env = MorphEnv(predict_wrapper, victim_data, attack_array, array_range, new_class, similarity, render_level, checkpoint_level, checkpoint, graph_file)
    checkpoint_callback = MorphCheckpoint(save_interval, rl_model)

    if framework not in {"A2C", "PPO", "TD3"}:
        raise Exception(f"Unknown Framework: {framework} - Available Frameworks: (A2C, PPO, TD3)")

    if rl_model is not None and exists(rl_model):
        model_attack = eval(f"{framework}.load(\"{rl_model}\", env=env, **hyperparams)")
    
    #RL models to use for testing.
    else:
        policy_name = "MlpPolicy"
        if framework == "A2C":
            model_attack = A2C(policy_name, env, **hyperparams)
        elif framework == "PPO":
            model_attack = PPO(policy_name, env)
        elif framework == "TD3":
            model_attack = TD3(policy_name, env)
    
    model_attack.learn(100000, progress_bar=True, callback=checkpoint_callback)