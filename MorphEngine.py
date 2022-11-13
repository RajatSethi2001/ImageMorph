import cv2
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

class MorphCheckpoint(CheckpointCallback):
    def __init__(self, save_interval, rl_model):
        super().__init__(save_interval, ".", name_prefix=rl_model)
        self.save_interval = save_interval
        self.rl_model = rl_model
    
    def _on_step(self) -> bool:
        if self.save_interval > 0 and self.rl_model is not None and self.n_calls % self.save_interval == 0:
            self.model.save(self.rl_model)
        return True

def run(predict_wrapper, image_file, grayscale, victim_data, new_class, action=0, similarity=0.7, render_level=0, checkpoint_level=0, checkpoint_file=None, framework="PPO", rl_model=None, save_interval=1000, param_file=None):
    hyperparams = {}
    if param_file is not None:
        study = pickle.load(open(param_file, 'rb'))
        hyperparams = study.best_params

    env = MorphEnv(predict_wrapper, image_file, grayscale, victim_data, new_class, action, similarity, render_level, checkpoint_level, checkpoint_file)
    checkpoint_callback = MorphCheckpoint(save_interval, rl_model)

    if framework not in {"A2C", "PPO", "TD3"}:
        raise Exception(f"Unknown Framework: {framework} - Available Frameworks: (A2C, PPO, TD3)")

    if rl_model is not None and exists(rl_model):
        model_attack = eval(f"{framework}.load(\"{rl_model}\", env=env, **hyperparams)")
    
    else:
        policy_name = "MlpPolicy"
        if framework == "A2C":
            model_attack = A2C(policy_name, env, **hyperparams)
        elif framework == "PPO":
            model_attack = PPO(policy_name, env)
        elif framework == "TD3":
            n_actions = env.action_space.shape
            action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions), np.ones(n_actions) * 0.5)
            model_attack = TD3(policy_name, env, action_noise=action_noise)
    
    model_attack.learn(100000, progress_bar=True, callback=checkpoint_callback)