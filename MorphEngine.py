import cv2
import numpy as np
import optuna
import pickle
import tensorflow as tf

from MorphEnv import MorphEnv
from os.path import exists
from stable_baselines3 import PPO, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

class MorphEngine:
    def __init__(self, image_file, victim, classes, new_class, action=0, similarity=0.9, scale_image=True, render_interval=0, save_interval=1000, framework="PPO", rl_model="DefaultModel.zip", param_file=None):
        self.image_file = image_file
        self.victim = victim
        self.classes = classes
        self.new_class = new_class
        self.action = action
        self.similarity = similarity
        self.scale_image = scale_image
        self.render_interval = render_interval
        self.save_interval = save_interval
        self.framework = framework
        self.rl_model = rl_model
        self.param_file = param_file

    def run(self):
        model_victim = tf.keras.models.load_model(self.victim)
        model_config = model_victim.get_config()
        dimensions = model_config["layers"][0]["config"]["batch_input_shape"][1::]

        greyscale = False
        #If greyscale
        if len(dimensions) == 2 or dimensions[2] == 1:
            greyscale = True
            image_input = cv2.imread(self.image_file, 0)
                
        #Else, image is a 3D RGB image
        else:
            image_input = cv2.imread(self.image_file)

        #Determine the smaller dimension
        min_length = min(dimensions[0], dimensions[1])
        #Scale images (sb3 requires images to be > 36x36)
        if min_length < 36:
            scale = 36 / min_length
            new_height = int(dimensions[0] * scale)
            new_width = int(dimensions[1] * scale)
            image_input = cv2.resize(image_input, (new_height, new_width))

        #If greyscale, add an extra dimension (makes processing easier)
        if greyscale:
            image_input = image_input.reshape(image_input.shape + (1,))

        self.env = MorphEnv(model_victim, image_input, self.image_file, self.classes, self.new_class, self.action, self.similarity, self.scale_image, self.render_interval, self.save_interval)
        n_timesteps = self.save_interval

        if self.framework not in {"A2C", "PPO", "TD3"}:
            raise Exception(f"Unknown Framework: {self.framework} - Available Frameworks: (A2C, PPO, TD3)")

        if exists(self.rl_model):
            model_attack = eval(f"{self.framework}.load(\"{self.rl_model}\", env=self.env)")

        else:
            policy_name = "CnnPolicy"
            if self.framework == "A2C":
                if self.param_file is not None:
                    study = pickle.load(open(self.param_file, 'rb'))
                    hyperparams = study.best_params
                    n_steps = hyperparams["n_steps"]
                    gamma = hyperparams["gamma"]
                    gae_lambda = hyperparams["gae_lambda"]
                    learning_rate = hyperparams["learning_rate"]
                    normalize_advantage = hyperparams["normalize_advantage"]
                    max_grad_norm = hyperparams["max_grad_norm"]
                    use_rms_prop = hyperparams["use_rms_prop"]
                    vf_coef = hyperparams["vf_coef"]
                    policy_kwargs = hyperparams["policy_kwargs"]
                    model_attack = A2C(policy_name, self.env, n_steps=n_steps, gamma=gamma, gae_lambda=gae_lambda, learning_rate=learning_rate, normalize_advantage=normalize_advantage, max_grad_norm=max_grad_norm, use_rms_prop=use_rms_prop, vf_coef=vf_coef, policy_kwargs=policy_kwargs)
                else:
                    model_attack = A2C(policy_name, self.env)
            elif self.framework == "PPO":
                model_attack = PPO(policy_name, self.env)

            elif self.framework == "TD3":
                n_actions = self.env.action_space.shape
                action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions), np.ones(n_actions) * 0.5)
                model_attack = TD3(policy_name, self.env, action_noise=action_noise)
        
        while True:
            model_attack.learn(n_timesteps, progress_bar=True)
            model_attack.save(f"{self.rl_model}")