import cv2
import numpy as np
import optuna
import pickle
import tensorflow as tf

from MorphEnv import MorphEnv
from os.path import exists
from stable_baselines3 import PPO, TD3, A2C
from torch import nn as nn

image_file = "MNIST.png"
victim = "mnist"
classes = [x for x in range(10)]
new_class = 5
scale_image = True
action = 0
similarity = 0.9
framework = "A2C"
param_file = "A2C-Params.pkl"

class ParamFinder:
    def __init__(self, image_file, victim, classes, new_class, scale_image, action, similarity, framework, param_file):
        self.image_file = image_file
        self.victim = tf.keras.models.load_model(victim)
        self.classes = classes
        self.new_class = new_class
        self.scale_image = scale_image
        self.action = action
        self.similarity = similarity
        self.framework = framework
        self.param_file = param_file

        self.image = self.get_image()
        if exists(param_file):
            self.study = pickle.load(open(self.param_file, 'rb'))
        else:
            self.study = optuna.create_study(direction="maximize")

        self.env = MorphEnv(self.victim, self.image, self.image_file, self.classes, self.new_class, self.action, self.similarity, self.scale_image, 0, 0)
    
    def get_image(self):
        model_config = self.victim.get_config()
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
        
        return image_input

    def run(self):
        if self.framework == "A2C":
            self.study.optimize(self.optimize_a2c, n_trials=200, show_progress_bar=True)
        else:
            raise Exception(f"Unknown Framework: {self.framework} - Available Frameworks: (A2C)")
        
        pickle.dump(self.study, open(self.param_file, 'wb'))
    
    def get_a2c(self, trial):
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
        gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1)
        ent_coef = trial.suggest_float("ent_coef", 0.000001, 0.1)
        vf_coef = trial.suggest_float("vf_coef", 0.1, 1)
        log_std_init = trial.suggest_float("log_std_init", -4, 1)
        ortho_init = trial.suggest_categorical("ortho_init", [False, True])
        net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
        activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

        net_arch = {
            "small": [dict(pi=[64, 64], vf=[64, 64])],
            "medium": [dict(pi=[256, 256], vf=[256, 256])],
        }[net_arch]

        activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

        return {
            "n_steps": n_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "normalize_advantage": normalize_advantage,
            "max_grad_norm": max_grad_norm,
            "use_rms_prop": use_rms_prop,
            "vf_coef": vf_coef,
            "policy_kwargs": dict(
                log_std_init=log_std_init,
                net_arch=net_arch,
                activation_fn=activation_fn,
                ortho_init=ortho_init,
            ),
        }

    def optimize_a2c(self, trial):
        hyperparams = self.get_a2c(trial)
        model = A2C("CnnPolicy", self.env, **hyperparams)
        model.learn(100, progress_bar=True)

        rewards = []
        reward_total = 0
        last_reward = 0
        same_reward_count = 0
        episode = 0
        obs = self.env.reset()
        while episode < 1:
            action, _ = model.predict(obs)
            obs, reward, done, _ = self.env.step(action)
            reward_total += reward
            if done:
                rewards.append(reward_total)
                reward_total = 0
                last_reward = 0
                same_reward_count = 0
                episode += 1
                obs = self.env.reset()
            elif reward == last_reward:
                same_reward_count += 1
                if same_reward_count > 4:
                    rewards.append(reward_total)
                    reward_total = 0
                    last_reward = 0
                    same_reward_count = 0
                    episode += 1
                    obs = self.env.reset()
            else:
                same_reward_count = 0
                last_reward = reward
        
        pickle.dump(self.study, open(self.param_file, 'wb'))
        return np.mean(rewards)

if __name__=='__main__':
    param_finder = ParamFinder(image_file, victim, classes, new_class, scale_image, action, similarity, framework, param_file)
    param_finder.run()


