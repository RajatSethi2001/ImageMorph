import cv2
import numpy as np
import tensorflow as tf

from MorphEnv import MorphEnv
from os.path import exists
from stable_baselines3 import PPO, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def run(image_file, victim, classes, new_class, action=0, similarity=0.9, scale_image=True, render_interval=0, save_interval=1000, framework="PPO", rl_model="DefaultModel.zip"):
    model_victim = tf.keras.models.load_model(victim)
    model_config = model_victim.get_config()
    dimensions = model_config["layers"][0]["config"]["batch_input_shape"][1::]

    greyscale = False
    #If greyscale
    if len(dimensions) == 2 or dimensions[2] == 1:
        greyscale = True
        image_input = cv2.imread(image_file, 0)
            
    #Else, image is a 3D RGB image
    else:
        image_input = cv2.imread(image_file)

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

    env = MorphEnv(model_victim, image_input, image_file, classes, new_class, action, similarity, scale_image, render_interval, save_interval)
    n_timesteps = save_interval

    if framework not in {"A2C", "PPO", "TD3"}:
        raise Exception(f"Unknown Framework: {framework} - Available Frameworks: (A2C, PPO, TD3)")

    policy_name = "CnnPolicy"
    if framework == "A2C":
        model_attack = A2C(policy_name, env)

    elif framework == "PPO":
        model_attack = PPO(policy_name, env)

    elif framework == "TD3":
        n_actions = env.action_space.shape
        action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions), np.ones(n_actions) * 0.5)
        model_attack = TD3(policy_name, env, action_noise=action_noise)
    
    if exists(rl_model):
        model_attack.set_parameters(f"{rl_model}")

    while True:
        model_attack.learn(n_timesteps, progress_bar=True)
        model_attack.save(f"{rl_model}")