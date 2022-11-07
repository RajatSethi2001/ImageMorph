import cv2
import numpy as np
import tensorflow as tf

from MorphEnv import MorphEnv
from stable_baselines3 import PPO, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def run(image_file, victim, classes, new_class, action=0, similarity=0.9, scale_image=True, checkpoint_file=None, render_interval=0, save_interval=1000, framework="PPO"):
    model_victim = tf.keras.models.load_model(victim)
    model_config = model_victim.get_config()
    dimensions = model_config["layers"][0]["config"]["batch_input_shape"][1::]
    
    checkpoint_image = None
    greyscale = False
    #If greyscale
    if len(dimensions) == 2 or dimensions[2] == 1:
        greyscale = True
        image_input = cv2.imread(image_file, 0)
        if checkpoint_file != None:
            checkpoint_image = cv2.imread(checkpoint_file, 0)
            
    #Else, image is a 3D RGB image
    else:
        image_input = cv2.imread(image_file)
        if checkpoint_file != None:
            checkpoint_image = cv2.imread(checkpoint_file)

    #Determine the smaller dimension
    min_length = min(dimensions[0], dimensions[1])
    #Scale images (sb3 requires images to be > 36x36)
    if min_length < 36:
        scale = 36 / min_length
        new_height = int(dimensions[0] * scale)
        new_width = int(dimensions[1] * scale)
        image_input = cv2.resize(image_input, (new_height, new_width))
        if checkpoint_file != None:
            checkpoint_image = cv2.resize(checkpoint_image, (new_height, new_width))

    #If greyscale, add an extra dimension (makes processing easier)
    if greyscale:
         image_input = image_input.reshape(image_input.shape + (1,))
         if checkpoint_file != None:
            checkpoint_image = checkpoint_image.reshape(checkpoint_image.shape + (1,))

    env = MorphEnv(model_victim, image_input, image_file, classes, new_class, action, similarity, scale_image, checkpoint_image, render_interval, save_interval)

    if framework == "TD3":
        policy_name = "CnnPolicy"
        buffer_size=200000
        n_timesteps = 1
        batch_size = 1

        n_actions = env.action_space.shape
        action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(n_actions), np.ones(n_actions) * 0.5)

        config = TD3(policy_name, env, buffer_size=buffer_size, batch_size=batch_size, action_noise=action_noise, verbose=0)
        config.learn(n_timesteps, progress_bar=True)
    
    elif framework == "PPO":
        policy_name = "CnnPolicy"
        n_timesteps = 1000000
        batch_size = 8
        config = PPO(policy_name, env, batch_size=batch_size)
        config.learn(n_timesteps, progress_bar=True)

    elif framework == "A2C":
        policy_name = "CnnPolicy"
        n_timesteps = 1000000
        config = A2C(policy_name, env)
        config.learn(n_timesteps, progress_bar=True)

    else:
        raise Exception(f"Unknown Framework: {framework} - Available Frameworks: (A2C, PPO, TD3)")

    print("Could not finish morph. Please start again with checkpoint image.")