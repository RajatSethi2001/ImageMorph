import copy
import cv2
import gym
import math
import matplotlib.pyplot as plt
import numpy as np

from gym.spaces import Box
from os.path import exists

class MorphEnv(gym.Env):
    def __init__(self, predict_wrapper, image_file, grayscale, victim_data, new_class, action=0, similarity=0.7, render_level=0, checkpoint_level=0, checkpoint_file=None):
        self.predict_wrapper = predict_wrapper
        self.victim_data = victim_data
        self.image_file = image_file
        self.checkpoint_file = checkpoint_file
        self.checkpoint_image = None
        self.grayscale = grayscale

        if self.grayscale:
            self.original_image = cv2.imread(self.image_file, 0)
            self.original_image = self.original_image.reshape(self.original_image.shape + (1,))
            if self.checkpoint_file is not None and exists(self.checkpoint_file):
                self.checkpoint_image = cv2.imread(self.checkpoint_file, 0)
                self.checkpoint_image = self.checkpoint_image.reshape(self.checkpoint_image.shape + (1,))
        #Else, image is a 3D RGB image
        else:
            self.original_image = cv2.imread(self.image_file)
            if self.checkpoint_file is not None and exists(self.checkpoint_file):
                self.checkpoint_image = cv2.imread(self.checkpoint_file)
        
        if self.checkpoint_image is None:
            self.perturb_image = copy.deepcopy(self.original_image)
        else:
            self.perturb_image = copy.deepcopy(self.checkpoint_image)

        self.shape = self.original_image.shape

        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)
        self.action = action
        if action == 0:
            self.action_space = Box(low=np.array([0, 0, 0, -1]), high=np.array([1, 1, 1, 1]), shape=(4,), dtype=np.float32)
        elif action == 1:
            self.action_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)
        else:
            raise Exception("Action must be an integer between 0-1")

        results = self.predict_wrapper(self.perturb_image, self.victim_data)
        self.result_type = "object"
        if isinstance(results, list) or isinstance(results, np.ndarray):
            self.result_type = "list"

        self.new_class = new_class

        self.best_reward = 0
        self.best_perturbance = 0
        self.best_similarity = 0
        self.similarity_threshold = similarity

        self.render_level = render_level
        self.checkpoint_level = checkpoint_level

        if self.render_level > 1:
            plt.ion()
            self.figure = plt.figure(figsize=(6,3))
            self.plot1 = self.figure.add_subplot(1,2,1)
            self.plot2 = self.figure.add_subplot(1,2,2)

    def step(self, action):
        perturb_test = copy.deepcopy(self.perturb_image)
        if self.action == 0:
            row = np.uint8(np.round(action[0] * (self.shape[0] - 1)))
            col = np.uint8(np.round(action[1] * (self.shape[1] - 1)))
            color = np.uint8(np.round(action[2] * (self.shape[2] - 1)))
            pixel_change = np.uint8(np.round(action[3] * 255))

            if pixel_change < 0:
                #check for overflow
                if (perturb_test[row][col][color] + pixel_change > perturb_test[row][col][color]):
                    perturb_test[row][col][color] = 0
                else:
                    perturb_test[row][col][color] += pixel_change
            else:
                if (perturb_test[row][col][color] + pixel_change < perturb_test[row][col][color]):
                    perturb_test[row][col][color] = 255
                else:
                    perturb_test[row][col][color] += pixel_change
        
        elif self.action == 1:
            row = np.uint8(np.round(action[0] * (self.shape[0] - 1)))
            col = np.uint8(np.round(action[1] * (self.shape[1] - 1)))
            color = np.uint8(np.round(action[2] * (self.shape[2] - 1)))
            pixel_change = np.uint8(np.round(action[3] * 255))

            perturb_test[row][col][color] = pixel_change

        results = self.predict_wrapper(perturb_test, self.victim_data)
        if self.result_type == "list":
            perturbance = results[self.new_class]
        else:
            if results == self.new_class:
                perturbance = 1
            else:
                perturbance = 0

        euclid_distance = 0
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                for color in range(self.shape[2]):
                    pixel_distance = ((int(perturb_test[row][col][color]) - int(self.original_image[row][col][color])) / 255.0) ** 2
                    euclid_distance += pixel_distance
        
        similarity = 1 - math.sqrt(euclid_distance / math.prod(self.shape))

        reduction_factor = 0.0001
        reduction_threshold = 0.95
        perturb_reward = perturbance
        if self.result_type == 'list' and perturbance >= reduction_threshold:
            perturb_reward = reduction_threshold + reduction_factor * (perturbance - reduction_threshold)
        
        # similar_reward = self.similarity
        # if self.similarity >= self.similarity_threshold:
        #     similar_reward = 1
        
        reward = perturb_reward * similarity
        improvement = True
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_perturbance = perturbance
            self.best_similarity = similarity
            self.perturb_image = copy.deepcopy(perturb_test)
        else:
            reward = 0
            improvement = False

        if improvement:
            if self.render_level > 0:
                print_perturb = np.format_float_scientific(self.best_perturbance, 3)
                print_similar = round(self.best_similarity * 100, 1)
                print(f"Perturbance: {print_perturb} - Similarity: {print_similar}%")
            if self.render_level > 1:
                self.render()

        if (self.result_type == 'list' and np.argmax(results) == self.new_class) or (self.result_type == 'object' and results == self.new_class):
            if self.best_similarity >= self.similarity_threshold:
                fake_image_file = f"Fake{self.image_file}"
                cv2.imwrite(fake_image_file, self.perturb_image)
                print(f"Successful perturb! Image saved at {fake_image_file}")
                exit()
            elif improvement and self.checkpoint_level > 0:
                fake_image_file = f"Checkpoint{self.image_file}"
                if self.checkpoint_file is not None:
                    fake_image_file = self.checkpoint_file
                cv2.imwrite(fake_image_file, self.perturb_image)
                print(f"Checkpoint image saved at {fake_image_file}")
        
        elif improvement and self.checkpoint_level > 1:
            checkpoint_image_file = f"Checkpoint{self.image_file}"
            if self.checkpoint_file is not None:
                checkpoint_image_file = self.checkpoint_file        
            cv2.imwrite(checkpoint_image_file, self.perturb_image)
            print(f"Checkpoint image saved at {checkpoint_image_file}")

        return self.perturb_image, reward, False, {}

    def reset(self):
        if self.checkpoint_image is None:
            self.perturb_image = copy.deepcopy(self.original_image)
        else:
            self.perturb_image = copy.deepcopy(self.checkpoint_image)
        
        self.best_reward = 0
        self.best_perturbance = 0
        self.best_similarity = 0

        return self.perturb_image

    def get_best_reward(self):
        return self.best_reward
    
    def render(self):
        self.plot_original()
        self.plot_morph()
        plt.pause(0.01)

    def plot_original(self):
        self.plot1.grid(False)
        self.plot1.set_xticks([])
        self.plot1.set_yticks([])

        self.plot1.imshow(self.original_image, cmap='gray')

        color = 'red'
        if self.best_similarity >= self.similarity_threshold:
            color = 'green'
        
        self.plot1.set_xlabel("Similarity = {}".format(round(self.best_similarity * 100, 1), color=color))

    def plot_morph(self):
        self.plot2.grid(False)
        self.plot2.set_xticks([])
        self.plot2.set_yticks([])

        self.plot2.imshow(self.perturb_image, cmap='gray')

        self.plot2.set_xlabel("Perturbance={:2.0f}".format(np.format_float_scientific(self.best_perturbance, 3)))

