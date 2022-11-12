import copy
import cv2
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
np.seterr(all='raise')

from gym.spaces import Box
from os.path import exists

class MorphEnv(gym.Env):
    def __init__(self, predict_wrapper, image_file, grayscale, victim_data, new_class, action=0, similarity=0.7, render_level=0, render_interval=0, save_interval=0, checkpoint_file=None):
        self.predict_wrapper = predict_wrapper
        self.victim_data = victim_data
        self.image_file = image_file
        self.checkpoint_file = checkpoint_file
        self.checkpoint_image = None
        self.grayscale = grayscale

        if self.grayscale:
            self.original_image = cv2.imread(self.image_file, 0)
            if self.checkpoint_file is not None and exists(self.checkpoint_file):
                self.checkpoint_image = cv2.imread(self.checkpoint_file, 0)
        #Else, image is a 3D RGB image
        else:
            self.original_image = cv2.imread(self.image_file)
            if self.checkpoint_file is not None and exists(self.checkpoint_file):
                self.checkpoint_image = cv2.imread(self.checkpoint_file)
        
        self.dim_height = self.original_image.shape[0]
        self.dim_width = self.original_image.shape[1]

        #Determine the smaller dimension
        min_length = min(self.dim_height, self.dim_width)
        #Scale images (sb3 requires images to be > 36x36)
        if min_length < 36:
            scale = 36 / min_length
            new_height = int(self.dim_height * scale)
            new_width = int(self.dim_width * scale)
            self.original_image = cv2.resize(self.original_image, (new_height, new_width))
            if self.checkpoint_file is not None and exists(self.checkpoint_file): 
                self.checkpoint_image = cv2.resize(self.checkpoint_image, (new_height, new_width))

        #If greyscale, add an extra dimension (makes processing easier)
        if self.grayscale:
            self.original_image = self.original_image.reshape(self.original_image.shape + (1,))
            if self.checkpoint_file is not None and exists(self.checkpoint_file):
                self.checkpoint_image = self.checkpoint_image.reshape(self.checkpoint_image.shape + (1,))

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
        elif action == 2:
            self.action_space = Box(low=-1, high=1, shape=self.shape, dtype=np.float32)
        elif action == 3:
            self.action_space = Box(low=0, high=1, shape=self.shape, dtype=np.float32)
        else:
            raise Exception("Action must be an integer between 0-3")

        image_input = cv2.resize(self.perturb_image, (self.dim_height, self.dim_width))
        if self.grayscale:
            image_input = image_input.reshape((self.dim_height, self.dim_width, 1))
        results = self.predict_wrapper(image_input, self.victim_data)
        self.result_type = "object"
        if isinstance(results, list) or isinstance(results, np.ndarray):
            self.result_type = "list"

        self.new_class = new_class

        self.best_reward = 0
        self.best_perturbance = 0
        self.best_similarity = 0
        self.similarity_threshold = similarity

        self.results = None
        # self.perturbance = None
        # self.similarity = None

        self.render_level = render_level

        plt.ion()
        self.figure = plt.figure(figsize=(6,3))
        self.plot1 = self.figure.add_subplot(1,2,1)
        self.plot2 = self.figure.add_subplot(1,2,2)
        # self.plot3 = self.figure.add_subplot(1,3,3)

        self.steps = 0
        self.render_interval = render_interval
        self.save_interval = save_interval

    def step(self, action):
        self.steps += 1
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

        elif self.action == 2:
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    for color in range(self.shape[2]):
                        pixel_change = np.uint8(np.round(action[row][col][color] * 255))

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

        elif self.action == 3:
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    for color in range(self.shape[2]):
                        pixel_change = np.uint8(np.round(action[row][col][color] * 255.0))
                        perturb_test[row][col][color] = pixel_change

        image_input = cv2.resize(perturb_test, (self.dim_height, self.dim_width))
        if self.grayscale:
            image_input = image_input.reshape((self.dim_height, self.dim_width, 1))
        self.results = self.predict_wrapper(image_input, self.victim_data)
        if self.result_type == "list":
            perturbance = self.results[self.new_class]
        else:
            if self.results == self.new_class:
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

        # perturb_reward = self.perturbance
        # if self.result_type == 'list' and np.argmax(self.results) == self.new_class:
        #     perturb_reward = 1
        
        # similar_reward = self.similarity
        # if self.similarity >= self.similarity_threshold:
        #     similar_reward = 1
        
        reward = perturbance * similarity
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_perturbance = perturbance
            self.best_similarity = similarity
            self.perturb_image = copy.deepcopy(perturb_test)
        else:
            reward = 0

        if self.render_interval > 0 and self.steps % self.render_interval == 0:
            if self.render_level > 0:
                print_perturb = np.format_float_scientific(self.best_perturbance, 3)
                print_similar = round(self.best_similarity * 100, 1)
                print(f"Perturbance: {print_perturb} - Similarity: {print_similar}%")
            if self.render_level > 1:
                self.render()
            
        if self.save_interval > 0 and self.steps % self.save_interval == 0:
            checkpoint_image = cv2.resize(self.perturb_image, (self.dim_height, self.dim_height))
            checkpoint_image_file = f"Checkpoint{self.image_file}"
            if self.checkpoint_file is not None:
                checkpoint_image_file = self.checkpoint_file        
            cv2.imwrite(checkpoint_image_file, checkpoint_image)
            print(f"Checkpoint image saved at {checkpoint_image_file}")

        if (self.result_type == 'list' and np.argmax(self.results) == self.new_class) or (self.result_type == 'object' and self.results == self.new_class):
            fake_image = cv2.resize(self.perturb_image, (self.dim_height, self.dim_width))
            if self.best_similarity >= self.similarity_threshold:
                fake_image_file = f"Fake{self.image_file}"
                cv2.imwrite(fake_image_file, fake_image)
                print(f"Successful perturb! Image saved at {fake_image_file}")
                exit()
            else:
                fake_image_file = f"SemiFake{self.image_file}"
                cv2.imwrite(fake_image_file, fake_image)
                print(f"Semi-successful perturb! Not enough similarity, but image saved at {fake_image_file}")

        return self.perturb_image, reward, False, {}

    def reset(self):
        if self.checkpoint_image is None:
            self.perturb_image = copy.deepcopy(self.original_image)
        else:
            self.perturb_image = copy.deepcopy(self.checkpoint_image)
        
        self.best_reward = 0
        self.best_perturbance = 0
        self.best_similarity = 0

        self.results = None
        # self.perturbance = None
        # self.similarity = None
        return self.perturb_image

    def render(self):
        self.plot_original()
        self.plot_morph()
        # self.plot3.clear()
        # self.plot_value_array()
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

    # def plot_value_array(self):
    #     self.plot3.grid(False)
    #     self.plot3.set_xticks(range(len(self.results)), self.classes)
    #     self.plot3.set_yticks([])
    #     thisplot = self.plot3.bar(range(len(self.results)), self.results, color="#777777")
    #     self.plot3.set_ylim([0, 1])
    #     predicted_label = self.classes[np.argmax(self.results)]

    #     thisplot[predicted_label].set_color('red')
    #     thisplot[self.original_label].set_color('blue')
    #     thisplot[self.new_label].set_color('green')

