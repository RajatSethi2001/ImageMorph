import copy
import cv2
import gym
import math
import matplotlib.pyplot as plt
import numpy as np

from gym.spaces import Box

class MorphEnv(gym.Env):
    def __init__(self, model, image, image_file, classes, new_class, action=0, similarity=0.9, scale_image=True, render_interval=0, save_interval=1000):
        self.observation_space = Box(low=0, high=255, shape=image.shape, dtype=np.uint8)
        
        self.action = action
        if action == 0:
            self.action_space = Box(low=np.array([0, 0, 0, -1]), high=np.array([1, 1, 1, 1]), shape=(4,), dtype=np.float32)
        elif action == 1:
            self.action_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)
        elif action == 2:
            self.action_space = Box(low=-1, high=1, shape=image.shape, dtype=np.float32)
        elif action == 3:
            self.action_space = Box(low=0, high=1, shape=image.shape, dtype=np.float32)
        else:
            raise Exception("Action must be an integer between 0-3")
        
        self.original_image = copy.deepcopy(image)
        self.perturb_image = copy.deepcopy(image)

        self.shape = image.shape
        self.image_file = image_file

        self.model = model
        self.classes = classes
        model_config = self.model.get_config()
        self.dimensions = model_config["layers"][0]["config"]["batch_input_shape"][1::]
        self.scale_image = scale_image
        results = self.predict()
        
        self.original_label = self.classes[np.argmax(results)]
        self.original_index = np.argmax(results)
        self.new_label = new_class
        self.new_index = self.classes.index(new_class)

        self.similarity_threshold = similarity

        self.old_perturbance = results[self.new_index]
        self.old_similarity = 1.0
        
        self.current_results = None
        self.current_perturbance = None
        self.current_similarity = None

        plt.ion()
        self.figure = plt.figure(figsize=(9,3))
        self.plot1 = self.figure.add_subplot(1,3,1)
        self.plot2 = self.figure.add_subplot(1,3,2)
        self.plot3 = self.figure.add_subplot(1,3,3)

        self.steps = 0
        self.render_interval = render_interval
        self.save_interval = save_interval

    def step(self, action):
        self.steps += 1

        if self.action == 0:
            row = np.uint8(np.round(action[0] * (self.shape[0] - 1)))
            col = np.uint8(np.round(action[1] * (self.shape[1] - 1)))
            color = np.uint8(np.round(action[2] * (self.shape[2] - 1)))
            pixel_change = np.uint8(np.round(action[3] * 255))

            if pixel_change < 0:
                #check for overflow
                if (self.perturb_image[row][col][color] - pixel_change > self.perturb_image[row][col][color]):
                    self.perturb_image[row][col][color] = 0
                else:
                    self.perturb_image[row][col][color] -= pixel_change
            else:
                if (self.perturb_image[row][col][color] + pixel_change < self.perturb_image[row][col][color]):
                    self.perturb_image[row][col][color] = 255
                else:
                    self.perturb_image[row][col][color] += pixel_change
        
        elif self.action == 1:
            row = np.uint8(np.round(action[0] * (self.shape[0] - 1)))
            col = np.uint8(np.round(action[1] * (self.shape[1] - 1)))
            color = np.uint8(np.round(action[2] * (self.shape[2] - 1)))
            pixel_change = np.uint8(np.round(action[3] * 255))

            self.perturb_image[row][col][color] = pixel_change

        elif self.action == 2:
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    for color in range(self.shape[2]):
                        pixel_change = np.uint8(np.round(action[row][col][color] * 255))

                        if pixel_change < 0:
                            #check for overflow
                            if (self.perturb_image[row][col][color] - pixel_change > self.perturb_image[row][col][color]):
                                self.perturb_image[row][col][color] = 0
                            else:
                                self.perturb_image[row][col][color] -= pixel_change
                        else:
                            if (self.perturb_image[row][col][color] + pixel_change < self.perturb_image[row][col][color]):
                                self.perturb_image[row][col][color] = 255
                            else:
                                self.perturb_image[row][col][color] += pixel_change

        elif self.action == 3:
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    for color in range(self.shape[2]):
                        pixel_change = np.uint8(np.round(action[row][col][color] * 255.0))
                        self.perturb_image[row][col][color] = pixel_change

        self.current_results = self.predict()
        self.current_perturbance = self.current_results[self.new_index]

        euclid_distance = 0
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                for color in range(self.shape[2]):
                    pixel_distance = ((int(self.perturb_image[row][col][color]) - int(self.original_image[row][col][color])) / 255.0) ** 2
                    euclid_distance += pixel_distance
        
        self.current_similarity = 1 - math.sqrt(euclid_distance / math.prod(self.shape))
        
        # delta_perturbance = self.current_perturbance - self.old_perturbance
        # delta_similarity = self.current_similarity - self.old_similarity
        # reward = delta_perturbance * delta_similarity

        reward = self.current_perturbance * self.current_similarity
        done = self.current_similarity < self.similarity_threshold

        if self.render_interval > 0 and self.steps % self.render_interval == 0:
            # self.render()
            print_perturb = np.format_float_scientific(self.current_perturbance, 3)
            print_similar = round(self.current_similarity * 100, 1)
            print(f"Perturbance: {print_perturb} - Similarity: {print_similar}%")

        if np.argmax(self.current_results) == self.new_index and self.current_similarity >= self.similarity_threshold:
            fake_image = cv2.resize(self.perturb_image, (self.dimensions[0], self.dimensions[1]))
            cv2.imwrite(f"Fake{self.image_file}", fake_image)
            input("Successful perturb! Press anywhere to continue")
            exit()
        
        self.old_perturbance = self.current_perturbance
        self.old_similarity = self.current_similarity        
        return self.perturb_image, reward, done, {}

    def reset(self):
        self.perturb_image = copy.deepcopy(self.original_image)
        results = self.predict()
        self.old_perturbance = results[self.new_index]
        self.old_similarity = 1.0
        
        self.current_results = None
        self.current_perturbance = None
        self.current_similarity = None
        return self.original_image

    def render(self):
        self.plot3.clear()
        self.plot_original()
        self.plot_morph()
        self.plot_value_array()
        plt.pause(0.01)

    def plot_original(self):
        self.plot1.grid(False)
        self.plot1.set_xticks([])
        self.plot1.set_yticks([])

        self.plot1.imshow(self.original_image, cmap='gray')

        color = 'red'
        if self.current_similarity >= self.similarity_threshold:
            color = 'green'
        
        self.plot1.set_xlabel("Similarity = {}".format(round(self.current_similarity * 100, 1), color=color))

    def plot_morph(self):
        self.plot2.grid(False)
        self.plot2.set_xticks([])
        self.plot2.set_yticks([])

        self.plot2.imshow(self.perturb_image, cmap='gray')

        predicted_label = self.classes[np.argmax(self.current_results)]
        if predicted_label == self.original_label:
            color = 'blue'
        elif predicted_label == self.new_label:
            color = 'green'
        else:
            color = 'red'

        self.plot2.set_xlabel("{} {:2.0f}% (true={} fake={})".format(predicted_label,
                                        100*np.max(self.current_results),
                                        self.original_label,
                                        self.new_label),
                                        color=color)

    def plot_value_array(self):
        self.plot3.grid(False)
        self.plot3.set_xticks(range(len(self.current_results)), self.classes)
        self.plot3.set_yticks([])
        thisplot = self.plot3.bar(range(len(self.current_results)), self.current_results, color="#777777")
        self.plot3.set_ylim([0, 1])
        predicted_label = self.classes[np.argmax(self.current_results)]

        thisplot[predicted_label].set_color('red')
        thisplot[self.original_label].set_color('blue')
        thisplot[self.new_label].set_color('green')
    
    def predict(self):
        image_input = cv2.resize(self.perturb_image, (self.dimensions[0], self.dimensions[1]))
        image_input = image_input.reshape((1,)+self.dimensions)
        if self.scale_image:
            image_input = image_input / 255.0
        results = self.model.predict(image_input, verbose=0)[0]
        return results

