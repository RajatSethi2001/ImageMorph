import cv2
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time

from gym.spaces import Box
from IPython import display
from tensorflow.keras import datasets, layers, models

class MorphEnv(gym.Env):
    def __init__(self, model, img, img_label, fake_label):
        self.observation_space = Box(low=0, high=255, shape=img.shape, dtype=np.uint8)
        self.action_space = Box(low=0, high=255, shape=img.shape, dtype=np.uint8)
        # self.action_space = Box(low=[0, 0, 0], high=[img.shape[0], img.shape[1], 1.0], shape=(3,), dtype=np.float32)
        
        self.img_original = img
        self.img_label = img_label

        self.img_perturb = img
        self.model = model
        self.outcome = fake_label
        self.results = None

    def step(self, action):
        self.img_perturb += np.uint8(np.round(action))
        self.img_perturb = np.clip(self.img_perturb, 0, 255)

        img_input = cv2.resize(self.img_perturb, (28, 28)).reshape(1, 28, 28, 1) / 255
        self.results = self.model.predict(img_input, verbose=0)[0]
        perturbance = self.results[self.outcome]

        self.render()

        euclid_distance = 0
        shape = self.img_perturb.shape
        for row in range(shape[0]):
            for col in range(shape[1]):
                for pixel in range(shape[2]):
                    euclid_distance += ((self.img_perturb[row][col][pixel] - self.img_original[row][col][pixel]) / 255.0) ** 2
        
        similarity_score = 1 - math.sqrt(euclid_distance / (shape[0] * shape[1]))

        done = False
        if np.argmax(self.results) == self.outcome and similarity_score >= 0.9:
            cv2.imwrite("Fake.png", self.img_perturb)
            input("Successful perturb! Press anywhere to continue")
            done = True
        
        return self.img_perturb, perturbance * similarity_score, done, {}

    def reset(self):
        self.img_perturb = self.img_original
        return self.img_original

    def render(self):
        img_display = self.img_perturb
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        self.plot_image(self.results, self.img_label, self.outcome, img_display)
        plt.subplot(1,2,2)
        self.plot_value_array(self.results, self.img_label, self.outcome)
        plt.show()
        plt.pause(0.01)
        display.clear_output(wait=True)

    def plot_image(self, predictions_array, true_label, fake_label, img):
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(img, cmap=plt.cm.binary)

        predicted_label = np.argmax(predictions_array)
        if predicted_label == true_label:
            color = 'blue'
        elif predicted_label == fake_label:
            color = 'green'
        else:
            color = 'red'

        plt.xlabel("{} {:2.0f}% (true={} fake={})".format(predicted_label,
                                        100*np.max(predictions_array),
                                        true_label,
                                        fake_label),
                                        color=color)

    def plot_value_array(self, predictions_array, true_label, fake_label):
        plt.grid(False)
        plt.xticks(range(len(predictions_array)))
        plt.yticks([])
        thisplot = plt.bar(range(len(predictions_array)), predictions_array, color="#777777")
        plt.ylim([0, 1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_color('blue')
        thisplot[fake_label].set_color('green')

