import copy
import cv2
import gym
import math
import matplotlib.pyplot as plt
import numpy as np

from gym.spaces import Box
from os.path import exists

class MorphEnv(gym.Env):
    def __init__(self, predict_wrapper, image_file, grayscale, victim_data, new_class, similarity=0.7, render_level=0, checkpoint_level=0, checkpoint_file=None, graph_file=None):
        self.predict_wrapper = predict_wrapper
        self.victim_data = victim_data
        self.image_file = image_file
        self.checkpoint_file = checkpoint_file
        self.checkpoint_image = None
        self.grayscale = grayscale
        
        #Grayscale images have to be opened independently from RGB images.
        if self.grayscale:
            self.original_image = cv2.imread(self.image_file, 0)
            #Grayscale images are reshaped with an extra dimension for pixels, which makes processing slightly easier.
            self.original_image = self.original_image.reshape(self.original_image.shape + (1,))
            if self.checkpoint_file is not None and exists(self.checkpoint_file):
                self.checkpoint_image = cv2.imread(self.checkpoint_file, 0)
                self.checkpoint_image = self.checkpoint_image.reshape(self.checkpoint_image.shape + (1,))
        #If not grayscale, image is a 3D RGB image
        else:
            self.original_image = cv2.imread(self.image_file)
            if self.checkpoint_file is not None and exists(self.checkpoint_file):
                self.checkpoint_image = cv2.imread(self.checkpoint_file)
        
        #If no checkpoint is given, start from the original.
        if self.checkpoint_image is None:
            self.perturb_image = copy.deepcopy(self.original_image)
        else:
            self.perturb_image = copy.deepcopy(self.checkpoint_image)
        
        #Shape of the image.
        self.shape = self.original_image.shape

        #The observation space is the space of all values that can be provided as input.
        #In this case, the agent should receive a matrix of pixels.
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

        self.action_space = Box(low=0, high=1, shape=(4,), dtype=np.float32)

        #Get current results to determine the type of output from classifier.
        #If the classifier returns a list, the agent will try to maximize the index determined from new_class
        #If the classifier returns something else, the agent will try to make new_class occur.
        results = self.predict_wrapper(self.original_image, self.victim_data)
        self.result_type = "object"
        if isinstance(results, list) or isinstance(results, np.ndarray):
            self.result_type = "list"

        self.new_class = new_class
        #Get the orignal class (which will be used for untargeted attacks and verifying final results)
        if self.result_type == "list":
            self.original_class = np.argmax(results)
        else:
            self.original_class = results

        #Agent will store the image with the highest reward.
        #That "best reward" image will also have its perturbance and similarity recorded for future reference.
        self.best_reward = 0
        self.best_perturbance = 0
        self.best_similarity = 0

        #Stores data for graphing
        self.graph_file = None
        if graph_file is not None:
            self.graph_file = graph_file
            self.perturb_scores = []
            self.similar_scores = []
            self.reward_scores = []

        #What similarity is required for a successful morph.
        self.similarity_threshold = similarity

        #Determines the verbosity of rendering progress and saving checkpoints.
        self.render_level = render_level
        self.checkpoint_level = checkpoint_level

        #If pyplot is active, create the figures/subplots.
        if self.render_level > 1:
            plt.ion()
            self.figure = plt.figure(figsize=(6,3))
            self.plot1 = self.figure.add_subplot(1,2,1)
            self.plot2 = self.figure.add_subplot(1,2,2)
        
        self.timesteps = 0

    #When the agent receives an action, it will act upon it and determine how good/bad the action was.
    def step(self, action):
        self.timesteps += 1
        #Create a copy of the current perturbed image, then sample the action on this copy.
        perturb_test = copy.deepcopy(self.perturb_image)
        
        row = np.uint8(np.round(action[0] * (self.shape[0] - 1)))
        col = np.uint8(np.round(action[1] * (self.shape[1] - 1)))
        color = np.uint8(np.round(action[2] * (self.shape[2] - 1)))
        pixel_change = np.uint8(np.round(action[3] * 255))

        perturb_test[row][col][color] = pixel_change

        #Get the results from the classifier.
        results = self.predict_wrapper(perturb_test, self.victim_data)
        successful_perturb = False
        if self.new_class is None:
            if self.result_type == "list" and np.argmax(results) != self.original_class:
                successful_perturb = True
            elif self.result_type == "object" and results != self.original_class:
                successful_perturb = True
        else:
            if self.result_type == "list" and np.argmax(results) == self.new_class:
                successful_perturb = True
            elif self.result_type == "object" and results == self.new_class:
                successful_perturb = True

        #If the attack is untargeted, perturbance is determined by how low the original class score is. 
        if self.new_class is None:
            #If the victim returns a list, the perturbance is how far the original score is from 1.
            if self.result_type == "list":
                perturbance = sum(results) - results[self.original_class]
            #If its not a list, the perturbance is 0 if the result is still the original class, 1 if otherwise.
            else:
                if results == self.original_class:
                    perturbance = 0
                else:
                    perturbance = 1
        #If the attack is targeted, perturbance is determined by how high the new class score is. 
        else:
            #If the victim returns a list, the perturbance is the value found at the desired index (specified by new_class)
            if self.result_type == "list":
                perturbance = results[self.new_class]
            #If its not a list, the perturbance is 1 if the result is the new class, 0 if otherwise.
            else:
                if results == self.new_class:
                    perturbance = 1
                else:
                    perturbance = 0

            #Similarity is measured by the distance between the original image and the perturbed image.
        euclid_distance = 0
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                for color in range(self.shape[2]):
                    # Find the difference in pixels, normalize the value, then square it.
                    pixel_distance = ((int(perturb_test[row][col][color]) - int(self.original_image[row][col][color])) / 255.0) ** 2
                    euclid_distance += pixel_distance
        
        # Renormalize the final result, take the square root, then subtract that value from 1 to find similarity.
        similarity = 1 - math.sqrt(euclid_distance / math.prod(self.shape))
        
        reward = perturbance * similarity
        
        #If this perturbed image has a higher reward than the current best, then this image is the new best.
        improvement = True
        if reward >= self.best_reward:
            if reward == self.best_reward:
                improvement = False
            self.best_reward = reward
            self.best_perturbance = perturbance
            self.best_similarity = similarity
            self.perturb_image = copy.deepcopy(perturb_test)
        #Else, there is no reward
        else:
            reward = 0
            improvement = False

        #If there is improvement, activate the render function (depending on the level provided).
        if improvement:
            #If render_level > 0, print out perturbance and similarity of the best image.
            if self.render_level > 0:
                print_perturb = np.format_float_scientific(self.best_perturbance, 3)
                print_similar = round(self.best_similarity * 100, 1)
                print(f"Perturbance: {print_perturb} - Similarity: {print_similar}%")

            #If render_level > 1, update the pyplot.
            if self.render_level > 1:
                self.render()

        if self.graph_file is not None:
            self.perturb_scores.append(self.best_perturbance)
            self.similar_scores.append(self.best_similarity)
            self.reward_scores.append(self.best_reward)
            if improvement:
                timestep_list = list(range(self.timesteps))
                plt.plot(timestep_list, self.perturb_scores, label="Perturbance")
                plt.plot(timestep_list, self.similar_scores, label="Similarity")
                plt.plot(timestep_list, self.reward_scores, label="Best Reward")
                plt.xlabel("Timesteps")
                plt.ylabel("Score [0-1]")
                plt.legend()
                plt.savefig(self.graph_file)
                plt.close()

        #If the image has been successfully misclassified, save the result. 
        if successful_perturb:
            #If the image has a higher similarity than the threshold, save and exit.
            if self.best_similarity >= self.similarity_threshold:
                fake_image_file = f"Fake{self.image_file}"
                cv2.imwrite(fake_image_file, self.perturb_image)
                print(f"Successful perturb! Image saved at {fake_image_file}")
                if self.result_type == "list":
                    print(f"Original Index: {self.original_class}")
                    print(f"New Index: {np.argmax(results)}")
                else:
                    print(f"Original Class: {self.original_class}")
                    print(f"New Class: {results}")
                exit()
            
            #If the image does not have a high enough similarity, save the checkpoint and continue (if checkpoint_level is 1 or higher).
            elif improvement and self.checkpoint_level > 0:
                fake_image_file = f"Checkpoint{self.image_file}"
                if self.checkpoint_file is not None:
                    fake_image_file = self.checkpoint_file
                cv2.imwrite(fake_image_file, self.perturb_image)
                print(f"Checkpoint image saved at {fake_image_file}")
        
        #If the image has improved at all, save it (if checkpoint_level is 2 or higher)
        elif improvement and self.checkpoint_level > 1:
            checkpoint_image_file = f"Checkpoint{self.image_file}"
            if self.checkpoint_file is not None:
                checkpoint_image_file = self.checkpoint_file        
            cv2.imwrite(checkpoint_image_file, self.perturb_image)
            print(f"Checkpoint image saved at {checkpoint_image_file}")

        return self.perturb_image, reward, False, {}

    #Reset the parameters (Only called during initialization (and sometimes evaluation))
    def reset(self):
        #Set the image back to what it was called
        if self.checkpoint_image is None:
            self.perturb_image = copy.deepcopy(self.original_image)
        else:
            self.perturb_image = copy.deepcopy(self.checkpoint_image)
        
        #Reset the best image statistics.
        self.best_reward = 0
        self.best_perturbance = 0
        self.best_similarity = 0

        #Reset the best image statistic scores.
        if self.graph_file is not None:
            self.perturb_scores = []
            self.similar_scores = []
            self.reward_scores = []

        return self.perturb_image

    #Return the best reward (Used by Optuna.py)
    def get_best_reward(self):
        return self.best_reward
    
    def get_best_perturbance(self):
        return self.best_perturbance
    
    def get_best_similarity(self):
        return self.best_similarity

    def softmax(self, results):
        return np.exp(results) / np.sum(np.exp(results))
    
    #If render_level > 1, display the original and perturbed images in pyplot.
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

