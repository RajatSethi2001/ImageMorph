import copy
import cv2
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random

from gym.spaces import Box
from os.path import exists

class MorphEnv(gym.Env):
    def __init__(self, predict_wrapper, victim_data, attack_array, array_range, new_class, similarity=0.7, render_level=0, checkpoint_level=0, checkpoint=None, graph_file=None):
        self.predict_wrapper = predict_wrapper
        self.victim_data = victim_data
        self.attack_array = attack_array
        self.array_range = array_range
        self.range = array_range[1] - array_range[0]
        self.dtype = attack_array.dtype
        self.checkpoint = checkpoint
        
        #Shape of the array.
        self.shape = self.attack_array.shape
        self.reset_set = set()
        #If no checkpoint is given, start from the original.
        if self.checkpoint is None:
            self.perturb_array = np.copy(self.attack_array)
        else:
            self.perturb_array = np.copy(self.checkpoint)
            for idx, _ in np.ndenumerate(self.perturb_array):
                if self.perturb_array[idx] != self.attack_array[idx]:
                    self.reset_set.add(idx)

        #The observation space is the space of all values that can be provided as input.
        #In this case, the agent should receive a matrix of pixels.
        self.observation_space = Box(low=self.array_range[0], high=self.array_range[1], shape=self.shape, dtype=self.dtype)

        num_actions = len(self.shape) + 2
        self.action_space = Box(low=0, high=1, shape=(num_actions,), dtype=np.float32)
        

        #Get current results to determine the type of output from classifier.
        #If the classifier returns a list, the agent will try to maximize the index determined from new_class
        #If the classifier returns something else, the agent will try to make new_class occur.
        results = self.predict_wrapper(self.attack_array, self.victim_data)
        self.result_type = "object"
        if isinstance(results, list) or isinstance(results, np.ndarray):
            self.result_type = "list"

        self.new_class = new_class
        #Get the orignal class (which will be used for untargeted attacks and verifying final results)
        if self.result_type == "list":
            self.original_class = np.argmax(results)
        else:
            self.original_class = results

        #Agent will store the array with the highest reward.
        #That "best reward" array will also have its perturbance and similarity recorded for future reference.
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
        
        self.timesteps = 0

    #When the agent receives an action, it will act upon it and determine how good/bad the action was.
    def step(self, action):
        self.timesteps += 1
        #Create a copy of the current perturbed array, then sample the action on this copy.
        perturb_test = np.copy(self.perturb_array)
        reset_test = np.copy(self.perturb_array)

        location = []
        for idx, dim in np.ndenumerate(action[0:len(self.shape)]):
            scaled_dim = int(np.round(dim * (self.shape[idx[0]] - 1)))
            location.append(scaled_dim)
        location = tuple(location)
        reset_location = location
        
        unit_change = np.round(action[len(action) - 2] * self.range + self.array_range[0]).astype(self.dtype)
        reset_strength = action[len(action) - 1]

        perturb_test[location] = unit_change
        if len(self.reset_set) > 0:
            reset_tickets = {}
            ticket_floor = 0
            for perturbed_location in self.reset_set:
                contrast = abs(int(reset_test[perturbed_location]) - int(self.attack_array[perturbed_location]))
                ticket_min = ticket_floor
                ticket_max = ticket_min + contrast - 1
                ticket_floor = ticket_max + 1
                reset_tickets[perturbed_location] = (ticket_min, ticket_max)
            
            ticket = random.randint(0, ticket_floor - 1)

            reset_location = location
            for perturbed_location in self.reset_set:
                location_tickets = reset_tickets[perturbed_location]
                if ticket >= location_tickets[0] and ticket <= location_tickets[1]:
                    reset_location = perturbed_location
                    break

            pixel_delta = (float(self.attack_array[reset_location]) - float(reset_test[reset_location])) * reset_strength
            reset_test[reset_location] = (reset_test[reset_location] + pixel_delta).astype(self.dtype)

        #Get the results from the classifier.
        perturb_data = self.collect_diagnostics(perturb_test)
        reset_data = self.collect_diagnostics(reset_test)
        
        perturb_action = True
        if perturb_data[2] >= reset_data[2]:
            perturbance, similarity, reward, results = perturb_data
        else:
            perturb_action = False
            perturbance, similarity, reward, results = reset_data
            location = reset_location

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
        
        #If this perturbed array has a higher reward than the current best, then this array is the new best.
        improvement = True
        if reward >= self.best_reward:
            if reward == self.best_reward:
                improvement = False
            self.best_reward = reward
            self.best_perturbance = perturbance
            self.best_similarity = similarity
            if perturb_action:
                self.perturb_array = copy.deepcopy(perturb_test)
            else:
                self.perturb_array = copy.deepcopy(reset_test)

        #Else, there is no reward
        else:
            reward = 0
            improvement = False

        if self.perturb_array[location] != self.attack_array[location]:
            if location not in self.reset_set:
                self.reset_set.add(location)
        elif location in self.reset_set:
            self.reset_set.remove(location)

        #If there is improvement, activate the render function (depending on the level provided).
        if improvement:
            #If render_level > 0, print out perturbance and similarity of the best array.
            if self.render_level:
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

        #If the array has been successfully misclassified, save the result. 
        if successful_perturb:
            #If the array has a higher similarity than the threshold, save and exit.
            if self.best_similarity >= self.similarity_threshold:
                fake_array_file = "FakeArray.npy"
                np.save(fake_array_file, self.perturb_array)
                print(f"Successful perturb! Array saved at {fake_array_file}")
                if self.result_type == "list":
                    print(f"Original Index: {self.original_class}")
                    print(f"New Index: {np.argmax(results)}")
                else:
                    print(f"Original Class: {self.original_class}")
                    print(f"New Class: {results}")
                exit()
            
            #If the array does not have a high enough similarity, save the checkpoint and continue (if checkpoint_level is 1 or higher).
            elif improvement and self.checkpoint_level > 0:
                fake_array_file = "Checkpoint.npy"
                np.save(fake_array_file, self.perturb_array)
                print(f"Checkpoint array saved at {fake_array_file}")
        
        #If the array has improved at all, save it (if checkpoint_level is 2 or higher)
        elif improvement and self.checkpoint_level > 1:
            checkpoint_array_file = "Checkpoint.npy"      
            np.save(checkpoint_array_file, self.perturb_array)
            print(f"Checkpoint array saved at {checkpoint_array_file}")

        return self.perturb_array, reward, False, {}
    
    def collect_diagnostics(self, perturb_array):
        results = self.predict_wrapper(perturb_array, self.victim_data)

        #If the attack is untargeted, perturbance is determined by how low the original class score is. 
        if self.new_class is None:
            #If the victim returns a list, the perturbance is the sum of all other values.
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

        #Similarity is measured by the distance between the original array and the perturbed array.
        euclid_distance = 0
        for idx, _ in np.ndenumerate(perturb_array):
            # Find the difference in pixels, normalize the value, then square it.
            pixel_distance = ((float(perturb_array[idx]) - float(self.attack_array[idx])) / self.range) ** 2
            euclid_distance += pixel_distance
        
        # Renormalize the final result, take the square root, then subtract that value from 1 to find similarity.
        similarity = 1 - math.sqrt(euclid_distance / math.prod(self.shape))

        reward = perturbance * similarity

        return (perturbance, similarity, reward, results)

    #Reset the parameters (Only called during initialization (and sometimes evaluation))
    def reset(self):
        #Set the array back to what it was called
        self.reset_set = set()
        if self.checkpoint is None:
            self.perturb_array = copy.deepcopy(self.attack_array)
        else:
            self.perturb_array = copy.deepcopy(self.checkpoint)
            for idx, _ in np.ndenumerate(self.perturb_array):
                if self.perturb_array[idx] != self.attack_array[idx]:
                    self.reset_set.add(idx)

        #Reset the best array statistics.
        self.best_reward = 0
        self.best_perturbance = 0
        self.best_similarity = 0

        #Reset the best array statistic scores.
        if self.graph_file is not None:
            self.perturb_scores = []
            self.similar_scores = []
            self.reward_scores = []

        return self.perturb_array

    #Return the best reward (Used by Optuna.py)
    def get_best_reward(self):
        return self.best_reward
    
    def get_best_perturbance(self):
        return self.best_perturbance
    
    def get_best_similarity(self):
        return self.best_similarity
    
    #If render_level > 1, display the original and perturbed arrays in pyplot.
    def render(self):
        print_perturb = np.format_float_scientific(self.best_perturbance, 3)
        print_similar = round(self.best_similarity * 100, 3)
        print(f"Perturbance: {print_perturb} - Similarity: {print_similar}%")

