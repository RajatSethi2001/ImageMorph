import copy
import cv2
import gc
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from gym.spaces import Box
from os.path import exists

class GaslightEnv(gym.Env):
    def __init__(self, predict_wrapper, victim_data, attack_array, array_range, new_class, similarity=0.8, result_file="ResultFile.npy", render_interval=0, save_interval=0, checkpoint_file=None, graph_file=None):
        #Wrapper that will take the perturbed array, have the victim make a prediction, then return the results.
        self.predict_wrapper = predict_wrapper

        #A collection of data that needs to be passed to predict_wrapper. Should include the model information and any preprocessing tools.
        self.victim_data = victim_data

        #The original array to get perturbed and force a misclassification.
        self.original_array = attack_array

        #A 2-length tuple containing the minimum and maximum values within the attack_array.
        self.array_range = array_range

        #The numerical change (max - min) from the array_range.
        self.range = array_range[1] - array_range[0]

        #The datatype of the array.
        self.dtype = self.original_array.dtype

        #A scaled version of the original array, used as a reference point.
        # self.attack_array = self.scaleDown(self.original_array, self.array_range[0], self.array_range[1])

        #A string containing the path to the checkpoint file.
        self.checkpoint_file = checkpoint_file

        #The checkpoint itself after being loaded by numpy.
        self.checkpoint = None
        if self.checkpoint_file is not None and exists(self.checkpoint_file):
            self.checkpoint = np.load(self.checkpoint_file)
            # self.checkpoint = self.scaleDown(self.checkpoint, self.array_range[0], self.array_range[1])
        
        #Shape of the array.
        self.shape = self.original_array.shape

        #A set containing all of the values that differ from the original. Will be used during the reset action.
        self.reset_set = set()

        #If no checkpoint is given, start from the original.
        if self.checkpoint is None:
            self.perturb_array = np.copy(self.original_array)
        else:
            self.perturb_array = np.copy(self.checkpoint)
            #If using a checkpoint file, add the changed values to reset_set
            for idx, _ in np.ndenumerate(self.perturb_array):
                if self.perturb_array[idx] != self.original_array[idx]:
                    self.reset_set.add(idx)

        #The observation space is the space of all values that can be provided as input.
        #In this case, the agent should receive the perturbed version of the array.
        self.observation_space = Box(low=self.array_range[0], high=self.array_range[1], shape=self.shape, dtype=np.float32)

        #Actions are made up of three components:
        #Strength - What the value should be changed to (after scaling)
        #Index of the value to pick. Each dimension gets their own action.
        self.action_space = Box(low=0, high=1, shape=(len(self.shape) + 1,), dtype=np.float32)
        
        #Get current results to determine the type of output from classifier.
        #If the classifier returns a list, the agent will try to maximize the index determined from new_class
        #If the classifier returns something else, the agent will try to make new_class occur.
        results = self.predict_wrapper(self.original_array, self.victim_data)
        self.result_type = "object"
        if isinstance(results, list) or isinstance(results, tuple) or isinstance(results, np.ndarray):
            self.result_type = "list"

        #The new classification after a successful perturbation.
        self.new_class = new_class
        #Get the orignal class (which will be used for untargeted attacks and verifying final results)
        if self.result_type == "list":
            #If the results return a list, the original class is the index of the highest prediction for the original array.
            self.original_class = np.argmax(results)
        else:
            self.original_class = results

        #What similarity is required for a successful morph.
        self.similarity_threshold = similarity

        #Location where the final results will be stored.
        self.result_file = result_file

        #Store the misclassification and similarity for the current perturbed array.
        self.best_misclassification = 0
        self.best_similarity = 0

        #Stores data for graphing
        self.graph_file = None
        if graph_file is not None:
            self.graph_file = graph_file
            if exists(self.graph_file):
                graph_data = np.load(self.graph_file).tolist()
                self.best_misclassification_record = graph_data[0]
                self.best_similarity_record = graph_data[1]
            else:
                self.best_misclassification_record = []
                self.best_similarity_record = []

        
        #Number of steps taken so far.
        self.timesteps = 0

        #Periodic interval that prints progress.
        self.render_interval = render_interval

        #Periodic interval that saves model, checkpoint, and graph (if they exist)
        self.save_interval = save_interval

    #When the agent receives an action, it will act upon it and determine how good/bad the action was.
    def step(self, action):
        self.timesteps += 1

        #Create a copy of the current perturbed array, then sample the action on this copy.
        perturb_test = np.copy(self.perturb_array)

        #The new value of the location
        perturb_strength = self.scaleUp(action[0], self.array_range[0], self.array_range[1], self.dtype)

        #The location of the change
        perturb_location = action[1::]

        #For each dimension in the action, scale it up to match the shape.
        for idx, dim in np.ndenumerate(perturb_location):
            perturb_location[idx[0]] = self.scaleUp(dim, 0, self.shape[idx[0]] - 1, int)
        
        perturb_location = tuple(perturb_location.astype(int))

        #Perform the unit change action on the designated location.
        perturb_test[perturb_location] = perturb_strength
        
        #Calculate the results of the change
        misclassification, similarity, results = self.collect_diagnostics(perturb_test)

        #If this perturbed array has a higher score than the current best, then this array is the new best.
        reward = 0
        if misclassification >= self.best_misclassification:
            reward = misclassification
            self.best_misclassification = misclassification
            self.best_similarity = similarity
            self.perturb_array = np.copy(perturb_test)

            #Add or remove the value from reset_set, depending on what the new value is and how it compares to the original.
            if self.perturb_array[perturb_location] != self.original_array[perturb_location]:
                if perturb_location not in self.reset_set:
                    self.reset_set.add(perturb_location)
            elif perturb_location in self.reset_set:
                self.reset_set.remove(perturb_location)

        #If the array reaches a high enough misclassification, but not similarity, start a reset
        if self.successful_misclass(results):
            if self.render_interval > 0:
                print("Successful Misclassification - Resetting")
            
            marked = set()
            reset_bar = 0.5
            #For every value that can be reset
            resets = -1
            while resets != 0:
                resets = 0
                for location in self.reset_set:
                    #Temporarily reset the value
                    changed_value = self.perturb_array[location]
                    original_value = self.original_array[location]
                    self.perturb_array[location] = changed_value + (float(original_value) - changed_value) * 0.5
                    #Calculate the diagnostics of this newly reset value
                    reset_misclass, reset_similarity, reset_results = self.collect_diagnostics(self.perturb_array)
                    if (self.successful_misclass(reset_results) and self.perturb_array[location] != changed_value):
                        reset_bar = reset_misclass
                        resets += 1
                    else:
                        self.perturb_array[location] = changed_value
                        reset_misclass, reset_similarity, reset_results = self.collect_diagnostics(self.perturb_array)
                    
                    if self.perturb_array[location] == original_value:
                        marked.add(location)
            
                # misclassification, similarity, results = self.collect_diagnostics(self.perturb_array)
                self.reset_set = self.reset_set.difference(marked)
                marked = set()
                print(f"Resets = {resets}")
            
            #Update the statistics to reflect the reset
            misclassification, similarity, results = self.collect_diagnostics(self.perturb_array)
            self.best_misclassification = misclassification
            self.best_similarity = similarity

        #When the render interval passes, print out the best score, misclassification, and similarity.
        if self.render_interval > 0 and self.timesteps % self.render_interval == 0:
            self.render()

        #If graphing is on, save the score progression.
        if self.save_interval > 0 and self.graph_file is not None:
            self.best_misclassification_record.append(self.best_misclassification)
            self.best_similarity_record.append(self.best_similarity)
            #If save interval passes, redraw the graph.
            if self.timesteps % self.save_interval == 0:
                graph_data = []
                graph_data.append(self.best_misclassification_record)
                graph_data.append(self.best_similarity_record)
                np.save(self.graph_file, graph_data)

        #If the array has been successfully misclassified and has a higher similarity than the threshold, save the result. 
        if self.successful_misclass(results):
            np.save(self.result_file, self.perturb_array)
            print(f"Successful perturb! Array saved at {self.result_file}")
            if self.result_type == "list":
                print(f"Original Index: {self.original_class}")
                print(f"New Index: {np.argmax(results)}")
            else:
                print(f"Original Class: {self.original_class}")
                print(f"New Class: {results}")
            self.render()
            exit()
        
        #If save interval passes, save the current checkpoint
        elif self.save_interval > 0 and self.timesteps % self.save_interval == 0 and self.checkpoint_file is not None:
            np.save(self.checkpoint_file, self.perturb_array)
            print(f"Checkpoint array saved at {self.checkpoint_file}")

        return self.scaleDown(perturb_test, self.array_range[0], self.array_range[1]), reward, False, {} 
    
    def successful_misclass(self, results):
        #Determine if misclassified
        #If untargeted
        if self.new_class is None:
            #If results are a list
            if self.result_type == "list" and np.argmax(results) != self.original_class:
                return True
            elif self.result_type == "object" and results != self.original_class:
                return True
        #If targeted
        else:
            if self.result_type == "list" and np.argmax(results) == self.new_class:
                return True
            elif self.result_type == "object" and results == self.new_class:
                return True
        return False
    
    def collect_diagnostics(self, perturb_array):
        #Get the results from the classifier.
        results = self.predict_wrapper(perturb_array, self.victim_data)

        #If the attack is untargeted, misclassification is determined by how low the original class score is. 
        if self.new_class is None:
            #If the victim returns a list, the misclassification is the sum of all other values.
            if self.result_type == "list":
                misclassification = sum(results) - results[self.original_class]
            #If its not a list, the misclassification is 0 if the result is still the original class, 1 if otherwise.
            else:
                if results == self.original_class:
                    misclassification = 0
                else:
                    misclassification = 1
        #If the attack is targeted, misclassification is determined by how high the new class score is. 
        else:
            #If the victim returns a list, the misclassification is the value found at the desired index (specified by new_class)
            if self.result_type == "list":
                misclassification = results[self.new_class]
            #If its not a list, the misclassification is 1 if the result is the new class, 0 if otherwise.
            else:
                if results == self.new_class:
                    misclassification = 1
                else:
                    misclassification = 0

        #Similarity is measured by the distance between the original array and the perturbed array.
        euclid_distance = 0
        for idx, _ in np.ndenumerate(perturb_array):
            # Find the difference in values, normalize the value, then square it.
            value_distance = (float(perturb_array[idx]) - self.original_array[idx]) ** 2
            euclid_distance += value_distance
        
        # Renormalize the final result, take the square root, then subtract that value from 1 to find similarity.
        similarity = 1 - math.sqrt(euclid_distance / (math.prod(self.shape) * (self.range ** 2)))

        return (misclassification, similarity, results)

    #Reset the parameters (Only called during initialization (and sometimes evaluation))
    def reset(self):
        #Rebuilt the reset set.
        self.reset_set = set()

        #Set the array back to what it was during initialization.
        if self.checkpoint is None:
            self.perturb_array = copy.deepcopy(self.original_array)
        else:
            self.perturb_array = copy.deepcopy(self.checkpoint)
            for idx, _ in np.ndenumerate(self.perturb_array):
                if self.perturb_array[idx] != self.original_array[idx]:
                    self.reset_set.add(idx)

        #Reset the best array statistics.
        self.best_misclassification = 0
        self.best_similarity = 0

        #Reset the best array statistic scores.
        if self.graph_file is not None:
            if exists(self.graph_file):
                graph_data = np.load(self.graph_file).tolist()
                self.best_misclassification_record = graph_data[0]
                self.best_similarity_record = graph_data[1]
            else:
                self.best_misclassification_record = []
                self.best_similarity_record = []

        return self.scaleDown(self.perturb_array, self.array_range[0], self.array_range[1])
    
    #If render_interval > 0, display the original and perturbed arrays in pyplot.
    def render(self):
        print_misclassification = np.format_float_scientific(self.best_misclassification, 3)
        print_similarity = round(self.best_similarity * 100, 3)
        print(f"Misclassification: {print_misclassification} - Similarity: {print_similarity}%")
    
    #Scale the array from [0, 1] to [Min, Max], then cast to the proper datatype
    def scaleUp(self, arr, min, max, dtype):
        range = max - min
        scaled_arr = (arr * range + min).astype(dtype)
        return scaled_arr

    # #Scale the array from [Min, Max] to [0, 1], then cast to the proper datatype
    def scaleDown(self, arr, min, max):
        range = max - min
        scaled_arr = ((arr.astype(np.float32) - min) / range)
        return scaled_arr
    
    #Return the best misclassification
    def get_best_misclassification(self):
        return self.best_misclassification
    
    #Return the best similarity
    def get_best_similarity(self):
        return self.best_similarity

