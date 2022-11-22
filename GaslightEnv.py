import copy
import cv2
import gym
import math
import matplotlib.pyplot as plt
import numpy as np
import random

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

        self.attack_array = self.scaleDown(self.original_array, self.array_range[0], self.array_range[1])

        #A string containing the path to the checkpoint file.
        self.checkpoint_file = checkpoint_file

        #The checkpoint itself after being loaded by numpy.
        self.checkpoint = None
        if self.checkpoint_file is not None and exists(self.checkpoint_file):
            self.checkpoint = np.load(self.checkpoint_file)
            self.checkpoint = self.scaleDown(self.checkpoint, self.array_range[0], self.array_range[1])
        
        #Shape of the array.
        self.shape = self.attack_array.shape

        #A set containing all of the values that differ from the original. Will be used during the reset action.
        self.reset_set = set()

        #If no checkpoint is given, start from the original.
        if self.checkpoint is None:
            self.perturb_array = np.copy(self.attack_array)
        else:
            self.perturb_array = np.copy(self.checkpoint)
            #If using a checkpoint file, add the changed values to reset_set
            for idx, _ in np.ndenumerate(self.perturb_array):
                if self.perturb_array[idx] != self.attack_array[idx]:
                    self.reset_set.add(idx)

        #The observation space is the space of all values that can be provided as input.
        #In this case, the agent should receive a matrix of values.
        self.observation_space = Box(low=0, high=1, shape=self.shape, dtype=np.float32)

        #Actions are made up of three components:
        #Strength - What the value should be changed to (after scaling)
        #Index of the value to pick. Each dimension gets their own action.
        self.action_space = Box(low=0, high=1, shape=(2, len(self.shape) + 1), dtype=np.float32)
        
        #Get current results to determine the type of output from classifier.
        #If the classifier returns a list, the agent will try to maximize the index determined from new_class
        #If the classifier returns something else, the agent will try to make new_class occur.
        input_array = self.scaleUp(self.perturb_array, self.array_range[0], self.array_range[1], self.dtype)
        results = self.predict_wrapper(input_array, self.victim_data)
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

        #Agent will store the array with the highest score.
        #That "best score" array will also have its misclassification and similarity recorded for future reference.
        self.best_score = 0
        self.best_misclassification = 0
        self.best_similarity = 0

        #Stores data for graphing
        self.graph_file = None
        if graph_file is not None:
            self.graph_file = graph_file
            self.best_score_record = []
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
        #Each action is actually two separate perturbances.
        #The first action changes a specified value to a specified magnitude, as determined by the neural network.
        #The second action takes a previously changed value and moves it closer to the original.
        self.timesteps += 1

        #Create a copy of the current perturbed array, then sample the action on this copy.
        perturb_test = np.copy(self.perturb_array)
        perturb_strength = action[0][0]
        perturb_location = action[0][1::]

        for idx, dim in np.ndenumerate(perturb_location):
            perturb_location[idx[0]] = self.scaleUp(dim, 0, self.shape[idx[0]] - 1, int)
        
        perturb_location = tuple(perturb_location.astype(int))

        #If the current similarity is too low, activate and sample the reset action on another copy.
        if len(self.reset_set) > 0 and self.best_similarity < self.similarity_threshold and self.best_score > 0:
            reset_test = np.copy(self.perturb_array)
            reset_strength = action[1][0]
            reset_location = action[1][1::]

            for idx, dim in np.ndenumerate(reset_location):
                reset_location[idx[0]] = self.scaleUp(dim, 0, self.shape[idx[0]] - 1, int)
            
            reset_location = tuple(reset_location.astype(int))

        #Perform the unit change action on the designated location.
        perturb_test[perturb_location] = perturb_strength

        #Reset actions only occur if there are values to revert and if similarity is too low.
        if len(self.reset_set) > 0 and self.best_similarity < self.similarity_threshold and self.best_score > 0:
            closest_distance = sum([(dim - 1) ** 2 for dim in self.shape])
            closest_location = None
            for changed_location in self.reset_set:
                distance = 0
                for idx in range(len(changed_location)):
                    distance += (reset_location[idx] - changed_location[idx]) ** 2
                if distance <= closest_distance:
                    closest_distance = distance
                    closest_location = changed_location
                
            #Reset the value by moving it's value closer to the original.
            value_delta = (self.perturb_array[closest_location] - reset_test[closest_location]) * reset_strength
            reset_test[closest_location] = reset_test[closest_location] + value_delta
            reset_location = closest_location

        #Get the results from the classifier.
        perturb_data = self.collect_diagnostics(perturb_test)
        #If reset action activated, also grab its results.
        if len(self.reset_set) > 0 and self.best_similarity < self.similarity_threshold and self.best_score > 0:
            reset_data = self.collect_diagnostics(reset_test)
        
        #Determine if the perturb or reset action yielded a better score, and use that for the primary action of this step.
        perturb_action = True
        affected_location = tuple(perturb_location)
        if len(self.reset_set) > 0 and self.best_similarity < self.similarity_threshold and self.best_score > 0 and reset_data[2] > perturb_data[2]:
            perturb_action = False
            misclassification, similarity, score, results = reset_data
            affected_location = tuple(reset_location)
        else: 
            misclassification, similarity, score, results = perturb_data

        # print(affected_location)
        #Determine if the perturb was successful (Misclassified and High Similarity)
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
        
        #If this perturbed array has a higher score than the current best, then this array is the new best.
        reward = score - self.best_score
        if score >= self.best_score:
            self.best_score = score
            self.best_misclassification = misclassification
            self.best_similarity = similarity
            if perturb_action:
                self.perturb_array = copy.deepcopy(perturb_test)
            else:
                self.perturb_array = copy.deepcopy(reset_test)

        #Add or remove the value from reset_set, depending on what the new value is and how it compares to the original.
        if self.perturb_array[affected_location] != self.attack_array[affected_location]:
            if affected_location not in self.reset_set:
                self.reset_set.add(affected_location)
        elif affected_location in self.reset_set:
            self.reset_set.remove(affected_location)

        #When the render interval passes, print out the best score, misclassification, and similarity.
        if self.render_interval > 0 and self.timesteps % self.render_interval == 0:
            self.render()

        #If graphing is on, save the score progression.
        if self.save_interval > 0 and self.graph_file is not None:
            self.best_score_record.append(self.best_score)
            self.best_misclassification_record.append(self.best_misclassification)
            self.best_similarity_record.append(self.best_similarity)
            #If save interval passes, redraw the graph.
            if self.timesteps % self.save_interval == 0:
                timestep_record = list(range(self.timesteps))
                plt.plot(timestep_record, self.best_score_record, label="Best Score")
                plt.plot(timestep_record, self.best_misclassification_record, label="Misclassification")
                plt.plot(timestep_record, self.best_similarity_record, label="Similarity")
                plt.xlabel("Timesteps")
                plt.ylabel("Score [0-1]")
                plt.legend()
                plt.savefig(self.graph_file)
                plt.close()

        #If the array has been successfully misclassified and has a higher similarity than the threshold, save the result. 
        if successful_perturb and self.best_similarity >= self.similarity_threshold:
            save_array = self.scaleUp(self.perturb_array, self.array_range[0], self.array_range[1], self.dtype)
            np.save(self.result_file, save_array)
            print(f"Successful perturb! Array saved at {self.result_file}")
            if self.result_type == "list":
                print(f"Original Index: {self.original_class}")
                print(f"New Index: {np.argmax(results)}")
            else:
                print(f"Original Class: {self.original_class}")
                print(f"New Class: {results}")
            exit()
        
        #If save interval passes, save the current checkpoint
        elif self.save_interval > 0 and self.timesteps % self.save_interval == 0 and self.checkpoint_file is not None:
            save_array = self.scaleUp(self.perturb_array, self.array_range[0], self.array_range[1], self.dtype)
            np.save(self.checkpoint_file, save_array)
            print(f"Checkpoint array saved at {self.checkpoint_file}")

        return self.perturb_array, reward, False, {}
    
    def collect_diagnostics(self, perturb_array):
        #Get the results from the classifier.
        perturb_input = self.scaleUp(perturb_array, self.array_range[0], self.array_range[1], self.dtype)
        results = self.predict_wrapper(perturb_input, self.victim_data)

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
            value_distance = (perturb_array[idx] - self.attack_array[idx]) ** 2
            euclid_distance += value_distance
        
        # Renormalize the final result, take the square root, then subtract that value from 1 to find similarity.
        similarity = 1 - math.sqrt(euclid_distance / math.prod(self.shape))
        
        score = misclassification * similarity

        return (misclassification, similarity, score, results)

    #Reset the parameters (Only called during initialization (and sometimes evaluation))
    def reset(self):
        #Rebuilt the reset set.
        self.reset_set = set()
        #Set the array back to what it was during initialization.
        if self.checkpoint is None:
            self.perturb_array = copy.deepcopy(self.attack_array)
        else:
            self.perturb_array = copy.deepcopy(self.checkpoint)
            for idx, _ in np.ndenumerate(self.perturb_array):
                if self.perturb_array[idx] != self.attack_array[idx]:
                    self.reset_set.add(idx)

        #Reset the best array statistics.
        self.best_score = 0
        self.best_misclassification = 0
        self.best_similarity = 0

        #Reset the best array statistic scores.
        if self.graph_file is not None:
            self.best_score_record = []
            self.best_misclassification_record = []
            self.best_similarity_record = []

        return self.perturb_array
    
    #If render_level > 1, display the original and perturbed arrays in pyplot.
    def render(self):
        print_score = np.format_float_scientific(self.best_score, 3)
        print_misclassification = np.format_float_scientific(self.best_misclassification, 3)
        print_similarity = round(self.best_similarity * 100, 3)
        print(f"Score: {print_score} - Misclassification: {print_misclassification} - Similarity: {print_similarity}%")
    
    #Scale the array from [0, 1] to [Min, Max], then cast to the proper datatype
    def scaleUp(self, arr, min, max, dtype):
        range = max - min
        scaled_arr = (arr * range + min).astype(dtype)
        return scaled_arr

    #Scale the array from [Min, Max] to [0, 1], then cast to the proper datatype
    def scaleDown(self, arr, min, max):
        range = max - min
        scaled_arr = ((arr - min) / range).astype(np.float32)
        return scaled_arr

    #Return the best score
    def get_best_score(self):
        return self.best_score
    
    #Return the best misclassification
    def get_best_misclassification(self):
        return self.best_misclassification
    
    #Return the best similarity
    def get_best_similarity(self):
        return self.best_similarity

