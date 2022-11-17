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
    def __init__(self, predict_wrapper, victim_data, attack_array, array_range, new_class, similarity=0.7, render_level=0, checkpoint_level=0, checkpoint_file=None, graph_file=None):
        #Wrapper that will take the perturbed array, have the victim make a prediction, then return the results.
        self.predict_wrapper = predict_wrapper

        #A collection of data that needs to be passed to predict_wrapper. Should include the model information and any preprocessing tools.
        self.victim_data = victim_data

        #The original array to get perturbed and force a misclassification.
        self.attack_array = attack_array

        #A 2-length tuple containing the minimum and maximum values within the attack_array.
        self.array_range = array_range

        #The numerical change (max - min) from the array_range.
        self.range = array_range[1] - array_range[0]

        #The datatype of the array.
        self.dtype = attack_array.dtype

        #A string containing the path to the checkpoint file.
        self.checkpoint_file = checkpoint_file

        #The checkpoint itself after being loaded by numpy.
        self.checkpoint = None
        if self.checkpoint_file is not None and exists(self.checkpoint_file):
            self.checkpoint = np.load(self.checkpoint_file)
        
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
        self.observation_space = Box(low=self.array_range[0], high=self.array_range[1], shape=self.shape, dtype=self.dtype)

        #Actions are made up of three components:
        #0 - Unit Change - What the value should be changed to (after scaling)
        #1 - Reset Strength - During a reset action, what percentage of the gap between the perturbed and original array should be closed.
        #2+ - The index of the value to pick. Each dimension gets their own action.
        num_actions = len(self.shape) + 2
        self.action_space = Box(low=0, high=1, shape=(num_actions,), dtype=np.float32)
        
        #Get current results to determine the type of output from classifier.
        #If the classifier returns a list, the agent will try to maximize the index determined from new_class
        #If the classifier returns something else, the agent will try to make new_class occur.
        results = self.predict_wrapper(self.attack_array, self.victim_data)
        self.result_type = "object"
        if isinstance(results, list) or isinstance(results, np.ndarray):
            self.result_type = "list"

        #The new classification after a successful perturbation.
        self.new_class = new_class
        #Get the orignal class (which will be used for untargeted attacks and verifying final results)
        if self.result_type == "list":
            #If the results return a list, the original class is the index of the highest prediction for the original array.
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
        
        #Number of steps taken so far.
        self.timesteps = 0

    #When the agent receives an action, it will act upon it and determine how good/bad the action was.
    def step(self, action):
        #Each action is actually two separate perturbances.
        #The first action changes a specified value to a specified magnitude, as determined by the neural network.
        #The second action takes a previously changed value and moves it closer to the original.

        self.timesteps += 1

        #Create a copy of the current perturbed array, then sample the action on this copy.
        perturb_test = np.copy(self.perturb_array)

        #If the current similarity is too low, activate and sample the reset action on another copy.
        if self.best_similarity < self.similarity_threshold:
            reset_test = np.copy(self.perturb_array)

        #Determine what the new value magnitude should be.
        unit_change = np.round(action[0] * self.range + self.array_range[0]).astype(self.dtype)
        
        #If a reset occurs, what percentage of the gap is closed between the original and current values.
        reset_strength = action[1]

        #Convert the rest of the action space into the designated index to modify.
        location = []
        for idx, dim in np.ndenumerate(action[2:len(action)]):
            scaled_dim = int(np.round(dim * (self.shape[idx[0]] - 1)))
            location.append(scaled_dim)
        location = tuple(location)
        reset_location = location
        
        #Perform the unit change action on the designated location.
        perturb_test[location] = unit_change

        #Reset actions only occur if there are values to revert and if similarity is too low.
        if len(self.reset_set) > 0 and self.best_similarity < self.similarity_threshold:

            #A dictionary of "tickets" that determine which value gets reset.
            reset_tickets = {}

            #Start with ticket 0
            ticket_floor = 0

            #Every perturbed location will get a range of tickets.
            for perturbed_location in self.reset_set:

                #Find the difference between original and perturbed values. Larger contrasts yield higher tickets.
                contrast = abs(int(reset_test[perturbed_location]) - int(self.attack_array[perturbed_location]))

                #Start with the last value of the previous value.
                ticket_min = ticket_floor

                #Add one ticket per level of contrast.
                ticket_max = ticket_min + contrast - 1

                #New floor for the next value is one above the current max.
                ticket_floor = ticket_max + 1

                #Store the ticket until its time to draw.
                reset_tickets[perturbed_location] = (ticket_min, ticket_max)
            
            #Draw a random ticket, values with higher contrasts are more likely to get picked.
            ticket = random.randint(0, ticket_floor - 1)

            #Re-iterate through all the perturbed values.
            reset_location = location
            for perturbed_location in self.reset_set:
                location_tickets = reset_tickets[perturbed_location]
                #Find the winner, store its location, then break.
                if ticket >= location_tickets[0] and ticket <= location_tickets[1]:
                    reset_location = perturbed_location
                    break
            
            #Reset the value by moving it's value closer to the original.
            value_delta = (float(self.attack_array[reset_location]) - float(reset_test[reset_location])) * reset_strength
            reset_test[reset_location] = (reset_test[reset_location] + value_delta).astype(self.dtype)

        #Get the results from the classifier.
        perturb_data = self.collect_diagnostics(perturb_test)
        #If reset action activated, also grab its results.
        if self.best_similarity < self.similarity_threshold:
            reset_data = self.collect_diagnostics(reset_test)
        
        #Determine if the perturb or reset action yielded a better reward, and use that for the primary action of this step.
        perturb_action = True
        if self.best_similarity >= self.similarity_threshold or perturb_data[2] >= reset_data[2]:
            perturbance, similarity, reward, results = perturb_data
        else:
            perturb_action = False
            perturbance, similarity, reward, results = reset_data
            location = reset_location

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

        #Add or remove the value from reset_set, depending on what the new value is and how it compares to the original.
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

        #If graphing is on, save the reward progression.
        if self.graph_file is not None:
            self.perturb_scores.append(self.best_perturbance)
            self.similar_scores.append(self.best_similarity)
            self.reward_scores.append(self.best_reward)
            #If there was an improvement, redraw the graph.
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
                if self.checkpoint_file is not None:
                    fake_array_file = self.checkpoint_file
                np.save(fake_array_file, self.perturb_array)
                print(f"Checkpoint array saved at {fake_array_file}")
        
        #If the array has improved at all, save it (if checkpoint_level is 2 or higher)
        elif improvement and self.checkpoint_level > 1:
            checkpoint_array_file = "Checkpoint.npy"
            if self.checkpoint_file is not None:
                checkpoint_array_file = self.checkpoint_file
            np.save(checkpoint_array_file, self.perturb_array)
            print(f"Checkpoint array saved at {checkpoint_array_file}")

        return self.perturb_array, reward, False, {}
    
    def collect_diagnostics(self, perturb_array):
        #Get the results from the classifier.
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
            # Find the difference in values, normalize the value, then square it.
            value_distance = ((float(perturb_array[idx]) - float(self.attack_array[idx])) / self.range) ** 2
            euclid_distance += value_distance
        
        # Renormalize the final result, take the square root, then subtract that value from 1 to find similarity.
        similarity = 1 - math.sqrt(euclid_distance / math.prod(self.shape))

        reward = perturbance * similarity

        return (perturbance, similarity, reward, results)

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
        self.best_reward = 0
        self.best_perturbance = 0
        self.best_similarity = 0

        #Reset the best array statistic scores.
        if self.graph_file is not None:
            self.perturb_scores = []
            self.similar_scores = []
            self.reward_scores = []

        return self.perturb_array

    #Return the best reward
    def get_best_reward(self):
        return self.best_reward
    
    #Return the best perturbance
    def get_best_perturbance(self):
        return self.best_perturbance
    
    #Return the best similarity
    def get_best_similarity(self):
        return self.best_similarity
    
    #If render_level > 1, display the original and perturbed arrays in pyplot.
    def render(self):
        print_perturb = np.format_float_scientific(self.best_perturbance, 3)
        print_similar = round(self.best_similarity * 100, 3)
        print(f"Perturbance: {print_perturb} - Similarity: {print_similar}%")

