# -*- coding: utf-8 -*-
"""
Class defining the possible actions for the agent to do and the corresponding
reward
"""

#%% Imports
import sys
sys.path.append(r'C:\Users\jwulff\cernbox\workspaces\RL-007-RFinPS\optimising-rf-manipulations-in-the-ps-through-rl')

import random
import numpy as np
import gym
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.lib.function_base import diff
from scipy import interpolate
from pathlib import Path
import torch
#from pytorch_fitmodule import FitModule
# from torch.autograd import Variable
import numpy as np
from torchvision import transforms, utils
# from torch.utils.data import Dataset, DataLoader
from dataloader import ToTensor, Normalize, AddNoise
import torch.optim as optim
import matplotlib.pyplot as plt
from datamatrix_lookup_class_quad import Datamatrix_lookup_class_quad
from utils import profile_reward_quad, isolate_bunches_from_dm_profile, loss_function_two

#from plots.enhancer import plot_finish


#%% Twosplit environment class

OFFSET = 5

REWARD_OFFSET = -1

BUNCH_LENGTH_INT_CRITERIA = 0.02 # Empirically evaluated diff_estimate that constitutes a "good" bunch splitting. Lower means longer training time, but smaller spread in bunch lengths/intensities.
PROFILE_MSE_CRITERIA = 0.0003 #
#SIMULATION_DATA = 'corrected'
SIMULATION_DATA = 'uncorrected'

#REWARD_FUNC = 'gauss'
#REWARD_FUNC = 'gauss_profile'
#REWARD_FUNC = 'gauss_linear_profile'
#REWARD_FUNC = 'gauss_profile_step'
#REWARD_FUNC = 'parabola'
#REWARD_FUNC = 'observable'
#REWARD_FUNC = 'simple'
#REWARD_FUNC = 'simple_profile'
REWARD_FUNC = 'your_reward'
#REWARD_FUNC = 'MSE'

ACTION_TYPE = 'relative'
#ACTION_TYPE = 'absolute'




transform=transforms.Compose([
            Normalize(),
            ToTensor(),
            AddNoise(),
        ])

# Loading simulation data


data_class = Datamatrix_lookup_class_quad() # data_class.get_interpolated... gets interp. datamatrix for CNN

phase = np.linspace(-30,30,121) #loaded_results['dpc20'] + OFFSET

min_setting=np.min(phase)
max_setting=np.max(phase)



# Normalize peak_mod_int

#peak_mod_int = peak_mod_int/max_spread



class QuadEnvBLInt(gym.Env):
    """
    Environment class for the h=42 splitting optimization.
    """

    phase_set = 0
    


    action_space = gym.spaces.Box(
            np.array([-1]), # change in p42 to apply. Propagated to p84 to decouple phases.
            np.array([1]),
            shape=(1,),
            dtype=np.float32)


    ### Define what the observations are to be expected
    observation_space = gym.spaces.Box(
            np.array([-1, -1, -1, -1]), # Rel Bunch lengths and intensity after h=42 splitting.
            np.array([1, 1, 1, 1]),
            shape=(4,),
            dtype=np.float32)
    
    ### Observable/State

    state = [0, 0]
    action = 0

    ### Evaluation info
    phase_correction = 0
    initial_offset = 0
    diff_estimate = 0
    diff_estimate_memory = []

    ### Status of the iterations
    # Steps, i.e. number of cycles
    counter = 0
    curr_step = -1  ## Not used ?
    
    # Episodes, i.e. number of MDs, number of LHC fills...
    curr_episode = -1
    action_episode_memory = []
    state_memory = []
    phase_set_memory = []
    reward_memory = []
    is_finalized = False
    def __init__(self, max_step_size=20,
                 min_setting=min_setting, max_setting=max_setting,
                 #min_spread=min_spread, max_spread=max_spread, 
                 max_steps=100,
                 seed=None, 
                 ):
        """
        Initialize the environment
        """
        ### Settings
        self.min_setting = min_setting
        self.max_setting = max_setting
        ### Define what the agent can do
        self.max_step_size = max_step_size
        self.max_steps = max_steps
   
        ### Set the seed
        self.seed(seed)
        
        ### Reset
        #self.reset()
        
        
    def step(self, action):
        """
        One step/action in the environment, returning the observable
        and reward.

        Stopping conditions: max_steps reached, or small enough difference between bunch lengths reached (BUNCH_LENGTH_INT_CRITERIA)
        """
        success = False    
        self.curr_step += 1
        #print(f" state before action {self.state}, action {action}")
        self._take_action(action)
        #print(f" state After action {self.state}")
        reward = self._get_reward()
        curr_diff_estimate = self.diff_estimate.copy()
        self.diff_estimate_memory[self.curr_episode].append(curr_diff_estimate)
        state = self.state
        if REWARD_FUNC == 'simple_profile':
            if abs(self.diff_estimate) < PROFILE_MSE_CRITERIA:
                self.is_finalized = True
                success = True
        elif REWARD_FUNC == 'your_reward':
            if abs(self.diff_estimate) < BUNCH_LENGTH_INT_CRITERIA:
                self.is_finalized = True
                success = True
        if self.counter >= self.max_steps: # Check if you have exceeded the maximum step limit.
            self.is_finalized = True

        
        self.reward_memory[self.curr_episode].append(reward)
        info = {'success': success, 'steps': self.counter, 'phase_corr': self.phase_correction, 'initial_phase': self.initial_offset, 'diff_estimate': self.diff_estimate, 'profile': self.profile}
        
        return state, reward, self.is_finalized, info
    
    def _take_action(self, action):
        """
        Actual action funtion.

        Action from model is scaled to be between [-1,1] for better optimization performance. 
        Converted back to phase setting in degrees using self.max_step_size.

        Args:
            action (ndarray): n-dimensional action. Datatype, dimension, and value ranges defined in self.action_space.
        """        """
        
        """
        
        self.action = action
        converted_action = action*self.max_step_size
        self.phase_correction += converted_action

        # Phase offset as action, add offset to current phase_set to get next setting.
        self.phase_set += converted_action
        
        
        self.state = self._get_state()
        curr_state = self.state.copy()
        curr_phase_set = self.phase_set.copy()
        self.action_episode_memory[self.curr_episode].append(action)
        self.state_memory[self.curr_episode].append(curr_state)
        self.phase_set_memory[self.curr_episode].append(curr_phase_set)
        
        self.counter += 1
    
    def _get_state(self):
        '''
        Get the observable for a given phase_set

        Comment: The edge cases of trying to move to datapoints outside the simulated dataset needs to be handled.
        Currently it is simply checked whether the phase setting is above the max setting or below the min setting,
        and if so a pre-defined dummy observation is presented. The important factor to consider is to make sure that
        all edge cases are covered by some dummy state, and that the dummy states are unique (so the model can learn
        what steps to take to get back in the right search area). It is also highly advised to give an additional penalty
        in the reward if the agent steps outside our region of simulated data.
        '''

        if (self.phase_set<self.min_setting):
            state = np.array([0.5, -0.5, 0.5, -0.5])
        elif (self.phase_set>self.max_setting):
            state = np.array([-0.5, 0.5, -0.5, 0.5])
        else:
            # Interpolating the state/observable from the simulated data
            datamatrix = data_class.get_interpolated_matrix(self.phase_set, 0) # Second phase does not affect the first. Since we only care about h42, no need to assign h84 offfset.
            #self.profile = data_class.get_interpolated_profile(self.phase_set[0], 0)
            # Convert input into normalized tensor
            sample = {}
            sample['image'] = datamatrix
            sample['labels'] = np.array([0,0])
            transformed_sample = transform(sample) # add some transforms to make the datapoint noisy, more similar to measurements.
            input = transformed_sample['image']
            self.profile = input[0,99,:].numpy() # 99th row approximately at c-timing 2793, where h=42 splitting is complete.

            ###############
            # CALCULATE BUNCH LENGTHS AND INTENSITIES FROM THE PROFILE
            ###############
            self.bunches, fwhms, intensities = isolate_bunches_from_dm_profile(self.profile, intensities=True, rel=True, plot_found_bunches=False)
            fwhms = fwhms -np.mean(fwhms)
            # Normalize and recenter intensities around 0
            intensities = intensities / max(intensities) #
            intensities = intensities - np.mean(intensities)
            self.fwhms, self.intensities = fwhms, intensities

            bls_and_intensities = np.append(fwhms, intensities) # state: [bl1,bl2,bi1,bi2]
            state = bls_and_intensities
        
        return state
    
    def _get_reward(self):
        """ Evaluating the reward from the observable/state. 
            The example reward 'simple_profile' is provided, and is based on the final profile
            after the splitting. This is using more information than just the state provided
            to the agent.

            Feel free to experiment and design your own reward as well! 

        Returns:
            float: The reward based on the current state. 
        """ 
        
        
        observable = self.state
        if REWARD_FUNC == 'your_reward':
            relative_bunch_lengths = observable[:2]
            relative_bunch_intensities = observable[2:]
            bunch_length_difference = abs(relative_bunch_lengths[0]-relative_bunch_lengths[1])
            bunch_intensity_difference = abs(relative_bunch_intensities[0]-relative_bunch_intensities[1])

            diff_estimate = bunch_intensity_difference+bunch_length_difference
            self.diff_estimate = diff_estimate
            reward = -diff_estimate

        
        elif REWARD_FUNC == 'simple_profile':
            reward = -1
            diff_estimate = 1000000
            #profile = data_class.get_interpolated_profile(self.phase_volt_set[0], self.phase_volt_set[1], self.phase_volt_set[2])
            diff_estimate = loss_function_two(self.bunches[0], self.bunches[1])
            self.diff_estimate = diff_estimate
            if (self.phase_set<self.min_setting) or (self.phase_set>self.max_setting):
                reward += -4
            
            elif abs(diff_estimate) < PROFILE_MSE_CRITERIA: # Tested to see where it is close to optimal setting
                reward += 101
            elif abs(diff_estimate) < 0.0012:
                reward += 0.75
            elif abs(diff_estimate) < 0.01:
                reward += 0.5
            elif abs(diff_estimate) < 0.02:
                reward += 0.25
       
        return reward
       
    def reset(self):
        """
        Reset to a random state to start over the training
        """
        
        # Resetting to start a new episode
        self.curr_episode += 1
        self.counter = 0
        self.is_finalized = False

        # Initializing lists to track data for episodes
        self.action_episode_memory.append([])
        self.state_memory.append([])
        self.phase_set_memory.append([])
        self.reward_memory.append([])
        self.diff_estimate_memory.append([])

        # Getting initial state
        self.phase_set = np.array(0.0)
        self.phase_set = random.uniform(self.min_setting,
                                            self.max_setting)
        
        self.initial_offset = np.copy(self.phase_set)
        self.phase_correction = 0
                                        
        self.state = self._get_state()
        state = self.state

        self.state_memory[self.curr_episode].append(state)
        self.phase_set_memory[self.curr_episode].append(self.phase_set)
        
        reward = self._get_reward()
        self.reward_memory[self.curr_episode].append(reward)
        curr_diff_estimate = np.copy(self.diff_estimate)
        self.diff_estimate_memory[self.curr_episode].append(curr_diff_estimate)
        return state

    def seed(self, seed=None):
        """
        Set the random seed
        """
        
        random.seed(seed)
        np.random.seed
        
    def render(self, mode='human'):
        
        """
        Rendering function meant to provide a human-readable output. Base function in gym
        environments to override. I provide a simple version that should let you observe 
        your trained agent during evaluation.
        """
        plt.figure(f'Rendering agent')
        plt.clf()
        plt.subplot(131)
        plt.suptitle(f'Episode {self.curr_episode}')
        plt.title('Current profile')
        plt.plot(self.profile,'b')
        plt.subplot(132)
        plt.title('Difference estimate')
        plt.plot(self.diff_estimate_memory[self.curr_episode], 'o-')
        plt.axhline(y=BUNCH_LENGTH_INT_CRITERIA, color='k', linestyle='--')
        plt.subplot(133)
        plt.title('h42 phase offset')
        plt.plot(self.phase_set_memory[self.curr_episode], 'go-')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.ylim((-30,30))
        
        #plot_finish(fig=fig, axes=axes, xlabel='Setting', ylabel='Observable')
        plt.pause(0.2)
        
        
        
        







