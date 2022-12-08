import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from utils import transform_profile, isolate_bunches_from_dm_profile
from datamatrix_lookup_class_quad import Datamatrix_lookup_class_quad
from datamatrix_lookup_class_double import Datamatrix_lookup_class_double
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import copy
import random
import gym

data_class = Datamatrix_lookup_class_double()
# some self variables that you could use in the real environment class

def get_state_from_profile(profile):
    """Returns a state description of a double splitting given a final profile.

    Args:
        profile (numpy.ndarray): array or list-like object describing a profile of a double splitting.

    Returns:
        numpy.ndarray: A numpy array of four values, [bunch_length_1, bunch_length_2, bunch_intensity_1, bunch_intensity_2]
    """   
    bunches, fwhms, intensities = isolate_bunches_from_dm_profile(profile, intensities=True, rel=True, plot_found_bunches=False)


    # Standardization/normalization
    fwhms = fwhms-np.mean(fwhms) # The FWHMs are returned normalized, but need to be centered around zero
    intensities = intensities / max(intensities) # The intensities are normalized, and centered around zero
    intensities = intensities - np.mean(intensities)
    bls_and_intensities = np.append(fwhms, intensities)
    return bls_and_intensities

def get_diff_estimate_from_state(state):
    """
    Returns an estimate descibing the difference between the final two bunches lengths and intensities
    given the final relative bunch lengths and intensities of a double splitting (a state above).

    Args:
        state (numpy.ndarray): A numpy array of four values, [bunch_length_1, bunch_length_2, bunch_intensity_1, bunch_intensity_2]

    Returns:
        diff_estimate: A single value aiming to describe the difference between the final bunches.
    """   
    observable = state
    relative_bunch_lengths = observable[:2]
    relative_bunch_intensities = observable[2:]
    bunch_length_difference = abs(relative_bunch_lengths[0]-relative_bunch_lengths[1])
    bunch_intensity_difference = abs(relative_bunch_intensities[0]-relative_bunch_intensities[1])

    diff_estimate = bunch_intensity_difference + bunch_length_difference
    
    return diff_estimate

class YourDoubleEnvDiscrete(gym.Env):
    
    # Define your action and observation spaces. You want to take actions in one dimension, changing the h42 phase offset,
    # and want a 4D observation, consisting of your relative bunch lengths and bunch intensities. Both your observations and 
    # actions will be should be normalized to be within [-1,1] for optimisation reasons.
    
    metadata= {'render.modes': ['human']}
    
    action_space = gym.spaces.Discrete(5) # # {-2, -1, 0, 1, 2}. Five possible actions.


    ### Define what the observations are to be expected
    observation_space = gym.spaces.Box(
                                low = np.array([-1,-1,-1,-1]), # Fill in the __ with your observation settings.
                                high = np.array([1,1,1,1]),
                                shape=(4,),
                                dtype=np.float32)
    
    ### Stop criteria (constituting "good" splittings with provided difference estimate).
    BUNCH_LENGTH_INT_CRITERIA = 0.02 # Empirically evaluated diff_estimate that constitutes a "good" bunch splitting. Lower means longer training time, but smaller spread in bunch lengths/intensities.
    

    
    

    def __init__(self,
                max_steps = 100,
                max_step_size = 20,
                min_setting = -45,
                max_setting = 45,):
        
        ### Assign hyperparameter settings to attributes
        self.max_steps = max_steps
        self.max_step_size = max_step_size
        self.min_setting = min_setting
        self.max_setting = max_setting
        
        ### Status of the iterations
        # Steps, initializing lists to store actions/states/rewards...
        
        self.counter = 0 # Counts the number of steps taken in the environment!
        self.curr_step = -1  ## Not used ?
        self.phase_correction = 0

        # Initialize lists for tracking episodes
        self.curr_episode = -1
        self.action_episode_memory = []
        self.diff_estimate_memory = []
        self.state_memory = []
        self.phase_set_memory = []
        self.reward_memory = []
        self.is_finalized = False
    
    def step(self, action):
        """
        One step/action in the environment, returning the observable
        and reward. 

        Stopping conditions: max_steps reached, or splitting good enough.
        """
        success = False    
        ################################################################################################################################################################
        # Implement the code below!!
        ################################################################################################################################################################
        
        # Hint: Number of steps taken in environment is tracked in self.counter
        
        
        self._take_action(action) # Actually take action: Define your _take_action function below!! 
        reward = self._get_reward() # Get your reward: Define your _get_reward function below!! Returns a reward value.
        
        state = self.state
        ### Check exit criteria: Achieved good enough state, or taken too many steps.
        ### BUNCH_LENGHT_INT_CRITERIA based on the given diff_estimate value.
        
        if abs(self.diff_estimate) < self.BUNCH_LENGTH_INT_CRITERIA: # Check if the diff_estimate is below criterion. If so, episode finalized and a success!
            self.is_finalized = True 
            success = True
        # print(self.counter)    
        if self.counter >= self.max_steps: # Check if you have exceeded the maximum step limit. If so, episode finalized but not a success...
            self.is_finalized = True

        # Here you can add any extra info you would like to be returned on each step, e.g. episode steps, rewards, actions etc.
        info = {'success': success, 'steps': self.counter, 'profile': self.profile} 
        
        ################################################################################################################################################################
        # Implement the code above!!
        ################################################################################################################################################################
        
        
        return state, reward.astype(np.float64), self.is_finalized, info # Standardized output according to gym framework.

    def _take_action(self, action):
        """
        Actual action funtion.

        Action from model is scaled to be between [-1,1] for better optimization performance. 
        Converted back to phase setting in degrees using self.max_step_size.
        
        Args:
            action (ndarray): n-dimensional action. Datatype, dimension, and value ranges defined in self.action_space.
        """
        
        # Input action is one of five values: -2,-1,0,1,2. We need to correlate these with different actions.
        
        # -2: Large negative step of -5 degrees
        # -1: Small negative step of -0.25 degrees
        # 0: No action
        # 1: Small positive step of +0.25 degrees
        # 2: Large positive step of +5 degrees
        
        if action == 0:
            converted_action = -5
        elif action==1:
            converted_action=-0.25
        elif action==2:
            converted_action=0
        elif action==3:
            converted_action=0.25
        elif action==4:
            converted_action=5
        
        self.phase_correction += converted_action # Phase correction tracks previous actions taken to get the cumulative change from start.

        # Phase offset as action, add offset to current phase_set to get next setting. This is what defines which simulated datapoint to collect in your _get_state function!!!
        self.phase_set += converted_action

        # Update the self.state parameter with the new state. The preprovided self._get_state() will provide you with a state based on the current self.phase_set attribute value.
        # The state will consist of a vector of 4 values: [bunch_length_1, bunch_length_2, bunch_intensity_1, bunch_intensity_2].
        self.state = self._get_state()


        ################################################################################################################################################################
        # Implement the code above!!
        ################################################################################################################################################################  
        curr_state = self.state.copy()
        curr_phase_set = np.copy(self.phase_set)
        self.action_episode_memory.append(action)
        self.state_memory.append(curr_state)
        self.phase_set_memory.append(curr_phase_set)
        
        self.counter += 1

     
    
    def _get_state(self):
        '''
        Get the observable for a given phase_set. This function is provided completed to help you collect datapoints from the pre-simulated dataset.
        The data_class class is written to provide datapoints from a quadsplit dataset, but by always providing h84=0 we only vary the first phase
        offset.

        Comment: The edge cases of trying to move to datapoints outside the simulated dataset needs to be handled.
        Currently it is simply checked whether the phase setting is above the max setting or below the min setting,
        and if so a pre-defined dummy observation is presented. The important factor to consider is to make sure that
        all edge cases are covered by some dummy state, and that the dummy states are unique (so the model can learn
        what steps to take to get back in the right search area). It is also highly advised to give an additional penalty
        in the reward if the agent steps outside our region of simulated data.
        '''
        
        ### Check whether we are within simulated settings
        if (self.phase_set<self.min_setting):
            state = np.array([0.5, -0.5, 0.5, -0.5])
        elif (self.phase_set>self.max_setting):
            state = np.array([-0.5, 0.5, -0.5, 0.5])
        else:
            
            ################################################################################################################################################################
            # Implement the code below!!
            #########################################################################################################################
            
            # Collecting the simulated datapoint, calculating state description
            profile = data_class.get_interpolated_profile(self.phase_set) # Second phase does not affect the first. Since we only care about h42, no need to assign h84 offfset.
            state = get_state_from_profile(profile)
            
            ################################################################################################################################################################
            # Implement the code above!!
            ################################################################################################################################################################
        
            self.profile = profile # Add a tracking of the profile for plotting purposes!
        return state.astype(np.float32)
    
    def _get_reward(self):
        """ Evaluating the reward from the observable/state. 
            The example reward 'simple_profile' is provided, and is based on the final profile
            after the splitting. This is using more information than just the state provided
            to the agent.

            Feel free to experiment and design your own reward as well! 

        Returns:
            float: The reward based on the current state. 
        """ 
        
        
        
        # The shape of your observable should match your optimization_space. For this excercise, it is expected
        # that you use an observation space of a vector with four values
        observable = self.state

        diff_estimate = get_diff_estimate_from_state(observable)
        self.diff_estimate = diff_estimate

        ################################################################################################################################################################
        # Implement the code below!!
        ################################################################################################################################################################
        """ 
         Define your own reward here. The diff_estimate provided above is provided 
         as a simple way to define the difference between your bunches length/intensity 
         after the splitting. You want this to be as small as possible, so a better reward
         should be given for a smaller diff_estimate. There is already a pre-defined
         criterion for the diff_estimate to be considered a "good" splitting provided in
         the BUNCH_LENGTH_INT_CRITERIA attribute. 
         """

        reward = -diff_estimate #______ # Define your own reward here!! 

        ################################################################################################################################################################
        # Implement the code above!!
        ################################################################################################################################################################
        
        ### Tracking of diff_estimate. Lets you use my render() function to observe your agent.
        curr_diff_estimate = self.diff_estimate.copy()
        self.diff_estimate_memory.append(curr_diff_estimate)
        self.reward_memory.append(reward)
            
        return reward
       
    def reset(self):
        """
        Reset to a random state to start over a training episode.
        This function will be called everytime an episode is started
        to provide an initial state.
        
        Returns:
            state: The inital state of the environment. Should match the shape defined in observation_space.
        """
        # Resetting to start a new episode
        self.curr_episode += 1
        self.counter = 0
        self.is_finalized = False # This tracks wether an episode is complete or not.
        
        #print(f'Resetting for episode {self.curr_episode}')

        # Initializing episode lists to track data for individual episodes. Some used for rendering.
        self.action_episode_memory = []
        self.state_memory= []
        self.phase_set_memory= []
        self.reward_memory= []
        self.diff_estimate_memory= []
        
        ################################################################################################################################################################
        # Implement the code below!!
        ################################################################################################################################################################
        
        # Getting initial state
        
        # Initialize random starting phase_set within [self.min_setting, self.max_setting], for example using random.uniform(min,max)
        self.phase_set = random.uniform(self.min_setting,
                                            self.max_setting)
        
        self.initial_offset = np.copy(self.phase_set)
        self.phase_correction = 0
                                        
        self.state = self._get_state() # call _get_state to get the initial state from the starting phase_set.
        state = self.state
        
        ################################################################################################################################################################
        # Implement the code above!!
        ################################################################################################################################################################
 
        ### Some tracking of state, phase, reward, diff_estimate. Lets you use my render() function to observe your agent.
        self.state_memory.append(state)
        self.phase_set_memory.append(self.phase_set)
        reward = self._get_reward()
        self.reward_memory.append(reward)
        
        return state.astype(np.float32)

    def seed(self, seed=None):
        """
        Set the random seed. Useful if you want to standardize trainings.
        """
        
        random.seed(seed)
        np.random.seed
        
    def render(self, mode='human'):
        
        """
        Rendering function meant to provide a human-readable output. Base function in gym
        environments to override. I provide a simple version that should let you observe 
        your trained agent during evaluation.
        """
        plt.figure('Agent')
        plt.clf()
        plt.subplot(131)
        plt.suptitle(f'Episode {self.curr_episode}')
        plt.title('Current profile')
        plt.plot(self.profile,'b')
        plt.subplot(132)
        plt.title('Difference estimate')
        plt.plot(self.diff_estimate_memory, 'o-')
        plt.axhline(y=self.BUNCH_LENGTH_INT_CRITERIA, color='k', linestyle='--')
        plt.subplot(133)
        plt.title('h42 phase offset')
        plt.plot(np.asarray(self.phase_set_memory, dtype=object), 'go-')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.ylim((-30,30))
        
        #plot_finish(fig=fig, axes=axes, xlabel='Setting', ylabel='Observable')
        plt.pause(0.2)
        

from stable_baselines3 import DQN

# Separate evaluation env
eval_env = YourDoubleEnvDiscrete()

# Use deterministic actions for evaluation
eval_callback = EvalCallback(eval_env,  best_model_save_path='./RL_logs/DQN',
    log_path='./RL_logs/', eval_freq=1000,
    deterministic=True, render=False)

env = YourDoubleEnvDiscrete()

seed = 7
np.random.seed(seed)
env.seed(seed)

dqn_model = DQN("MlpPolicy", env, verbose=1, learning_starts=10000, tensorboard_log="./hands-on_rl_tensorboard")
print("Starting training...")
dqn_model.learn(total_timesteps=50000, callback=eval_callback, log_interval=5, tb_log_name="hands_on_rl_DQN")
dqn_model.save('./saved_models/RL_agent_DQN')
print("Completed training")