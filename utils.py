import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from torchvision import transforms
from dataloader import ToTensor, Normalize, AddNoise
import gymnasium as gym

transform=transforms.Compose([
            Normalize(),
            ToTensor(),
            AddNoise(),
        ])
# Utilities file for RL functions

def test_reward(test_environment):
    phases = np.linspace(-45,45,361)
    rewards = []
    diff_estimates = []

    for phase in phases:
        test_environment.reset() # Initialize 
        test_environment.phase_set = phase # manually change phase setting
        test_environment.state = test_environment._get_state() # Manually update the state based on current self.phase_set
        reward = test_environment._get_reward() # Manually calculate reward based on current self.state

        rewards.append(reward)
        diff_estimates.append(test_environment.diff_estimate) # Also track difference estimates for visited states.

    # Plotting

    ax1= plt.subplot(211)
    plt.title('Reward/diff_estimate given at different phase offsets')
    plt.plot(rewards, 'r')
    plt.ylabel('Received reward')
    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(diff_estimates)
    plt.ylabel('Difference estimate')
    plt.xticks(np.linspace(0,361,5), np.linspace(-45,45,5))
    plt.xlabel('Phase offset [deg]')


# Calculate criterion as a loss
def loss_function_two(b1,b2, verbose=False): 
    # Single out each bunch. 400 bins and three bunches, initial bucket 308ns/3 ~ 102.6667

    mse12 = np.mean((b1-b2)**2)

    # Sum all MSE:s to get a single loss
    loss = mse12
    if verbose:
        print(f"loss: {loss}")
    #reward = -loss
    return loss  

# Calculate criterion as a loss
def loss_function_tri(b1,b2,b3, verbose=False): 
    # Single out each bunch. 400 bins and three bunches, initial bucket 308ns/3 ~ 102.6667

    mse12 = np.mean((b1-b2)**2)
    mse13 = np.mean((b1-b3)**2)
    mse23 = np.mean((b2-b3)**2)

    # Sum all MSE:s to get a single loss
    loss = (mse12 + mse13+ mse23)/3
    if verbose:
        print(f"loss: {loss}")
    #reward = -loss
    return loss 

def transform_profile(input):
    # Convert input into normalized tensor
    sample = {}
    sample['image'] = input
    sample['labels'] = np.array([0,0])
    transformed_sample = transform(sample) # add some transforms to make the datapoint noisy, more similar to measurements.
    input = transformed_sample['image']

    return input.numpy()[0]

def loss_function_quad(b1,b2,b3, b4, verbose=False): 
    # Single out each bunch. 400 bins and three bunches, initial bucket 308ns/3 ~ 102.6667

    mse12 = np.mean((b1-b2)**2)
    mse13 = np.mean((b1-b3)**2)
    mse14 = np.mean((b1-b4)**2)

    mse23 = np.mean((b2-b3)**2)
    mse24 = np.mean((b2-b4)**2)

    mse34 = np.mean((b3-b4)**2)

    # Sum all MSE:s to get a single loss
    loss = (mse12 + mse13+ mse23 + mse14 + mse24 + mse34)/6
    if verbose:
        print(f"loss: {loss}")
    #reward = -loss
    return loss  

def tri_phase_loss(b1,b3, verbose=False):
    mse13 = np.mean((b1-b3)**2)

    # Sum all MSE:s to get a single loss
    loss = mse13
    if verbose:
        print(f"loss: {loss}")
    #reward = -loss
    return loss 

def peaks(profile, height = 0.2, distance=100):
    profile = profile#/np.max(profile) # normalize
    index_of_peaks, peak_heights = find_peaks(profile, height=height, distance=distance)
    return index_of_peaks, peak_heights['peak_heights']


def isolate_bunches_from_dm_profile(profile, plot_found_bunches=False, bunch_width = 15, intensities=False, rel=False, distance=50, height=0.2):
    index_of_peaks, peak_heights = peaks(profile, height=height, distance=distance)
    # print(index_of_peaks)
    #plt.figure()
    #plt.plot(profile)
    #plt.plot(index_of_peaks,peak_heights,'x')

    # Find l,r edges
    centers = np.array([0]*len(index_of_peaks))
    fwhms = np.array([0]*len(index_of_peaks))

    l_edges = []
    r_edges = []
    i = 0
    for peak, peak_height in zip(index_of_peaks, peak_heights):
        maximum = peak_height
        minimum = 0 # Something else??
        half_max = maximum/2
        #locate left edge
        l_edge_found = False
        search_idx = peak
        while not l_edge_found:
            value = profile[search_idx]
            #print(value)
            if value <= half_max:
                left_edge = search_idx
                l_edges.append(left_edge)
                l_edge_found=True
            search_idx -= 1
        #locate right edge
        r_edge_found = False
        search_idx = peak
        while not r_edge_found:
            value = profile[search_idx]
            if value <= half_max:
                right_edge = search_idx
                r_edges.append(right_edge)
                r_edge_found=True
            search_idx += 1
        centers[i] = int((right_edge-left_edge)/2)+left_edge # Given in bins
        fwhms[i] = right_edge-left_edge # FWHM in bins
        i += 1
    if plot_found_bunches:

        plt.figure()
        plt.plot(profile)
        plt.plot(index_of_peaks,peak_heights,'x')
        for l_edge in l_edges:
            plt.axvline( x=l_edge,color='g')
        for r_edge in r_edges:
            plt.axvline( x=r_edge, color='r')
        for center in centers:
            plt.axvline( x=center, color='b')
    
    if rel:
        # Normalize relative to current fwhms
        fwhms = fwhms/np.max(fwhms)
    else:
        # Normalize with global constant
        fwhms =fwhms/52 # Fixed division to scale bunch lengths similarly as simulated data. At 0,0 offset, blengths of around 0.85. TESTING!!
    bunches = [] # list of individual bunches
    for center in centers:
        b = profile[center-bunch_width:center+bunch_width]
        bunches.append(b)
    if intensities:
        ints = np.zeros(len(index_of_peaks))
        for i, (l, r) in enumerate(zip(l_edges, r_edges)):
            ints[i] = np.sum(profile[l:r])
        return bunches, fwhms, ints
    return bunches, fwhms

def isolate_bunches_from_dm_profile_tri(profile, plot_found_bunches=False, bunch_width = 20, intensities=False, rel=False):
    index_of_peaks, peak_heights = peaks(profile, distance=50)
    # print(index_of_peaks)
    #plt.figure()
    #plt.plot(profile)
    #plt.plot(index_of_peaks,peak_heights,'x')

    # Find l,r edges
    centers = np.array([0]*len(index_of_peaks))
    fwhms = np.array([0]*len(index_of_peaks))

    l_edges = []
    r_edges = []
    i = 0
    for peak, peak_height in zip(index_of_peaks, peak_heights):
        maximum = peak_height
        minimum = 0 # Something else??
        half_max = maximum/2
        #locate left edge
        l_edge_found = False
        search_idx = peak
        while not l_edge_found:
            value = profile[search_idx]
            #print(value)
            if value <= half_max:
                left_edge = search_idx
                l_edges.append(left_edge)
                l_edge_found=True
            search_idx -= 1
        #locate right edge
        r_edge_found = False
        search_idx = peak
        while not r_edge_found:
            value = profile[search_idx]
            if value <= half_max:
                right_edge = search_idx
                r_edges.append(right_edge)
                r_edge_found=True
            search_idx += 1
        centers[i] = int((right_edge-left_edge)/2)+left_edge # Given in bins
        fwhms[i] = right_edge-left_edge # FWHM in bins
        i += 1
    if plot_found_bunches:

        plt.figure()
        plt.plot(profile)
        plt.plot(index_of_peaks,peak_heights,'x')
        for l_edge in l_edges:
            plt.axvline( x=l_edge,color='g')
        for r_edge in r_edges:
            plt.axvline( x=r_edge, color='r')
        for center in centers:
            plt.axvline( x=center, color='b')
    
    if rel:
        # Normalize relative to current fwhms
        fwhms = fwhms/np.max(fwhms)
    else:
        # Normalize with global constant
        fwhms =fwhms/52 # Fixed division to scale bunch lengths similarly as simulated data. At 0,0 offset, blengths of around 0.85. TESTING!!
    bunches = [] # list of individual bunches
    for center in centers:
        b = profile[center-bunch_width:center+bunch_width]
        bunches.append(b)
    if intensities:
        ints = np.zeros(3)
        for i, (l, r) in enumerate(zip(l_edges, r_edges)):
            ints[i] = np.sum(profile[l:r])
        return bunches, fwhms, ints
    return bunches, fwhms

# Calculate criterion as a loss
def profile_reward_tri(profile, verbose=False, using_old_dataset=False):
    # Single out each bunch. 400 bins and three bunches, initial bucket 308ns/3 ~ 102.6667
    #### Not centered well on bunches with ref voltage dataset. Switching to function used in vpc, loss_function_tri
    if using_old_dataset: # Old dataset simulated using different voltage program: Center of bunches in different location
        b1 = profile[46:149]
        b2 = profile[149:252]
        b3 = profile[252:355]
    else:
        b1 = profile[71:131] # Center 101
        b2 = profile[169:229] # Center 199
        b3 = profile[267:327] # Center 297
    #MSE between bunch 1 and bunch 2 profiles
    mse12 = np.mean((b1-b2)**2)
    #MSE between bunch 1 and bunch 3 profiles  
    mse13 = np.mean((b1-b3)**2)
    #MSE between bunch 2 and bunch 3 profiles
    mse23 = np.mean((b2-b3)**2)
    if verbose:
        print(f"MSE 12: {mse12}")
        print(f"MSE 13: {mse13}")
        print(f"MSE 23: {mse23}")
        plt.figure()
        plt.plot(b1)
        plt.plot(b2)
        plt.plot(b3)

    # Sum all MSE:s to get a single loss
    loss = mse12+mse13+mse23
    reward = -loss
    return reward

def profile_reward_quad(profile, verbose=False):
    # Single out each bunch. 400 bins and three bunches, initial bucket 308ns/3 ~ 102.6667
    b1 = profile[:50]
    b2 = profile[50:100]
    b3 = profile[100:150]
    b4 = profile[150:]

    #MSE between bunch 1 and bunch 2 profiles
    mse12 = np.mean((b1-b2)**2)
    mse13 = np.mean((b1-b3)**2)
    mse14 = np.mean((b1-b4)**2)

    mse23 = np.mean((b2-b3)**2)
    mse24 = np.mean((b2-b4)**2)

    mse34 = np.mean((b3-b4)**2)

    # if verbose:
    #     print(f"MSE 12: {mse}")


    # Sum all MSE:s to get a single loss
    loss = (mse12 + mse13 + mse14 + mse23 + mse24 + mse34)/6
    reward = -loss
    return reward


# Some testing functions to check implementations

def test_spaces(obs_space, act_space):
    action_space = gym.spaces.Box(
                                low = np.array([-1,]),# Fill in the _ with your lowest action. Dimensions of array correlate with dimensions of actions.
                                high = np.array([1,]),# Fill in the _ with your highest action.
                                shape=(1,),           # Fill in the _ with the dimensions of your action space.
                                dtype=np.float32)


    ### Define what the observations are to be expected
    observation_space = gym.spaces.Box(
                                low = np.array([-1,-1,-1,-1]), # Fill in the __ with your observation settings.
                                high = np.array([1,1,1,1]),
                                shape=(4,),
                                dtype=np.float32)
    obs_good = True
    act_good = True
    try:
        assert obs_space == observation_space
    except AssertionError:
        print(f'Observation space not as expected.')
        print(f'Expected: {observation_space}, received {obs_space}')
        obs_good=False
    try:    
        assert act_space == action_space
    except AssertionError:
        print(f'Action space not as expected.')
        print(f'Expected: {action_space}, received {act_space}')
        act_good=False
    if act_good and obs_good:
        print(f'Observation and action space as expected, good job!')
