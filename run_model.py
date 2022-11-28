#%%
from stable_baselines3.common.buffers import ReplayBuffer
import sys
sys.path.append(r'C:\Users\jwulff\cernbox\workspaces\RL-007-RFinPS\optimising-rf-manipulations-in-the-ps-through-rl')
import gym
import numpy as np
from numpy.core.fromnumeric import mean, shape
from numpy.core.function_base import linspace
#from quadsplit_class_bunch_lengths import QuadSplitContinuousBL
from quad_env_rel_bl_int_42 import QuadEnvBLInt
#from quad_env_profiles_cnn import QuadEnvProfilesCNN
#from quad_env_profiles import QuadEnvProfiles
from stable_baselines3 import SAC
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3 import her
#from stable_baselines3.common.policies import FeedForwardPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
#from stable_baselines3 HerReplayBuffer
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from scipy import interpolate
import os
import time


model_name = r"best_model"
model = SAC.load("./RL_logs/{}".format(model_name))

env = QuadEnvBLInt()

obs = env.reset()
performance = [0,0]
steps_per_episode = []
predicted_phase_error_when_done = []

start_diff_estimates = []
end_diff_estimates = []
initial_profiles = []
final_profiles = []
counter = 0
render_opt = True

for episode in range(100):
    done=False
    first_step = True
    #print(obs)
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        #print(action*10)
        obs, rewards, done, info = env.step(action)

        if first_step:
            start_diff_estimates.append(info['diff_estimate'])
            if counter < 5:
                    initial_profiles.append(info['profile'])
        first_step=False
        #print(obs)
        if render_opt:
            env.render() # Use if you want to observe the agent
            if done:
                time.sleep(1) # allow some time to see complete trajectory before starting next episode.
        if done:
            
            print(f"Took {info['steps']} steps before terminating test")
            print(f"Info {info['success']}, {info['steps']}, {info['initial_phase']}, {info['phase_corr']}")
            steps_per_episode.append(info['steps'])
            end_diff_estimates.append(info['diff_estimate'])
            #print(f"Info {info}")
            steps_per_episode.append(info['steps'])
            if info['success'] == True:
                performance[0] += 1
            else:
                performance[1] += 1
            
            predicted_phase_error_when_done.append(info['initial_phase'] + info['phase_corr'])
            if counter < 5:
                final_profiles.append(info['profile'])
            counter += 1
            obs = env.reset()
print(f"Succesful optimizations: {performance[0]}")
print(f"Unsuccesful optimizations: {performance[1]}")
print(f"Accuracy: {performance[0]/(performance[0]+performance[1])}")
print(f"Mean episode length: {np.sum(steps_per_episode)/len(steps_per_episode)}")
print(f"Max episode length: {np.max(steps_per_episode)}, Min episode length: {np.min(steps_per_episode)}")
p42_err = []
p84_err = []
for entry in predicted_phase_error_when_done:
    p42_err.append(entry)
#    p84_err.append(entry[1])

# Create saving directory
if not os.path.exists('RL_plots/{}'.format(model_name)):
    os.makedirs('RL_plots/{}'.format(model_name))

# plt.plot(p42_err, p84_err, '.')
# plt.title('{}: Scatter plot of correction errors'.format(model_name))
# plt.xlabel('p42 prediction error')
# plt.ylabel('p84 prediction error')
# plt.xlim(-5,5)
# plt.ylim(-5,5)
# plt.savefig('RL_plots/{}/scatter_plot.png'.format(model_name))
# plt.show()

plt.figure()
plt.title('Start and end Agent Criterion')
plt.plot(np.linspace(0,len(start_diff_estimates),len(start_diff_estimates)+1),np.ones(len(start_diff_estimates)+1)*0.0003, 'k--', label='Stop Criterion')
plt.plot(start_diff_estimates, 'r', label='start')
plt.plot(end_diff_estimates, 'g', label='End')
plt.xlabel('Episodes')
plt.legend()
plt.savefig('RL_plots/{}/init_end_criterion.png'.format(model_name))
plt.show()

plt.figure()
plt.suptitle('Five initial and final profiles, {}'.format(model_name))
plt.subplot(251)
plt.title('%.5f'%start_diff_estimates[0])
plt.plot(initial_profiles[0])
plt.subplot(252)
plt.title('%.5f'%start_diff_estimates[1])
plt.plot(initial_profiles[1])
plt.subplot(253)
plt.title('%.5f'%start_diff_estimates[2])
plt.plot(initial_profiles[2])
plt.subplot(254)
plt.title('%.5f'%start_diff_estimates[3])
plt.plot(initial_profiles[3])
plt.subplot(255)
plt.title('%.5f'%start_diff_estimates[4])
plt.plot(initial_profiles[4])
plt.subplot(256)
plt.title('%.5f'%end_diff_estimates[0])
plt.plot(final_profiles[0])
plt.subplot(257)
plt.title('%.5f'%end_diff_estimates[1])
plt.plot(final_profiles[1])
plt.subplot(258)
plt.title('%.5f'%end_diff_estimates[2])
plt.plot(final_profiles[2])
plt.subplot(259)
plt.title('%.5f'%end_diff_estimates[3])
plt.plot(final_profiles[3])
plt.subplot(2,5,10)
plt.title('%.5f'%end_diff_estimates[4])
plt.plot(final_profiles[4])
plt.savefig('RL_plots/{}/init_end_profiles.png'.format(model_name))
plt.show()
