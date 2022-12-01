#%%
from os import name
import gym
import numpy as np
from quad_env_rel_bl_int_42 import QuadEnvBLInt
# from quad_env_profiles_cnn import QuadEnvProfilesCNN
# from quad_env_profiles import QuadEnvProfiles

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
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Save a checkpoint every 1000 steps
# Separate evaluation env
eval_env = QuadEnvBLInt()
#eval_env = QuadEnvProfilesCNN()
# Use deterministic actions for evaluation
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=101, verbose=1)
#for reward_scale in range(20):
#    ent_coef = 1/(reward_scale+1)
#callback_on_new_best=callback_on_best,
eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best,  best_model_save_path='./RL_logs/',
    log_path='./RL_logs/', eval_freq=500,
    deterministic=True, render=False)


env = QuadEnvBLInt()

seed = 7
np.random.seed(seed)
env.seed(seed)
log_dir = './'


# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))

#policy_kwargs = dict(net_arch=[64,64])# policy_kwargs=policy_kwargs, buffer_size=5000,
#learning_rate=0.001,
# action_noise=action_noise,
# ent_coef=ent_coef,
ent_coef = 1/5
model = SAC("MlpPolicy", env, verbose=1, ent_coef=ent_coef, learning_starts=100, tensorboard_log="./hands-on_rl_tensorboard")
print("Starting training...")
model.learn(total_timesteps=1000, callback=eval_callback, log_interval=5, tb_log_name="hands_on_rl")
print("Completed training")
model.save("doublesplit_cnn_fixed")
env = model.get_env()

# %%
