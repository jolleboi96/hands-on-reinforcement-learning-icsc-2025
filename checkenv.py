from stable_baselines3.common.env_checker import check_env
from quad_env_rel_bl_int_42 import QuadEnvBLInt
env = QuadEnvBLInt()

result = check_env(env)