import time

import numpy as np
import torch
from sb3_contrib import TRPO
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EveryNTimesteps, CheckpointCallback

from experiments.ReinforcementLearning.env import create_env
from utils.rl_utils import create_validation_points
from utils.sb3_callbacks import FigureRecorderCallback, TestComparisonCallback, ComparisonCallback

device = torch.device("cpu")


def train():
    env = create_env(max_episode_steps=1)

    fig_callback = FigureRecorderCallback(env)
    ntimestep_callback = EveryNTimesteps(n_steps=500, callback=fig_callback)

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./models/", name_prefix="trpo_model")

    case_nr = 1
    num_val_points = 100
    eval_env = create_env(max_episode_steps=100)
    val_points = create_validation_points(case_nr, num_val_points, map_size=(128, 128))
    eval_callback = EveryNTimesteps(n_steps=1000, callback=TestComparisonCallback(eval_env, val_points=val_points))
    eval_callback_2 = EveryNTimesteps(n_steps=1000, callback=ComparisonCallback(eval_env))

    model = TRPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log="./turbine_env/", learning_rate=5e-4)
    model.learn(total_timesteps=200000, progress_bar=True, tb_log_name="TRPO", reset_num_timesteps=False, callback=[checkpoint_callback, ntimestep_callback, eval_callback, eval_callback_2])
    model.save("TRPOTurbineEnvModel")

def predict():
    env = create_env()
    model = A2C.load("TRPOTurbineEnvModel")
    times = []

    for i in range(10):
        obs, info = env.reset()
        start = time.time()
        action, _states = model.predict(obs)
        obs, rewards, dones, truncations, info = env.step(action)
        end = time.time()
        times.append(end-start)
        env.render()
    print(np.mean(times))


if __name__ == "__main__":
    # train()
    predict()