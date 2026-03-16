import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import numpy as np
import torch

import imageio.v2 as imageio
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.use("Agg")

from envs.register import register_custom_envs

from algo.sac import SAC
from algo.model_eigen import Psi
from algo.utils import create_directory
from algo.arguments import parser_args


def psi_plot_image(history, width, height):
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(history, color="orange", linewidth=2)
    ax.set_title("psi")
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    img = img[:, :, :3].copy()  # drop alpha channel
    return img


env_name = "Ant-ball-v5"    # Halfcheetah-run-low-v5, Hopper-run-high-v5, Ant-ball-v5, LunarLander-safety

exp_name = "exp0101-eigen-ant"
num_episode = 80000

# Load arguments
args = parser_args()

# Logging experiment
exp_num_directory = os.path.join("results", exp_name)
create_directory(exp_num_directory)

# Sub-directories
log_dir   = os.path.join(exp_num_directory, "evals")

create_directory(log_dir)
frames_dir = os.path.join(log_dir, "frames")
create_directory(frames_dir)

# Environment
register_custom_envs()

if env_name == "LunarLander-safety":
    env = gym.make(env_name, render_mode="rgb_array")
else:
    env = gym.make(env_name, render_mode="rgb_array", camera_id=0)

# Device
device = torch.device("cuda" if args.cuda else "cpu")

# For reproduce
env.action_space.seed(args.seed)   
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
agent.load_checkpoint(f"results/{exp_name}/checkpoints/sac_checkpoint_{env_name}_{num_episode}")

# Psi
psi = Psi(env.observation_space.shape[0] + env.action_space.shape[0], args).to(device)
psi.load_checkpoint(f"results/{exp_name}/checkpoints/psi_checkpoint_{env_name}_{num_episode}")
# Check action dim
print("env_name :", env_name)
print("max_episode_step :", env._max_episode_steps)
print("state_dim :", env.observation_space.shape[0])


avg_pseudo_reward = 0.
avg_reward = 0.
avg_step = 0.

psi_value = []
first_unsafe_steps = []

video_path = os.path.join(log_dir, f"{exp_name}_eigen.mp4")
video_writer = imageio.get_writer(video_path, fps=100)


num_eval_episodes = 3

for i in range(num_eval_episodes): # Make sure that 'episodes' == 'num_intervals', so all cases can be evaluated.
    
    episode_steps = 0
    episode_reward = 0
    done = False
    state, info = env.reset()
    
    info['safety'] = 1.0
    safety_current = info['safety']
    
    first_unsafe_step = None
    
    psi_history = []

    for t in range(env._max_episode_steps):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, trunc, info = env.step(action) # Step
        safety_current = info['safety']
        
        if safety_current == 0.0 and first_unsafe_step is None:
            first_unsafe_step = t + 1  
            
        episode_steps += 1
        episode_reward += reward
        
        psi_in = np.concatenate([state, action], axis=0)
        psi_current = psi.forward_np(psi_in)
        psi_value.append(psi_current)
        psi_history.append(float(np.squeeze(psi_current)))
        
        state = next_state
        frame = env.render()
        if frame is not None:
            psi_img = psi_plot_image(psi_history, frame.shape[1], frame.shape[0])
            concat_frame = np.concatenate([frame, psi_img], axis=1)
            video_writer.append_data(concat_frame)
        #     frame_path = os.path.join(frames_dir, f"ep{i}_step{t:04d}.png")
        #     imageio.imwrite(frame_path, frame)
        if done or trunc:
            break
        
    if first_unsafe_step is None:
        first_unsafe_step = env._max_episode_steps
        
    avg_reward += episode_reward
    avg_step += episode_steps
    
    first_unsafe_steps.append(first_unsafe_step)
    
avg_reward /= num_eval_episodes
avg_step /= num_eval_episodes
avg_first_unsafe_step = sum(first_unsafe_steps) / len(first_unsafe_steps)



env.close()
video_writer.close()
