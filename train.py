import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import numpy as np
import itertools

import torch
from torch.utils.tensorboard import SummaryWriter

from envs.register import register_custom_envs

from algo.utils import VideoRecorder, create_directory, copy_files_and_directories
from algo.sac import SAC
from algo.buffer import ReplayMemory
from algo.model_eigen import Psi
from algo.arguments import parser_args


# Load arguments
args = parser_args()

# Logging experiment
exp_num_directory = os.path.join("results", args.exp_name)
create_directory(exp_num_directory)

# Sub-directories
log_dir   = os.path.join(exp_num_directory, "runs")
video_dir = os.path.join(exp_num_directory, "video")
ckpt_dir  = os.path.join(exp_num_directory, "checkpoints")

create_directory(log_dir)
create_directory(video_dir)
create_directory(ckpt_dir)

paths_to_copy = [
    "algo",
    "envs",
    # Add more files or directories as needed
]

copy_files_and_directories(paths_to_copy, exp_num_directory)

# Environment
register_custom_envs()

if args.env_name == "LunarLander-safety":
    env = gym.make(args.env_name, render_mode="rgb_array")
else:
    env = gym.make(args.env_name, render_mode="rgb_array", camera_id=0)

# Device
device = torch.device("cuda" if args.cuda else "cpu")

# For reproduce
env.action_space.seed(args.seed)   
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# For video
video = VideoRecorder(dir_name = video_dir)

# Tensorboard
writer = SummaryWriter(
    os.path.join(
        log_dir,
        f"eigen_{args.env_name}"
    )
)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
episode_idx = 0
updates = 0

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# Psi
psi = Psi(env.observation_space.shape[0] + env.action_space.shape[0], args).to(device)

# Check action dim
print("env_name :", args.env_name)
print("max_episode_step :", env._max_episode_steps)
print("state_dim :", env.observation_space.shape[0])

# Training Loop
for i_epoch in itertools.count(1):
    for i_episode in range(args.episodes_per_epoch):
        episode_reward = 0
        episode_steps = 0
        episode_idx += 1
        episode_trajectory = []

        done = False
        state, info = env.reset()

        info['safety'] = 1.0
        safety_current = info['safety']
    
        for i in range(env._max_episode_steps):
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy
            
            next_state, reward, done, trunc, info = env.step(action) 
            safety_next = info['safety'] 
            
            # update
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            mask = 1 if episode_steps == env._max_episode_steps else float(not done)
            memory.push(state, safety_current, action, reward, next_state, safety_next, mask)

            state = next_state
            safety_current = safety_next

        writer.add_scalar('train/gt_reward', episode_reward, episode_idx)

        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}" \
              .format(episode_idx, total_numsteps, episode_steps, round(episode_reward, 2)))


    ## Alternating Update Method
    if (len(memory) > args.batch_size):
        # Number of updates per step in environment
        for i in range(args.gradient_steps_per_epoch):

            # Update parameters of SAC networks
            critic_1_loss, critic_2_loss, policy_loss, lambda_loss, lambda_value, ent_loss, alpha = agent.update_parameters(psi, memory, updates, args)

            # SAC Loss
            writer.add_scalar('loss/critic_1', critic_1_loss, updates)
            writer.add_scalar('loss/critic_2', critic_2_loss, updates)
            writer.add_scalar('loss/policy', policy_loss, updates)
            writer.add_scalar('loss/lambda', lambda_loss, updates)
            writer.add_scalar('loss/entropy_loss', ent_loss, updates)
            writer.add_scalar('entropy_temperature/alpha', alpha, updates)

            # Update parameters of Psi networks
            psi_loss, psi_mean, eigen_value = psi.update_parameters(agent, memory, args)
            
            # Encoder Loss
            writer.add_scalar('loss/psi_loss', psi_loss, updates)
            writer.add_scalar('value/avg_psi', psi_mean, updates)
            writer.add_scalar('value/eigen', eigen_value, updates)
            writer.add_scalar('value/lambda', lambda_value, updates)
            
            updates += 1


    # Evaluate policy
    if episode_idx % (args.episodes_per_epoch*args.eval_epoch_ratio) == 0:
        avg_pseudo_reward = 0.
        avg_reward = 0.
        avg_step = 0.

        episode_rewards = []
        episode_steps_list = []
        first_unsafe_steps = []

        num_episodes = 5
        
        ######
        video.init(enabled=True)
        ######

        for i in range(num_episodes): 

            episode_steps = 0
            episode_reward = 0

            done = False
            state, info = env.reset()

            info['safety'] = 1.0
            safety_current = info['safety']
            first_unsafe_step = None

            for i in range(env._max_episode_steps):
                action = agent.select_action(state, evaluate=True)
                next_state, reward, done, trunc, info = env.step(action) 

                safety_current = info['safety']

                if safety_current == 0.0 and first_unsafe_step is None:
                    first_unsafe_step = i+1  

                episode_steps += 1
                episode_reward += reward
                
                state = next_state

                #######
                video.record(env.render())
                #######

            if first_unsafe_step is None:
                first_unsafe_step = env._max_episode_steps

            episode_rewards.append(episode_reward)
            episode_steps_list.append(episode_steps)
            first_unsafe_steps.append(first_unsafe_step)

        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_step = np.mean(episode_steps_list)
        avg_first_unsafe_step = np.mean(first_unsafe_steps)
        std_first_unsafe_step = np.std(first_unsafe_steps)


        # Save video
        video.save('test_{}.mp4'.format(episode_idx))
        video.init(enabled=False)
        
        # For tensorboard
        writer.add_scalar('test/avg_gt_reward', avg_reward, episode_idx)
        writer.add_scalar('test/std_gt_reward', std_reward, episode_idx)
        writer.add_scalar('test/avg_first_unsafe_step', avg_first_unsafe_step, episode_idx)
        writer.add_scalar('test/std_first_unsafe_step', std_first_unsafe_step, episode_idx)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}, Avg. step: {}"\
              .format(num_episodes, round(avg_reward, 2), round(avg_step, 2)))
        print("----------------------------------------")
        
    if episode_idx % (args.episodes_per_epoch*args.save_epoch_ratio) == 0:
        agent.save_checkpoint(args,"{}".format(episode_idx))
        psi.save_checkpoint(args,"{}".format(episode_idx))
        
    if episode_idx > args.num_episodes:
        break   
    
env.close()

