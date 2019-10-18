# from build_rlbench_env import RLBenchEnv
from build_reach_env_dense_reward import RLBenchEnv
import numpy as np
from buffer import ReplayBuffer
from sac import SAC_Trainer
import torch
import argparse
import time
import matplotlib.pyplot as plt


GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()

def plot(rewards):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2.png')
    # plt.show()
    plt.clf()

state_types = ['joint_positions',
                'left_shoulder_rgb',
                'right_shoulder_rgb']
env=RLBenchEnv('ReachTarget', state_type_list = state_types )

# hyper-parameters for RL training
max_episodes  = 1000
max_steps   = 30
frame_idx   = 0
batch_size  = 32
explore_steps = 0  # for random action sampling in the beginning of training
update_itr = 1
AUTO_ENTROPY=True
DETERMINISTIC=False
hidden_dim = 64
rewards     = []
model_path = './model/sac_v2'
action_range = 0.5

action_dim=env.action_space.shape[0]
state_space=env.observation_space

replay_buffer_size = 1e6
replay_buffer = ReplayBuffer(replay_buffer_size)
sac_trainer = SAC_Trainer(replay_buffer, action_dim, state_space, hidden_dim, action_range=action_range, device=device)

if args.train:
    # training loop
    for eps in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if frame_idx > explore_steps:
                action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC, device=device)
            else:
                action = sac_trainer.policy_net.sample_action()
    
            next_state, reward, done, _ = env.step(action)

            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            frame_idx += 1
            
            
            if len(replay_buffer) > batch_size:
                for i in range(update_itr):
                    _=sac_trainer.update(batch_size, device, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*action_dim)

            if done:
                break

        if eps % 20 == 0 and eps>0: # plot and model saving interval
            plot(rewards)
            sac_trainer.save_model(model_path)
        print('Episode: ', eps, '| Episode Reward: ', episode_reward)
        rewards.append(episode_reward)
    sac_trainer.save_model(model_path)

if args.test:
    sac_trainer.load_model(model_path)
    for eps in range(10):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = sac_trainer.policy_net.get_action(state, deterministic = DETERMINISTIC, device=device)
            next_state, reward, done, _ = env.step(action)

            episode_reward += reward
            state=next_state

        print('Episode: ', eps, '| Episode Reward: ', episode_reward)


print('Done')
env.shutdown()


