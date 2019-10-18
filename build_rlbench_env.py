""" 
Wrap the rlbench env into openai gym env
"""
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *
import numpy as np
import gym
from gym import spaces
import sys
# Don't forget to add: export PYTHONPATH=PATH_TO_YOUR_LOCAL_RLBENCH_REPO

# list of state types
state_types = [ 'left_shoulder_rgb',
                'left_shoulder_depth',
                'left_shoulder_mask',
                'right_shoulder_rgb',
                'right_shoulder_depth',
                'right_shoulder_mask',
                'wrist_rgb',
                'wrist_depth',
                'wrist_mask',
                'joint_velocities',
                'joint_velocities_noise',
                'joint_positions',
                'joint_positions_noise',
                'joint_forces',
                'joint_forces_noise',
                'gripper_pose',
                'gripper_touch_forces',
                'task_low_dim_state']

image_types=[ 'left_shoulder_rgb',
                'left_shoulder_depth',
                'left_shoulder_mask',
                'right_shoulder_rgb',
                'right_shoulder_depth',
                'right_shoulder_mask',
                'wrist_rgb',
                'wrist_depth',
                'wrist_mask',]

class RLBenchEnv():
    """ make RLBench env to have same interfaces as openai.gym """
    def __init__(self, task_name, state_type_list=['left_shoulder_rgb']):
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=False)
        self.env.launch()
        try: 
            self.task = self.env.get_task(getattr(sys.modules[__name__], task_name))
        except:
            raise NotImplementedError

        _, obs = self.task.reset()
        if len(state_type_list)>0:
            self.observation_space=[]
            for state_type in state_type_list:
                state=getattr(obs, state_type)
                self.observation_space.append(spaces.Box(low=-np.inf, high=np.inf, shape=state.shape))
        else:
            raise ValueError('No State Type!')
        self.action_space =  spaces.Box(low=-1.0, high=1.0, shape=(action_mode.action_size,), dtype=np.float32)
        
        self.state_type_list = state_type_list
        self.spec = Spec(task_name)

    def seed(self, seed_value):
        # set seed as in openai.gym env
        pass 

    def render(self):
        # render the scene
        pass

    def _get_state(self, obs):
        if len(self.state_type_list)>0:
            state=[]
            for state_type in self.state_type_list:
                if state_type in image_types:
                    image = getattr(obs, state_type)
                    image=np.moveaxis(image, 2, 0)  # change (H, W, C) to (C, H, W) for torch
                    state.append(image)
                else:
                    state.append(getattr(obs, state_type))
        else:
            raise ValueError('State Type Not Exists!')
        return state

    def reset(self):
        descriptions, obs = self.task.reset()
        return self._get_state(obs)

    def step(self, action):
        obs_, reward, terminate = self.task.step(action)  # reward in original rlbench is binary for success or not
        return self._get_state(obs_), reward, terminate, None


    def close(self):
        self.env.shutdown()

        
class Spec():
    """ a fake spec """
    def __init__(self, id_name):
        self.id = id_name

