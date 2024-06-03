import numpy as np
from   scipy.integrate import solve_ivp
import scipy.sparse as sparse

import matplotlib.pyplot as plt

import gym
from   gym import spaces
from   numpy.core.function_base import linspace
# from sqlalchemy import true

from   stable_baselines3.common.env_util import make_vec_env
from   stable_baselines3 import PPO, SAC
from   stable_baselines3.common.utils import set_random_seed
from   stable_baselines3.sac.policies import MlpPolicy

from   stable_baselines3.common.vec_env import SubprocVecEnv

from   stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from   stable_baselines3.common.callbacks import CheckpointCallback
import math
from random import random
# import csv
import time
# from   torch import dsmm



from pybullet_env.BasicEnvironment import SoftRobotBasicEnvironment
import numpy as np


class SoftManipulatorEnv(gym.Env):
    def __init__(self,env_id) -> None:
        super(SoftManipulatorEnv, self).__init__()

        self.simTime = 0
        self._env_id  = env_id
        
        
        self._env = SoftRobotBasicEnvironment(moving_base=True,sphere_radius=0.02,_number_of_segment=2,gui=False if self._env_id>0 else True)
        base_link_shape = self._env._pybullet.createVisualShape(self._env._pybullet.GEOM_BOX, halfExtents=[0.05, 0.05, 0.03], rgbaColor=[0.6, 0.6, 0.6, 1])
        base_link_pos, base_link_ori = self._env._pybullet.multiplyTransforms([0,0,0.5], [0,0,0,1], [0,-0.0,0], [0,0,0,1])
        base_link_id    = self._env._pybullet.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=base_link_shape,
                                                            baseVisualShapeIndex=base_link_shape,
                                                            basePosition= base_link_pos , baseOrientation=base_link_ori)
        self._base_pos = np.array([0,0,0.5])
        self._base_ori = np.array([-np.pi/2,0,0])
        shape, ode_sol = self._env.move_robot_ori(action=np.array([0.0, 0.0, 0.0,
                                                             0.0, 0.0, 0.0]),
                                            base_pos=self._base_pos, base_orin = self._base_ori, camera_marker=False)
        
        self.reset()
            
        ### IK
        self.action_space = spaces.Box(low=np.array([-0.02,-0.02,
                                                     -0.02, -0.02,-0.02]),
                                       high=np.array([0.02,0.02,
                                                      0.03, 0.02,0.02]), dtype="float32")
        observation_bound = np.array([1, 1, 1]) # pos 
        self.observation_space = spaces.Box(low = -observation_bound, high = observation_bound, dtype="float32")
        
        ### FK
        # self.action_space = spaces.Box(low=np.array([-0.02,-0.02,0.0]), high=np.array([0.2,0.2,0.2]), dtype="float32")
        # observation_bound = np.array([np.inf, np.inf, np.inf]) # l uy ux 

        # self.observation_space = spaces.Box(low = -observation_bound, high = observation_bound, dtype="float32")
        
    

    def observe(self):        
        # ob      = ([self.ol ,self.ouy, self.oux])
        ob = self.desired_pos
        return ob

    def step(self, action):

        assert self.action_space.contains(action)

        self._shape, self._ode_sol = self._env.move_robot_ori(action=np.array([0.0, action[0], action[1],
                                                                         action[2], action[3], action[4]]),
                                            base_pos=self._base_pos, base_orin = self._base_ori, camera_marker=False)
        
        self.pos = self._shape[-1][:3]
        # self.distance = np.linalg.norm(self.pos-self.posPred)
        self.distance = np.linalg.norm(self.pos-self.desired_pos)

        reward = (math.exp(-(self.distance**2)/0.005))
        
        observation = self.observe()
        terminal = True

        info = {"rew":reward}
        return observation, reward, terminal, info


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


    def reset(self):
        
        des_x  = np.random.uniform(low=-0.08, high=0.08, size=(1,))
        des_y  = np.random.uniform(low=-0.08, high=0.08, size=(1,))
        des_z  = np.random.uniform(low= 0.26, high=0.32, size=(1,))
        self.desired_pos = np.squeeze(np.array((des_x,des_y,des_z)))
        
        

        # self.ol    = np.random.uniform(low= -0.03, high=0.04, size=(1,))[0]
        # self.ouy   = np.random.uniform(low=-0.015, high=0.015, size=(1,))[0]
        # self.oux   = np.random.uniform(low=-0.015, high=0.015, size=(1,))[0]
        
        # self.ol    = np.random.uniform(low= -0.01, high=0.01, size=(1,))[0]
        # self.ouy   = np.random.uniform(low=-0.005, high=0.005, size=(1,))[0]
        # self.oux   = np.random.uniform(low=-0.005, high=0.005, size=(1,))[0]
        
        # self.ol    = np.random.uniform(low= -0.005, high=0.005, size=(1,))[0]
        # self.ouy   = np.random.uniform(low=-0.003, high=0.003, size=(1,))[0]
        # self.oux   = np.random.uniform(low=-0.003, high=0.003, size=(1,))[0]
        
        
        
        if (self._env_id == 0): #Test env
            self._env._set_marker(self.desired_pos)
        
            print ("reset Env 0")
    
        observation = self.observe()
        
        return observation  # reward, done, info can't be included

    def close (self):
        print ("Environment is closing....")



    
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = SoftManipulatorEnv(env_id)
        #DummyVecEnv([lambda: CustomEnv()]) #gym.make(env_id)
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    set_random_seed(seed)
    return _init

if __name__ =="__main__":
    
    num_cpu_core = 2
    max_epc = 200000
    
    # from gym.envs.registration import register
    # register(
    #     id='SoftManipulatorEnv-v0',
    #     entry_point='custom_env:SoftManipulatorEnv',
    # )

    if (num_cpu_core == 1):
        sf_env = SoftManipulatorEnv()
    else:
        sf_env = SubprocVecEnv([make_env(i, i) for i in range(1, num_cpu_core)]) # Create the vectorized environment
    
    timestr   = time.strftime("%Y%m%d-%H%M%S")
    logdir    = "logs/learnedPolicies/log_"  + timestr

    model = SAC("MlpPolicy", sf_env, verbose=1, tensorboard_log=logdir)

    # model.load("./learnedPolicies/Model_20220608-174240.zip")
    # model = SAC.load("./learnedPolicies/Model_20220608-192046.zip", env = env, tensorboard_log=logdir)
    
    model.learn(total_timesteps=25000,log_interval=10)
    timestr   = time.strftime("%Y%m%d-%H%M%S")
    modelName = "logs/learnedPolicies/model_"+ timestr
    model.save(modelName)
    sf_env.close()
    print(f"finished. The model saved at {modelName}")
    
    
        
    # model = SAC.load("logs/learnedPolicies/model_20240603-001601.zip", env = sf_env)

    # obs = sf_env.reset()
    # timesteps = 5000
    # for i in range(timesteps):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = sf_env.step(action)
    #     #callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-99.0, verbose=1)
    #     #eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1)
    #     if done:
    #         obs = sf_env.reset()
    #     time.sleep(2)



    # env.test()
    # model = PPO("MlpPolicy", env,verbose=0,tensorboard_log=logdir)    
    # model.learn(total_timesteps = max_epc, log_interval=1)
   
    # print ("saving the learned policy")
    # model.save(modelName)
    #del model
    # sf_env.close()
    


   



