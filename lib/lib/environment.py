from collections import deque
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
import gym



class AtariEnvironment:

    def __init__(self,config):
        self.env_name = config["name"]
        self.env = gym.make(self.env_name)
        self.no_actions = self.env.action_space.n
        self.stack_size = config.get("stack_size",4)
        self.frame_shape = config.get("frame_shape",[84,84])
        self.state_shape = self.frame_shape + [self.stack_size]
        self.stacked_frames = deque([],maxlen=self.stack_size)
        print("Environment Initialized")

    def step(self,action):
        next_frame,reward,done,info =  self.env.step(action)
        if done:
            next_state = self.preprocess_state(np.zeros(self.frame_shape))
        else:
            next_state = self.preprocess_state(next_frame)
        return next_state,self.process_reward(reward),done,info

    def reset(self):
        return self.preprocess_state(self.env.reset(),True)

    def render(self,give_array=None,show=True):
        if show:
            self.env.render()
        if give_array is not None:
            return self.env.render("rgb_array")

    def process_reward(self,reward):
        return np.sign(reward)

    def __preprocess_frame(self,frame):

        gray = rgb2gray(frame)
        normalized_frame = gray/255.0
        preprocessed_frame = transform.resize(normalized_frame,self.frame_shape)
        return preprocessed_frame

    def preprocess_state(self, state, is_new_episode=False):
        frame = self.__preprocess_frame(state)

        if is_new_episode:
            for i in range(self.stack_size):
                self.stacked_frames.append(frame)
            preprocessed_state = np.stack(self.stacked_frames, axis=2)

        else:
            self.stacked_frames.append(frame)
            preprocessed_state = np.stack(self.stacked_frames,axis=2)
        return preprocessed_state

class ClassicControlEnvironment:

    def __init__(self,config):
        self.env_name = config["name"]
        self.env = gym.make(self.env_name)
        self.no_actions = self.env.action_space.n
        self.frame_shape = list(self.env.observation_space.shape)
        self.state_shape = self.frame_shape
        print("Environment Initialized")

    def step(self,action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def render(self,give_array=None,show=False):
        if show:
            self.env.render()
        if give_array is not None:
            return self.env.render("rgb_array")




