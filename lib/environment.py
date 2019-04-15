from collections import deque
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
import gym


class AtariEnvironment:

    def __init__(self,config):
        self.name = config["name"]
        self.type = config["type"]
        self.env = gym.make(self.name)
        self.dtype = self.env.reset().dtype
        self.no_actions = self.env.action_space.n
        self.stack_size = config.get("stack_size",1)
        self.frame_shape = config.get("frame_shape",self.env.observation_space.shape)
        self.state_shape = [self.stack_size,*self.frame_shape]
        self._stacked_frames = deque([],maxlen=self.stack_size)
        self.state = None #np.zeros(self.state_shape)
        self.lives = 0
        print("Environment Initialized")

    def step(self,action):
        next_frame,reward,done,info = self.env.step(action)

        done_replay = done
        if info.get('ale.lives',0) < self.lives:
            done_replay = True

        self.lives = info.get('ale.lives',0)

        if done:
            self.state = None
        else:
            self.state = self.preprocess_state(next_frame)

        return next_frame,self.process_reward(reward), done, done_replay

    def reset(self):
        self.lives = 0
        frame = self.env.reset()
        self.state = self.preprocess_state(frame,True)
        return frame

    def render(self,mode="human"):
        return self.env.render(mode)

    def process_reward(self,reward):
        return reward
        # return np.sign(reward)

    def preprocess_frame(self,frame):
        if self.type == "atari":
            gray = rgb2gray(frame)
            if self.name == "SpaceInvaders-v0" or self.name == "Seaquest-v0":
                cropped = gray[10:-15,:-15]
            elif self.name == "Enduro-v0":
                cropped = gray[40:-55,20:]
            elif self.name == "Pong-v0":
                cropped = gray[35:-15,:]
            else:
                cropped = gray
            return np.array(transform.resize(cropped,self.frame_shape)*255.0,dtype=np.uint8)
        return frame

    def preprocess_state(self, frame, is_new_episode=False):
        frame = self.preprocess_frame(frame)

        if is_new_episode:
            for i in range(self.stack_size):
                self._stacked_frames.append(frame)
            preprocessed_state = np.stack(self._stacked_frames)

        else:
            self._stacked_frames.append(frame)
            preprocessed_state = np.stack(self._stacked_frames)
        return preprocessed_state


    def close(self):
        self.env.close()


