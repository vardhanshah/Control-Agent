import numpy as np
import random
from collections import deque
import os


class memory:

    def __init__(self, size, batch_size, frame_shape, stack_size, state_shape):

        self.size = size
        self.batch_size = batch_size
        self.frame_shape = frame_shape
        self.stack_size = stack_size
        self.state_shape = state_shape
        self.frames = np.empty((self.size,*self.frame_shape),dtype=np.uint8)
        self.actions = np.empty((self.size),dtype=np.int)
        self.rewards = np.empty((self.size),dtype=np.float32)
        self.dones = np.empty((self.size),dtype=np.bool)

        self.pointer = 0
        self.occupied = 0

    def add(self, action, frame, reward, done):

        self.actions[self.pointer] = action
        self.frames[self.pointer, ...] = frame
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.size
        self.occupied = min(self.occupied+1,self.size)

        self.indices = np.empty((self.batch_size),dtype=np.int)
        self.states = np.empty((self.batch_size,*self.state_shape),dtype=np.float)
        self.new_states = np.empty((self.batch_size,*self.state_shape),dtype=np.float)

    def _get_state(self,index):
        if self.occupied is 0:
            raise ValueError("Replay buffer is empty")
        return self.frames[index-self.stack_size+1 : index+1, ...]

    def _get_indices(self):

        for i in range(self.batch_size):
            while True:
                index = random.randint(self.stack_size,self.occupied-1)
                if index >=self.pointer and index - self.stack_size <= self.pointer:
                    continue
                if self.dones[index-self.stack_size:index].any():
                    continue
                break
            self.indices[i] = index

    def sample(self):
        if self.occupied < self.stack_size:
            raise ValueError('not enough memory to get sample')

        self._get_indices()

        for i in range(self.batch_size):
            self.states[i] = self._get_state(self.indices[i]-1)
            self.new_states[i] = self._get_state(self.indices[i])

        return self.states,self.actions[self.indices],self.rewards[self.indices],self.new_states,self.dones[self.indices]