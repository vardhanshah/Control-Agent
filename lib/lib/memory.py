import numpy as np
from collections import deque
import os
class memory:
    id = 0
    def __init__(self,config,env_name):
        max_size = config.get("memory_size",1e6)
        path = config.get("path",None)
        self.buffer = deque(maxlen=int(max_size))
        memory.id += 1
        if not path:
            path = os.getcwd() + "/replay_memory/{}".format(env_name)
        self.path = path
        self.filename = "/memory{}.npy".format(memory.id)
        os.makedirs(self.path,exist_ok=True)
        print("memory initialized !")
        if config.get("restore",False):
            self.load()
    def add(self,experience):
        self.buffer.append(experience)

    def sample(self,batch_size):
        buffer_size = len(self.buffer)

        index = np.random.choice(
            np.arange(buffer_size),
            size=batch_size,
            replace=False
        )
        return [self.buffer[i] for i in index]
    def save(self):
        file = self.path+self.filename
        np.save(file,np.asarray(self.buffer))
        print("replay_buffer stored at {}".format(file))
    def load(self):
        try:
            self.buffer = deque(np.load(self.path+self.filename))
            print("replay memory restored !")
            return True
        except:
            print("no replay_memory exist")
            return False

