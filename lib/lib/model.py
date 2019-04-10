
from lib.memory import memory as mem
import random
import numpy as np
import os
class model:

    last_save = 1

    def __init__(self,config):
        self.env_type = config["environment"].get("type")
        global Environment
        if self.env_type == "atari":
            from lib.environment import AtariEnvironment as Environment
        elif self.env_type == "classic-control" or self.env_type == "classic":
            from lib.environment import ClassicControlEnvironment as Environment
        # elif self.env_type == "custom":

        self.env = Environment(config["environment"])
        self.env_name = self.env.env_name
        self.state_shape = self.env.state_shape
        self.no_actions = self.env.no_actions


        if config["agent"]["type"] == "dqn":
            global Agent
            from lib.dqn import Agent
        elif config["agent"]["type"] == "double_dqn":
            global Agent
            from lib.double_dqn import Agent
        elif config["agent"]["type"] == "target_dqn":
            global Agent
            from lib.target_dqn import Agent

        self.agent = Agent(self.state_shape,self.no_actions,config["agent"])
        self.agent_type = config["agent"]["type"]
        self.explore_start = config.get("explore_start",1.0)
        self.explore_stop = config.get("explore_stop",0.01)
        self.decay_rate = config.get("decay_rate",0.00001)
        self.explore_probability = self.explore_start
        self.max_steps = config.get("max_steps",None)
        self.max_steps_each_episode = config.get("max_steps_each_episode",None)
        self.max_episodes = config.get("max_episodes", None)
        self.steps = 0
        self.episodes = 0
        self.memory_save = config.get("memory_save",None)
        self.model_save = config.get("model_save",None)
        self.episode_render = config.get("episode_render",1)
        self.last_checkpoint = self.join(os.getcwd(),"checkpoints","{}".format(self.env_name))
        os.makedirs(self.last_checkpoint,exist_ok=True)
        self.last_checkpoint = self.join(self.last_checkpoint,'last_checkpoint.txt')
        self.avg_expected_reward = config.get("avg_expected_reward",None)
        self.avg_expected_reward_count = config.get("avg_expected_reward_count",100)
        self.last_k_rewards = 0
        self.avg_reward = 0
        self.max_saves = 3
        self.model_path = self.join(os.getcwd(),'rl_models','{}_{}'.format(self.env_name,self.agent_type))
        os.makedirs(self.model_path,exist_ok=True)
        self.model_name = lambda x: "model{}.ckpt".format(x)

        # ------------------replay_buffer_start--------------------

        if config["training"]:
            self.replay_buffer = mem(config.get("replay_buffer",{}),self.env_name)
            self.batch_size = config.get("batch_size",64)
            self.pretrain_length = config.get("pretrain_length",self.batch_size)
            self.pretrain_init = config.get("pretrain_init","random") == "random"
            self.memory_initialization()

        # ------------------replay_buffer_end----------------------

        self.reward_list = []
        self.loss_list = []
        self.reward_path = self.join(os.getcwd(),'rewards',self.env_name)
        os.makedirs(self.reward_path,exist_ok=True)
        self.reward_name = 'rewards.npy'
        self.loss_path = self.join(os.getcwd(),'losses',self.env_name)
        os.makedirs(self.loss_path,exist_ok=True)
        self.loss_name = 'losses.npy'

    def join(self,*args):
        # print(args)
        if len(args) == 1:
            # print(type(args[0]))
            return args[0]
        # print(args)
        return os.path.join(args[0],self.join(*args[1:]))

    def calculate_avg_reward(self):
        self.last_k_rewards += self.reward_list[-1]
        if self.avg_expected_reward_count >= self.episodes :
            return self.last_k_rewards/self.episodes
        self.last_k_rewards -= self.reward_list[self.episodes-self.avg_expected_reward_count - 1]
        return self.last_k_rewards/self.avg_expected_reward_count

    def memory_initialization(self):
        print("starting pretraining")
        state = self.env.reset()
        for i in range(self.pretrain_length):
            if self.pretrain_init:
                action = random.randint(1,self.no_actions) - 1
            else:
                action = self.agent.evaluate(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.add((state,action,reward,next_state,done))
            if done:
                state = self.env.reset()

        print("ending pretraining")

    def predict_action(self,state):
        epsilon = np.random.rand()
        if self.explore_probability > epsilon:
            action = random.randint(1,self.no_actions)-1
        else:
            action = self.agent.evaluate(state)
        return action

    def increase_step(self):
        self.steps += 1
        self.explore_probability = max(self.explore_stop,self.explore_probability-4*self.decay_rate)
        #self.explore_probability =  self.explore_stop + (self.explore_start-self.explore_stop)*np.exp(-self.steps*self.decay_rate)

    def episode_check(self):
        if self.max_episodes:
            return self.episodes <= self.max_episodes
        if self.max_steps:
            return self.steps <= self.max_steps
        return self.explore_probability > 0.01



    def train_per_episode(self):
        episode_rewards = 0
        state = self.env.reset()
        loss = 0
        current_step = 0
        render = False
        self.episodes += 1
        if self.episode_render > 0 and self.episodes % self.episode_render == 0:
            render = True
        if self.max_steps_each_episode:
            step_check = lambda x: x<=self.max_steps_each_episode
        else:
            step_check = lambda x: True

        while step_check(current_step):
            action = self.predict_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.add((state,action,reward,next_state,done))
            current_step += 1
            self.increase_step()
            self.env.render(show=render)
            episode_rewards += reward
            batch = self.replay_buffer.sample(self.batch_size)
            loss += self.agent.train(batch,done)
            if done:
                break
        return episode_rewards,loss/max(current_step,1)

    def train(self):

        while self.episode_check():

            reward, loss = self.train_per_episode()
            self.reward_list.append(reward)
            self.loss_list.append(loss)
            avg_reward = self.calculate_avg_reward()
            print("Episode: {} Step: {} Reward: {} Avg_reward: {} Explore_probability: {} loss: {}".format(
                self.episodes,self.steps,reward,avg_reward,self.explore_probability,loss
            ))
            if self.model_save is not None and self.episodes%self.model_save == 0:

                if self.last_save >= self.max_saves:
                    self.last_save = 1
                else:
                    self.last_save += 1

                model = self.join(self.model_path,self.model_name(self.last_save))
                self.agent.save(model)
                print("path: {}".format(model))

                with open(self.last_checkpoint,"w") as f:
                    f.write(model)
                    f.close()

                np.save(self.join(self.reward_path,self.reward_name),self.reward_list)
                np.save(self.join(self.loss_path,self.reward_name),self.loss_list)

            if self.memory_save is not None and self.episodes%self.memory_save == 0:
                self.replay_buffer.save()
            self.avg_reward += reward

            if self.avg_expected_reward is not None and avg_reward >= self.avg_expected_reward:
                print("Average Reward: {} Average Expected Reward : {}".format(self.avg_reward,self.avg_expected_reward))
                print("Average Expected Reward Found !\nEnding the training")
                break


        np.save(self.join(self.reward_path,self.reward_name),self.reward_list)
        np.save(self.join(self.loss_path,self.loss_name),self.loss_list)
    def last_checkpoint_path(self):
        with open(self.last_checkpoint,"r") as f:
            model = f.readline()
            print(model)
            f.close()
        return model

    def restore_agent(self,model=None):
        if not model:
            model = self.last_checkpoint_path()
        return self.agent.restore(model)

    def run(self,total_episodes,render=True):
        highest_reward = -1e18
        avg_reward = 0
        render_array = []
        for episode in range(1,total_episodes+1):
            state = self.env.reset()
            episode_reward = 0
            tmp_render = []
            done = False
            while not done:
                # action = self.predict_action(state)
                action = self.predict_action(state)
                next_state, reward, done, _  = self.env.step(action)
                tmp_render.append(self.env.render(show=True,give_array=True))

                episode_reward += reward
                # print(episode_reward,action)

            print("Episode: {} Score: {}".format(episode,episode_reward))
            if highest_reward < episode_reward:
                highest_reward = episode_reward
                render_array[:] = tmp_render[:]
            highest_reward = max(episode_reward,highest_reward)
            avg_reward += episode_reward
        avg_reward /= total_episodes
        print("Avg Reward: {}".format(avg_reward))
        return avg_reward,highest_reward,render_array

    def evaluate(self,total_episodes=3,render=True,model=None,for_all=False):
        models = []
        if model:
            models.append(model)
        elif for_all:
            for i in range(1,self.max_saves+1):
                model = self.model_path+self.model_name(i)
                models.append(model)
        else :
            models.append(self.last_checkpoint_path())
        import imageio
        for model in models:
            if not self.restore_agent(model):
                print("can't restore the model {}".format(model))
                continue
            avg_reward, highest_reward, render_array = self.run(total_episodes,render)
            _, save_gif = os.path.split(model)
            save_gif = self.join(os.getcwd(),'gifs',self.env_name,save_gif)
            os.makedirs(save_gif,exist_ok=True)
            with imageio.get_writer(self.join(save_gif,'best.mp4'),fps=20) as writer:
                for render in render_array:
                    writer.append_data(render)
            import matplotlib.pyplot as plt
            self.plot_losses(plt)
            self.plot_rewards(plt)


    def plot_losses(self,plt,load=False):
        if load:
            self.loss_list = np.load(self.join(self.loss_path,self.loss_name))
        plt.plot(self.loss_list,color="blue")
        plt.xlabel("Episodes")
        plt.ylabel("Losses")
        plt.savefig(self.join(self.reward_path,"loss.png"))
        try:
            plt.show()
        except:
            pass
        plt.clf()
    def plot_rewards(self,plt,load=False):
        if load:
            self.reward_list = np.load(self.join(self.reward_path,self.reward_name))
        avg_reward_list = []
        last_100_total = 0
        last_10_total = 0
        last_10_list = []
        last_100_list = []
        for i in range(1,len(self.reward_list)+1):
            last_10_total += self.reward_list[i-1]
            last_100_total += self.reward_list[i-1]
            if i>10:
                last_10_total -= self.reward_list[i-10-1]
            if i>100:
                last_100_total -= self.reward_list[i-100-1]
            last_10_list.append(last_10_total/10)
            last_100_list.append(last_100_total/100)

        plt.plot(self.reward_list,label="rewards",color="blue")
        plt.plot(last_10_list,label="mean reward(last 10)",color="orange")
        plt.plot(last_100_list,label="mean rewards(last 100)",color="black")
        plt.legend(loc="lower right")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.savefig(self.join(self.reward_path,"rewards.png"))
        try:
            plt.show()
        except:
            pass
        plt.clf()
