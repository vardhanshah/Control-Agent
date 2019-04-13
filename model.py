from memory import memory as mem
import random
import numpy as np
import os
class model:

    last_save = 1

    def __init__(self,config):

        # if config["environment"]["type"] == "atari":
        global Environment
        from environment import AtariEnvironment as Environment

        self.env = Environment(config["environment"])
        self.env_name = config["environment"]["name"]
        self.state_shape = self.env.state_shape
        self.no_actions = self.env.no_actions
        self.frame_shape = self.env.frame_shape
        self.stack_size = self.env.stack_size
        self.env_type = self.env.type
        self.env_dtype = self.env.dtype
        self.scaled = 1.0
        if self.env_type=="atari":
            self.scaled = 255.0
        self.agent_type = config["agent"]["type"]
        if self.agent_type == "dqn":
            global Agent
            from dqn import Agent
        elif self.agent_type == "double_dqn":
            global Agent
            from double_dqn import Agent
        elif self.agent_type == "target_dqn":
            global Agent
            from target_dqn import Agent

        self.agent = Agent(self.state_shape,self.no_actions,config["agent"])
        self.explore_start = config.get("explore_start",1.0)
        self.explore_stop = config.get("explore_stop",0.01)
        self.decay_rate = config.get("decay_rate",0.00001)
        self.explore_probability = self.explore_start
        self.max_steps = config.get("max_steps",None)
        self.max_steps_each_episode = config.get("max_steps_each_episode",None)
        self.max_episodes = config.get("max_episodes", None)
        self.steps = 0
        self.episodes = 0
        self.model_saving_counter = config.get("model_save",None)
        self.episode_render = config.get("episode_render",1)
        self.last_checkpoint = self.join(os.getcwd(),"checkpoints","{}_{}".format(self.env_name,self.agent_type))
        os.makedirs(self.last_checkpoint,exist_ok=True)
        self.last_checkpoint = self.join(self.last_checkpoint,'last_checkpoint.txt')
        self.avg_expected_reward = config.get("avg_expected_reward",None)
        self.avg_expected_reward_count = config.get("avg_expected_reward_count",100)
        self.max_saves = 3
        self.training_frequency = config.get("training_frequency",4)
        self.model_path = self.join(os.getcwd(),'rl_models','{}_{}'.format(self.env_name,self.agent_type))
        self.model_name = lambda x: "model{}.ckpt".format(x)
        os.makedirs(self.model_path,exist_ok=True)

        # ------------------replay_buffer_start--------------------

        self.batch_size = config.get("batch_size",64)
        self.pretrain_length = config.get("pretrain_length",self.batch_size)
        self.pretrain_init = config.get("pretrain_init","random") == "random"

        self.replay_buffer = mem(config.get("memory_size",1e6),self.batch_size,self.frame_shape,self.stack_size,self.state_shape,self.env_dtype)

        self.memory_initialization()

        # ------------------replay_buffer_end----------------------

        self.reward_list = []
        self.loss_list = []
        self.reward_path = self.join(os.getcwd(),'rewards','{}_{}'.format(self.env_name,self.agent_type))
        os.makedirs(self.reward_path,exist_ok=True)
        self.reward_file = self.join(self.reward_path,'rewards.npy')
        self.loss_path = self.join(os.getcwd(),'losses','{}_{}'.format(self.env_name,self.agent_type))
        os.makedirs(self.loss_path,exist_ok=True)
        self.loss_file = self.join(self.loss_path,'losses.npy')

        self.last_k_rewards = 0

    def join(self,*args):
        if len(args) == 1:
            return args[0]
        return os.path.join(args[0],self.join(*args[1:]))

    def replay_add(self,action,frame, reward,done):
        frame = self.env.preprocess_frame(frame)
        self.replay_buffer.add(action,frame,reward,done)


    def memory_initialization(self):
        print("starting pretraining")
        self.env.reset()
        for i in range(self.pretrain_length):
            if self.pretrain_init:
                action = random.randint(1,self.no_actions) - 1
            else:
                action = self.agent.evaluate(self.env.state)
            next_frame, reward, done, done_replay = self.env.step(action)
            self.replay_add(action,next_frame,reward,done_replay)
            if done:
                self.env.reset()

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
        self.explore_probability = max(self.explore_stop,self.explore_probability-self.decay_rate)
        # self.explore_probability =  self.explore_stop + (self.explore_start-self.explore_stop)*np.exp(-self.steps*self.decay_rate)

    def episode_check(self):
        if self.max_episodes:
            return self.episodes <= self.max_episodes
        if self.max_steps:
            return self.steps <= self.max_steps
        return self.explore_probability > 0.01

    def calculate_avg_reward(self):
        self.last_k_rewards += self.reward_list[-1]
        if self.avg_expected_reward_count >= self.episodes :
            return self.last_k_rewards/self.episodes
        self.last_k_rewards -= self.reward_list[self.episodes-self.avg_expected_reward_count - 1]
        return self.last_k_rewards/self.avg_expected_reward_count

    def train_per_episode(self):
        episode_rewards = 0
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

        self.env.reset()
        while step_check(current_step):
            action = self.predict_action(self.env.state)
            next_frame, reward, done, done_replay = self.env.step(action)
            if render:
                self.env.render()
            current_step += 1
            self.increase_step()
            self.replay_add(action,next_frame,reward,done_replay)
            episode_rewards += reward
            states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = self.replay_buffer.sample()

            if self.scaled > 1.0:
                states_mb = states_mb/self.scaled #states scaling

                next_states_mb = next_states_mb/self.scaled #new_states scaling

            if current_step%self.training_frequency == 0:
                loss += self.agent.train(states_mb,actions_mb,rewards_mb,next_states_mb,dones_mb,done)
            if done:
                self.env.close()
                break
        if render:
            self.env.close()
        return episode_rewards,loss/max(current_step//4,1)

    def save_reward(self):
        np.save(self.reward_file,self.reward_list)

    def save_loss(self):
        np.save(self.loss_file,self.loss_list)

    def save_model(self):
        model = self.join(self.model_path,self.model_name(self.last_save))
        self.agent.save(model)
        print("path: {}".format(model))
        if self.last_save >= self.max_saves:
            self.last_save = 1
        else:
            self.last_save += 1

        with open(self.last_checkpoint,"w") as f:
            f.write(model)
            f.close()


    def train(self):

        
        while self.episode_check():

            reward, loss = self.train_per_episode()
            self.reward_list.append(reward)
            self.loss_list.append(loss)
            avg_reward = self.calculate_avg_reward()

            print("Episode: {} Step: {} Reward: {} Avg_reward: {} Explore_probability: {} loss: {}".format(
                self.episodes,self.steps,reward,avg_reward,self.explore_probability,loss
            ))

            if self.model_saving_counter is not None and self.episodes%self.model_saving_counter == 0:
                self.save_model()
                self.save_reward()
                self.save_loss()
                print(self.replay_buffer.occupied_memory() )
            if self.avg_expected_reward is not None and avg_reward >= self.avg_expected_reward:
                print("Average Reward: {} Average Expected Reward : {}".format(avg_reward,self.avg_expected_reward))
                print("Average Expected Reward Found !\nEnding the training")
                break

        self.save_model()
        self.save_reward()
        self.save_loss()
        try:
            import matplotlib.pyplot as plt
            self.plot_losses(plt)
            self.plot_rewards(plt)
        except:
            pass
    def last_checkpoint_path(self):
        try:
            with open(self.last_checkpoint,"r") as f:
                model = f.readline()
                f.close()
            return model
        except:
            return None
    def restore_agent(self,model=None):
        if not model:
            model = self.last_checkpoint_path()
        return self.agent.restore(model)

    def run(self,total_episodes,render=True):
        highest_reward = -1e18
        avg_reward = 0
        render_array = []
        for episode in range(1,total_episodes+1):
            tmp_render = []
            self.env.reset()
            tmp_render.append(self.env.render("rgb_array"))
            episode_reward = 0
            done = False
            while not done:
                action = self.predict_action(self.env.state)
                # action = self.agent.evaluate(self.env.state)
                next_frame, reward, done, _ = self.env.step(action)
                # if render:
                    # self.env.render()
                tmp_render.append(self.env.render("rgb_array"))
                episode_reward += reward
                # print(episode_reward,action)
            print("Episode: {} Score: {}".format(episode,episode_reward))
            self.env.close()
            if highest_reward < episode_reward:
                highest_reward = episode_reward
                render_array[:] = tmp_render[:]
            highest_reward = max(episode_reward,highest_reward)
            avg_reward += episode_reward
        avg_reward /= total_episodes
        print("Avg Reward: {}".format(avg_reward))
        return avg_reward,highest_reward,render_array

    def evaluate(self,total_episodes=3,render=True,model=None,for_all=False,plot=False):
        models = []
        if model:
            models.append(model)
        elif for_all:
            for i in range(1,self.max_saves+1):
                model = self.join(self.model_path,self.model_name(i))
                models.append(model)
        else :
            models.append(self.last_checkpoint_path())
        import imageio
        for model in models:
            if not self.restore_agent(model):
                print("can't restore the model {}".format(model))
                continue
            avg_reward, highest_reward, render_array = self.run(total_episodes,render)
            front, save_gif = os.path.split(model)
            _,game_name = os.path.split(front)
            save_gif = self.join(os.getcwd(),'gifs',game_name,save_gif)
            os.makedirs(save_gif,exist_ok=True)
            with imageio.get_writer(self.join(save_gif,'best.gif'),fps=50) as writer:
                for render in render_array:
                    writer.append_data(render)
            #  imageio.mimwrite(save_gif + '/per.gif',render_array)
        if plot:
            try:
                import matplotlib.pyplot as plt
                self.plot_losses(plt,load=True)
                self.plot_rewards(plt,load=True)
            except:
                pass

    def plot_losses(self,plt,load=False):
        if load:
            self.loss_list = np.load(self.loss_file)
        plt.plot(self.loss_list,color="blue")
        plt.xlabel("Episodes")
        plt.ylabel("Losses")
        saved_plot = self.join(self.loss_path,"losses.png")
        plt.savefig(saved_plot)
        print("losses plot saved at {}".format(saved_plot))
        # try:
        #     # plt.show()
        # except:
        #     pass
        plt.clf()

    def plot_rewards(self,plt,load=False):
        if load:
            self.reward_list = np.load(self.reward_file)
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
            last_10_list.append(last_10_total/min(i,10))
            last_100_list.append(last_100_total/min(i,100))

        plt.plot(self.reward_list,label="rewards",color="blue")
        plt.plot(last_10_list,label="mean reward(last 10)",color="orange")
        plt.plot(last_100_list,label="mean rewards(last 100)",color="black")
        plt.legend(loc="lower right")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        saved_plot = self.join(self.reward_path,"rewards.png")
        plt.savefig(saved_plot)
        print("rewards plot saved at {}".format(saved_plot))
        # try:
        #     # plt.show()
        # except:
        #     pass
        plt.clf()

