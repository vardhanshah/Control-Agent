from memory import memory
import random
import numpy as np
import os
import time

try:
    from yaml import load, dump, CLoader as Loader, CDumper as Dumper
except:
    from yaml import load, dump, Loader, Dumper


class model:
    last_save = 1
    max_saves = 3

    def __init__(self, config):

        self.config = config

        if "folder_id" in config:
            self.folder_id = config["folder_id"]
        else:
            from datetime import datetime
            from uuid import uuid4
            eventid = datetime.now().strftime('%Y:%m:%d-%H:%M:%S-') + str(uuid4())
            self.env_name = config["environment"]["name"]
            self.agent_type = config["agent"]["type"]
            self.folder_id = '{}_{}_{}'.format(self.env_name, self.agent_type, eventid)

        self.prime_folder = self.join(os.getcwd(), "data", self.folder_id)
        os.makedirs(self.prime_folder, exist_ok=True)
        config_file = self.join(self.prime_folder, "config.yaml")
        if "environment" in config and "agent" in config:
            self.env_agent_config = {"environment": config["environment"], "agent": config["agent"]}

            dump(self.config, open(config_file, "w"), Dumper)
        else:
            self.env_agent_config = load(open(config_file, "r"), Loader)

        self.steps = 0
        self.episodes = 0

        self.reward_list = []
        self.loss_list = []

        self.model_path = self.join(self.prime_folder, 'rl_models')
        self.model_name = lambda x: "model{}.ckpt".format(x)
        self.last_checkpoint = self.join(self.prime_folder, "checkpoints")
        os.makedirs(self.last_checkpoint, exist_ok=True)
        self.last_checkpoint = self.join(self.last_checkpoint, 'last_checkpoint.txt')
        self.reward_path = self.join(self.prime_folder, 'rewards')

        self.reward_file = self.join(self.reward_path, 'rewards.npy')
        self.loss_path = self.join(self.prime_folder, 'losses')
        self.loss_file = self.join(self.loss_path, 'losses.npy')

        self.last_k_rewards = 0
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.reward_path, exist_ok=True)
        os.makedirs(self.loss_path, exist_ok=True)
        self.memory_path = self.join(self.prime_folder, 'memory')
        self.memory_name = "data.npy"
        self.memory_file = self.join(self.memory_path, self.memory_name)
        self.total_episode_time = 0
        self.total_time = 0
        self.done = False

    def load_env_agent(self, config):
        from environment import Environment

        self.env = Environment(config["environment"])
        self.env_name = config["environment"]["name"]

        self.state_shape = self.env.state_shape
        self.no_actions = self.env.no_actions
        self.frame_shape = self.env.frame_shape
        self.stack_size = self.env.stack_size
        self.env_type = self.env.type
        self.env_dtype = self.env.dtype
        self.scaled = 1.0
        if self.env_type == "atari":
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

        self.agent = Agent(self.state_shape, self.no_actions, config["agent"])

    def train(self, restore_model=False, restore_memory=False):
        start_time = time.time()
        self.load_env_agent(self.env_agent_config)
        if restore_model:
            model, self.explore_probability = self.last_checkpoint_path()
            self.restore_model(model)
        self.batch_size = self.config.get("batch_size", 64)
        self.pretrain_length = self.config.get("pretrain_length", self.batch_size)
        self.pretrain_init = self.config.get("pretrain_init", "random") == "random"

        self.replay_buffer = memory(self.config.get("memory_size", 1e6), self.batch_size, self.frame_shape,
                                    self.stack_size,
                                    self.state_shape, self.env_dtype)

        if restore_memory:
            self.replay_buffer.load(self.join(self.memory_path, self.memory_name))
        else:
            self.memory_initialization()

        self.max_steps = self.config.get("max_steps", None)
        self.max_steps_each_episode = self.config.get("max_steps_each_episode", None)
        self.max_episodes = self.config.get("max_episodes", None)
        self.model_saving_counter = self.config.get("model_save", None)
        self.episode_render = self.config.get("episode_render", 1)
        self.avg_expected_reward = self.config.get("avg_expected_reward", None)
        self.avg_expected_reward_count = self.config.get("avg_expected_reward_count", 100)

        self.training_frequency = self.config.get("training_frequency", 1)
        self.explore_start = self.config.get("explore_start", 1)
        self.explore_stop = self.config.get("explore_stop", 0.1)
        self.decay_rate = self.config.get("decay_rate", 9e-7)
        if not restore_model:
            self.explore_probability = self.explore_start
        self.memory_saving_counter = None
        if "memory_save" in self.config:
            os.makedirs(self.memory_path, exist_ok=True)
            self.memory_saving_counter = self.config["memory_save"]

        while self.episode_check():

            reward, loss = self.train_per_episode()

            self.reward_list.append(reward)
            self.loss_list.append(loss)
            avg_reward = self.calculate_avg_reward()

            print("Episode: {} Step: {} Reward: {} Avg_reward: {} Explore_probability: {} loss: {}".format(
                self.episodes, self.steps, reward, avg_reward, self.explore_probability, loss
            ))

            if self.model_saving_counter is not None and self.episodes % self.model_saving_counter == 0:
                self.save_model()
                self.save_reward()
                self.save_loss()
                self.replay_buffer.occupied_memory()
                print("Average time on each episodes: {}".format(
                    self.standard_representation_time(self.total_episode_time / self.episodes)))

            if self.memory_saving_counter is not None and self.episodes % self.memory_saving_counter == 0:
                self.replay_buffer.save(self.memory_file)
                print("saved the memory {}".format(self.memory_file))

            if self.avg_expected_reward is not None and avg_reward >= self.avg_expected_reward:
                print("Average Reward: {} Average Expected Reward : {}".format(avg_reward, self.avg_expected_reward))
                print("Average Expected Reward Found !\nEnding the training")
                self.done = True
                break

        self.total_time = time.time() - start_time
        self.save_model()
        self.save_reward()
        self.save_loss()

        print("Total time on each episodes: {}".format(self.standard_representation_time(self.total_time)))
        print("Average time on each episodes: {}".format(
            self.standard_representation_time(self.total_episode_time / self.episodes)))
        self.plot()

        self.test(total_episodes=200, render=False, plot_train=False)

    def train_per_episode(self):
        start_episode = time.time()
        episode_rewards = 0
        loss = 0
        current_step = 0
        render = False
        self.episodes += 1

        if self.episode_render > 0 and self.episodes % self.episode_render == 0:
            render = True

        if self.max_steps_each_episode:
            step_check = lambda x: x <= self.max_steps_each_episode
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
            self.replay_add(action, next_frame, reward, done_replay)
            episode_rewards += reward
            states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb = self.replay_buffer.sample()

            if self.scaled > 1.0:
                states_mb = states_mb / self.scaled  # states scaling

                next_states_mb = next_states_mb / self.scaled  # new_states scaling

            if current_step % self.training_frequency == 0:
                loss += self.agent.train(states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb, done)
            if done:
                self.env.close()
                break
        if render:
            self.env.close()
        self.total_episode_time += time.time() - start_episode
        return episode_rewards, loss / max(current_step // 4, 1)

    def join(self, *args):
        if len(args) == 1:
            return args[0]
        return os.path.join(args[0], self.join(*args[1:]))

    def replay_add(self, action, frame, reward, done):
        frame = self.env.preprocess_frame(frame)
        self.replay_buffer.add(action, frame, reward, done)

    def memory_initialization(self):
        print("starting pretraining")
        self.env.reset()
        for i in range(self.pretrain_length):
            if self.pretrain_init:
                action = random.randint(1, self.no_actions) - 1
            else:
                action = self.agent.evaluate(self.env.state)
            next_frame, reward, done, done_replay = self.env.step(action)
            self.replay_add(action, next_frame, reward, done_replay)
            if done:
                self.env.reset()

        print("ending pretraining")

    def predict_action(self, state):
        epsilon = np.random.rand()
        if self.explore_probability > epsilon:
            action = random.randint(1, self.no_actions) - 1
        else:
            action = self.agent.evaluate(state)
        return action

    def increase_step(self):
        self.steps += 1
        self.explore_probability = max(self.explore_stop, self.explore_probability - self.decay_rate)
        # self.explore_probability =  self.explore_stop + (self.explore_start-self.explore_stop)*np.exp(-self.steps*self.decay_rate)

    def episode_check(self):
        if self.max_episodes:
            return self.episodes <= self.max_episodes
        if self.max_steps:
            return self.steps <= self.max_steps
        return self.explore_probability > 0.01

    def calculate_avg_reward(self):
        self.last_k_rewards += self.reward_list[-1]
        if self.avg_expected_reward_count >= self.episodes:
            return self.last_k_rewards / self.episodes
        self.last_k_rewards -= self.reward_list[self.episodes - self.avg_expected_reward_count - 1]
        return self.last_k_rewards / self.avg_expected_reward_count

    def save_reward(self):
        np.save(self.reward_file, self.reward_list)

    def save_loss(self):
        np.save(self.loss_file, self.loss_list)

    def save_model(self):
        model = self.join(self.model_path, self.model_name(self.last_save))
        self.agent.save(model)
        details = {
            "model_name": self.model_name(self.last_save),
            "explore_probability": self.explore_probability,
            "steps": self.steps,
            "episodes": self.episodes,
            "total_time": self.standard_representation_time(self.total_episode_time),
            "avg_episode_time": self.standard_representation_time(self.total_episode_time / self.episodes),
            "done": self.done
        }
        with open(self.last_checkpoint, "w") as f:
            dump(details, f, Dumper)

        print("path: {}".format(model))
        if self.last_save >= self.max_saves:
            self.last_save = 1
        else:
            self.last_save += 1

    def standard_representation_time(self, seconds):
        time_format = lambda x, y, z: "{} hr {} min {} s".format(x, y, z)
        if seconds < 60:
            return time_format(0, 0, seconds)
        minutes = seconds // 60
        seconds = seconds % 60
        if minutes < 60:
            return time_format(0, minutes, seconds)
        hours = minutes // 60
        minutes %= 60
        print("{} hr {} min {} s".format(hours, minutes, seconds))
        return time_format(hours, minutes, seconds)

    def last_checkpoint_path(self):
        try:
            details = {}
            with open(self.last_checkpoint, "r") as f:
                details = load(f, Loader)
            model = os.path.join(self.model_path, details['model_name'])
            print(model)
            return model, details['explore_probability']
        except:
            return None

    def restore_model(self, model):
        return self.agent.restore(model)

    def run(self, total_episodes, render):
        highest_reward = -1e18
        reward_list = []
        render_array = []
        for episode in range(1, total_episodes + 1):
            tmp_render = []
            self.env.reset()
            tmp_render.append(self.env.render("rgb_array"))
            episode_reward = 0
            done = False
            while not done:

                # action = self.predict_action(self.env.state)
                action = self.agent.evaluate(self.env.state)
                next_frame, reward, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                tmp_render.append(self.env.render(mode="rgb_array"))
                episode_reward += reward
                # print(episode_reward,action)
            print("Episode: {} Score: {}".format(episode, episode_reward))
            # self.env.close()
            if highest_reward < episode_reward:
                highest_reward = episode_reward
                render_array[:] = tmp_render[:]
            reward_list.append(episode_reward)
        return reward_list, render_array

    def test(self, total_episodes=3, render=True, model=None, for_all=False, plot_train=False):

        self.load_env_agent(self.env_agent_config)

        self.explore_start = self.config.get("explore_start", 0.3)
        self.explore_stop = self.config.get("explore_stop", 0.3)
        self.decay_rate = self.config.get("decay_rate", )

        self.explore_probability = self.explore_start
        reward_list = []
        models = []
        if model:
            models.append(model)
        elif for_all:
            for i in range(1, self.max_saves + 1):
                model = self.join(self.model_path, self.model_name(i))
                models.append(model)
        else:
            models.append(self.last_checkpoint_path()[0])
        import imageio

        for model in models:
            if not self.restore_model(model):
                print("can't restore the model {}".format(model))
                continue
            reward_list, render_array = self.run(total_episodes, render)
            if total_episodes >= 100:
                import matplotlib.pyplot as plt
                saved_plot = self.join(self.reward_path,
                                       "rewards_test_{}.png".format(os.path.splitext(os.path.basename(model))[0]))
                self.plot_with_average(plt, reward_list, [10, 100], saved_plot, "Rewards")

            print(
                "Average reward: {}, Highest Reward: {}".format(sum(reward_list) / len(reward_list), max(reward_list)))
            front, save_gif = os.path.split(model)
            _, game_name = os.path.split(front)
            save_gif = self.join(self.prime_folder, 'gifs', save_gif)
            os.makedirs(save_gif, exist_ok=True)
            save_gif = self.join(save_gif, 'best.gif')
            with imageio.get_writer(save_gif, fps=50) as writer:
                for r in render_array:
                    writer.append_data(r)
            print("Saved gif: {}".format(save_gif))
            #  imageio.mimwrite(save_gif + '/per.gif',render_array)
        if plot_train:
            self.plot(load=True)

    def plot_losses(self, plt, load=False):
        if load:
            self.loss_list = np.load(self.loss_file)
        saved_plot = self.join(self.loss_path, "losses.png")
        self.plot_with_average(plt, self.loss_list, [], saved_plot, "Losses")

    def plot(self, load=False):
        try:
            import matplotlib.pyplot as plt
            self.plot_losses(plt, load=load)
            self.plot_rewards(plt, load=load)
        except:
            pass

    def plot_with_average(self, plt, list, counter_list, location, ylabel, xlabel="Episodes", show=False):

        last_k_total = [0] * len(counter_list)
        last_k_list = [[] for i in range(len(counter_list))]
        for i in range(0, len(list)):
            for j in range(len(last_k_total)):
                last_k_total[j] += list[i]
                if i >= counter_list[j]:
                    last_k_total[j] -= list[i - counter_list[j]]
                last_k_list[j].append(last_k_total[j] / min(i + 1, counter_list[j]))

        plt.plot(list, label=ylabel, color="blue")
        # colors = ["orange","black"]
        for i in range(len(last_k_list)):
            plt.plot(last_k_list[i], label="mean over last {}".format(counter_list[i]))
        plt.legend(loc="upper left")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(location)
        print("{} plot saved at {}".format(ylabel, location))
        try:
            if show:
                plt.show()
        except:
            pass
        plt.clf()

    def plot_rewards(self, plt, load=False):
        if load:
            self.reward_list = np.load(self.reward_file)

        saved_plot = self.join(self.reward_path, "rewards.png")
        self.plot_with_average(plt, self.reward_list, [10, 100], saved_plot, "Rewards")
