try:
    from yaml import load, CLoader as Loader
except:
    from yaml import load, Loader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--id", help="give folder id", type=str)
parser.add_argument("-tr", "--train", help="train with given folder id", action="store_true")
parser.add_argument("-ts", "--test", "--evaluate", "-ev",
                    help="test with given folder id, folder id is folder name in data", action="store_true")
parser.add_argument("-yf", "--yaml_file", help="hyper-parameters file", type=str)
parser.add_argument("--explore_start", help="exploring probability at start", type=float)
parser.add_argument("--explore_stop", help="minimum exploring probablity", type=float)
parser.add_argument("--decay_rate", help="decaying rate of exploring", type=float)
parser.add_argument("-aer", "--avg_expected_reward", help="Average expected reward", type=float)
parser.add_argument("--model_save", help="Model saving after given number of episodes", type=int)
parser.add_argument("--memory_save",help="Memory saving after given number of episodes", type=int)
parser.add_argument("-er", "--episode_render", help="Episode Rendering after given number of episodes", type=int)
parser.add_argument("--max_steps", help="maximum number of steps for training the agent", type=int)
parser.add_argument("--max_episodes",
                    help="maximum number of episodes for training the agent. This has higher precedence over max_steps",
                    type=int)
parser.add_argument("--training_frequency", help="training periodically after given steps in each episode", type=int)
parser.add_argument("--batch_size", help="batch size of experiences to give Network", type=int)
parser.add_argument("--pretrain_length", help="pretraining experiences", type=int)
parser.add_argument("--pretrain_init", help="pretraining randomly or with agent", type=str, choices=["random", "agent"])
parser.add_argument("--for_all", help="Only to be used if testing. this flag will evaluate all saved models",
                    action="store_true")
parser.add_argument("--plot",
                    help="Only to be used if testing. this flag will plot rewards and losses during training",
                    action="store_true")
parser.add_argument("--only_plot",
                    help="plots rewards and losses during training, give folder_id",
                    action="store_true")
parser.add_argument("--total_episodes",
                    help="testing will done for given number of episodes",
                    type=int)
parser.add_argument("--restore_model",
                    help="only to be used to restore model for training purpose",
                    action="store_true")
parser.add_argument("--restore_memory",
                    help="only to be used if saved memory during training, want to restore that memory",
                    action="store_true")

args = parser.parse_args()
config = {}
if args.yaml_file:
    with open(args.yaml_file, "r") as f:
        config = load(f, Loader)
if args.id:
    config["folder_id"] = args.id
if args.train:
    config["training"] = True
if config.get("training", False):
    args.train = True
for k, v in vars(args).items():
    if k in config:
        if v:
            config[k] = v
print("Final Config File: ")
for k, v in config.items():
    print(k, v)

# Main juice

from model import model

m = model(config)
if args.only_plot:
    m.plot(True)

elif args.test == args.train:
    m.train(restore_model=args.restore_model,restore_memory=args.restore_memory)
    m.explore_stop = 0.1
    m.test(for_all=True)

elif args.train:
    m.train(restore_model=args.restore_model,restore_memory=args.restore_memory)

elif args.test:
    if args.total_episodes:
        tmp = args.total_episodes
    else:
        tmp = 1
    m.test(total_episodes=tmp, render=True, plot_train=args.plot, for_all=args.for_all)

