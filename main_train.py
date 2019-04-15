from lib.model import model
import gym,sys
from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("specify the config file")
        sys.exit()

    fname = sys.argv[1]
    config = load(open(fname,"r"),Loader)
    for k,v in config.items():
        print("{}: {}".format(k,v))
    if "environment" not in config:
        print("please specify environment")
        sys.exit(0)
    m = model(config)
    if config["training"]:
        m.train()

