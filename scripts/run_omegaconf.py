
import argparse
import os 
import random 
import torch 
import numpy as np 
from omegaconf import OmegaConf
import datetime 

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir")
parser.add_argument("--epochs", type=int)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()


flags  = OmegaConf.create({})
flags.done = False
flags.datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")
for k, v in vars(args).items():
    print(">>>", k, ":" , v)
    setattr(flags, k, v)

# ---- Set seed ----
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.set_num_threads(4)

# ---- make dirs ----
file_name = __file__.split("/")[-1].split(".")[0]
train_idx=0
while os.path.exists(os.path.join(args.save_dir, f'{file_name}_{train_idx}')):
    train_idx+=1 
flags.save_dir = os.path.join(args.save_dir, f'{file_name}_{train_idx}')
os.makedirs(flags.save_dir)

print(flags.save_dir)
OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))