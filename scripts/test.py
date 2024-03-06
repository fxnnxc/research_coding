import argparse
import os 
import random 
import torch 
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir")
parser.add_argument("--epochs", type=int)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

for k, v in vars(args).items():
    print(">>>", k, ":" , v)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.set_num_threads(4)


file_name = __file__.split("/")[-1].split(".")[0]
train_idx=0
while os.path.exists(os.path.join(args.save_dir, f'{file_name}_{train_idx}')):
    train_idx+=1 
args.save_dir = os.path.join(args.save_dir, f'{file_name}_{train_idx}')
os.makedirs(args.save_dir)
