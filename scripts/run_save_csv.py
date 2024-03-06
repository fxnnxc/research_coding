
import argparse
import os 
import random 
import torch 
import numpy as np 
from omegaconf import OmegaConf
import datetime 
from tqdm import tqdm 

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


# ------------ Required or CSV ---------------
import csv
import numpy as np 
import datetime 

def append_right(values, length=15):
    return [str(v)+" "*(length - len(str(v))) for v in values]
    
names = ['EPOCH', 'loss', 'lr', 'Save Time']    
delimiter = '\t'
with open(os.path.join(flags.save_dir, 'results.csv'), 'w', newline='') as csvfile:
            csv.writer(csvfile, delimiter=delimiter,).writerow(append_right(names))


import time 
with tqdm(range(flags.epochs)) as pbar:
    pbar.set_description("🗂️:"+flags.save_dir)
    
    for epoch in pbar:
        time.sleep(0.5)
        for i in range(100):
            # minibatch
            pass 

        lr = (epoch+1)/10000
        loss = np.random.random()/(epoch+1)
        pbar.set_postfix({'lr':lr, 'loss':loss})
        
            
        with open(os.path.join(flags.save_dir, 'results.csv'), 'a', newline='') as csvfile:
            csv.writer(csvfile, delimiter=delimiter,).writerow(append_right([epoch, 
                                            str(np.round(loss,6)), 
                                            str(np.round(lr,5)), 
                                            datetime.datetime.now().strftime("🗓️ %y/%m/%d %H:%M:%S")]))    
        


OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))
