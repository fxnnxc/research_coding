from imgcap.data import get_data
from imgcap.models import get_model
from imgcap.utils.caption_scores import get_top_k_captions, score_image_descriptions
from imgcap.utils.clip_utils import get_batches_of_image_input_ids, get_batches_of_text_input_ids
from imgcap.utils.caption_utils import get_all_captions
from imgcap.data import get_kakao_images
from imgcap.utils.plot import visualize_top_k_results
import matplotlib.pyplot as plt 
from tqdm import tqdm 

import os 
import random 
import torch 
import numpy as np 
import datetime 
import argparse
from omegaconf import OmegaConf 

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model_name", default='openai/clip-vit-base-patch32')
parser.add_argument("--model_cache_dir", default='datahub')
parser.add_argument("--data_name", default='flickr30k')
parser.add_argument("--data_cache_dir", default='datahub')
parser.add_argument("--img_cache_dir", default='datahub')
args = parser.parse_args()

flags = OmegaConf.create({})
batch_size=32
max_len=10
K=5
device='cuda:0'

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
OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))


# --------- Load Models ---------
model, processor = get_model(flags.model_name, flags.model_cache_dir)
model.to(device)
dataset = get_data(flags.data_name, flags.data_cache_dir)


# --------- Images ---------
images = get_kakao_images(flags.img_cache_dir)
print(f"[INFO] process image embeddings")
image_batches = get_batches_of_image_input_ids(processor, images, 8)
for k, v in image_batches[0].items():
        image_batches[0][k] = v.to(device)
image_embeds = model.get_image_features(**image_batches[0])

# --------- Texts ---------
print(f"[INFO] get image captions")
captions = get_all_captions(flags.data_name, dataset)
text_batches  = get_batches_of_text_input_ids(processor, captions, batch_size, max_len)
text_embeds = []
print(f"[INFO] process text embeddings")
for batch in tqdm(text_batches):
    for k, v in batch.items():
        batch[k] = v.to(device)
    text_embeds.append(model.get_text_features(**batch).detach().cpu())
    
    
# --------- Dot Products ---------
scores = score_image_descriptions(image_embeds, text_embeds, device=device)
assert scores.shape[1] == len(captions), (scores.shape[1], len(captions) )

# --------- Top-K Resutls ---------
topk_values, topk_indices = get_top_k_captions(scores, K )        
visualize_top_k_results(images, captions, topk_indices, topk_values)
plt.savefig(os.path.join(flags.save_dir, "result.png"))
print(f"[INFO] image is saved in {flags.save_dir}")

flags.done=True
OmegaConf.save(flags, os.path.join(flags.save_dir, "config.yaml"))
