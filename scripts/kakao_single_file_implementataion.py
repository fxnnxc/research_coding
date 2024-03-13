# -----------------
import torch 
import os 
import matplotlib.pyplot as plt 
from datasets import load_dataset
from tqdm import tqdm 
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# -- Prepare model and dataset
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="datahub")
model.to("cuda:0")
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="datahub")
dataset = load_dataset("nlphuji/flickr30k", cache_dir="datahub")

def get_all_flicker_captions(flicker_dataset):
    captions = []
    for sample in tqdm(flicker_dataset):
        captions.extend(sample['caption'])
    return captions 

captions = get_all_flicker_captions(dataset['test'])


base_dir = "datahub/kakaofriends/"
images = [Image.open(os.path.join(base_dir, f)) for f in sorted(os.listdir(base_dir))]
fig, axes = plt.subplots(2,4, figsize=(6,3)) 
axes = axes.flat
for im in images:
    ax = next(axes)
    ax.imshow(im)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()


# -- Getting embddings
def get_batches_of_image_input_ids(images, batch_size):
    outputs = []
    N = len(images)//batch_size if len(images)%batch_size == 0 else len(images)//batch_size
    if N==0:
        N=1
    for b in range(N):
        batch = processor(images=images[b*batch_size:(b+1)*batch_size], return_tensors="pt", padding=True)
        outputs.append(batch)
    return outputs

image_batches = get_batches_of_image_input_ids(images, 8)

for k, v in image_batches[0].items():
    image_batches[0][k] = v.to("cuda:0")
    
image_embeds = model.get_image_features(**image_batches[0])

batch_size=32
def get_batches_of_text_input_ids(texts, batch_size=32, max_len=50):
    outputs = []
    N = len(texts)//batch_size if len(texts)%batch_size == 0 else len(texts)//batch_size + 1
    if N==0:
        N=1
    for b in tqdm(range(N)):
        candidate_text = texts[b*batch_size:(b+1)*batch_size]
        candidate_text = [" ".join(s.split()[:max_len]) for s in candidate_text]
        batch = processor(text=candidate_text, return_tensors="pt", padding=True)
        outputs.append(batch)
    return outputs

max_len=10
text_batches  = get_batches_of_text_input_ids(captions, batch_size, max_len)

text_embeds = []
for batch in tqdm(text_batches):
    for k, v in batch.items():
        batch[k] = v.to("cuda:0")
    text_embeds.append(model.get_text_features(**batch).detach().cpu())

# -- compuse vector similarity 
def score_image_descriptions(img_tensor, list_of_text_tensors):
    scores = []
    for text_tensor in tqdm(list_of_text_tensors):
        score = img_tensor@(text_tensor.T).to("cuda:0")
        scores.append(score.cpu())
    scores = torch.concat(scores, dim=-1)
    return scores

scores = score_image_descriptions(image_embeds, text_embeds)
assert scores.shape[1] == len(captions), (scores.shape[1], len(captions) )
scores.shape 


# -- compute top-k scores
def get_top_k_captions(scores, k):
    topk_values, topk_indices = torch.topk(scores, k, dim=1)
    return topk_values, topk_indices
                             
k=5
topk_values, topk_indices = get_top_k_captions(scores, k )        

def visualize_top_k_results(images, captions, topk_indices, topk_values, figsize=(10,10)):
    K = topk_indices.shape[1]
    fig, axes = plt.subplots(len(images), K+1, figsize=figsize)
    for i in range(len(images)):
        axes[i,0].imshow(images[i])
        for k in range(K):
            c = captions[topk_indices[i,k].item()]
            c = c.split( )
            s = 2
            line =5 
            v = topk_values[i, k].item()
            c = f"score:{v:.2f} \n\n"+ " ".join([" ".join(c[s*i:s*(i+1)]) +"\n" for i in range(line)])
            axes[i,k+1].text(0.0, 1,c,clip_on=True,horizontalalignment='left',verticalalignment='top',)
            axes[i,k+1].set_xticks([])
            axes[i,k+1].set_yticks([])
            axes[i,k+1].axis("off", )
        axes[i,0].set_xticks([])
        axes[i,0].set_yticks([])
    plt.tight_layout()
visualize_top_k_results(images, captions, topk_indices, topk_values)
if not os.path.exists("outputs"):
    os.makedirs("outputs")
plt.savefig("outputs/test.png")