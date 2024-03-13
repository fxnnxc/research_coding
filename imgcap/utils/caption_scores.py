import torch 
from tqdm import tqdm 

# -- compuse vector similarity 
def score_image_descriptions(img_tensor, list_of_text_tensors, device):
    scores = []
    for text_tensor in tqdm(list_of_text_tensors):
        score = img_tensor.to(device)@(text_tensor.T).to(device)
        scores.append(score.cpu())
    scores = torch.concat(scores, dim=-1)
    return scores

# -- compute top-k scores
def get_top_k_captions(scores, k):
    topk_values, topk_indices = torch.topk(scores, k, dim=1)
    return topk_values, topk_indices