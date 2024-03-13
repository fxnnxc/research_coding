
import os 
from datasets import load_dataset
from datasets import load_dataset


def get_data(name, cache_dir):
    assert os.path.exists(cache_dir) 
    if name =="flickr30k":
        dataset = load_dataset("nlphuji/flickr30k", cache_dir=cache_dir)
    elif name =="pokemon":
        dataset = load_dataset("lambdalabs/pokemon-blip-captions", cache_dir=cache_dir)
    else:
        raise ValueError()
    return dataset 


from PIL import Image
def get_kakao_images(img_dir):
    if "kakaofriends" not in img_dir:
        img_dir = os.path.join(img_dir, "kakaofriends")
    images = [Image.open(os.path.join(img_dir, f)) for f in sorted(os.listdir(img_dir))]
    return images