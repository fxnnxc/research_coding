

from tqdm import tqdm 
def get_all_captions(name, dataset):
    captions = []
    if name =="flickr30":
        for sample in tqdm(dataset['test']):
            captions.extend(sample['caption'])
    elif name =="pokemon":
        for sample in tqdm(dataset['train']):
            captions.append(sample['text'])
    return captions 