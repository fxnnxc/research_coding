from tqdm import tqdm 

def get_batches_of_image_input_ids(processor, images, batch_size):
    outputs = []
    N = len(images)//batch_size if len(images)%batch_size == 0 else len(images)//batch_size
    if N==0:
        N=1
    for b in range(N):
        batch = processor(images=images[b*batch_size:(b+1)*batch_size], return_tensors="pt", padding=True)
        outputs.append(batch)
    return outputs


def get_batches_of_text_input_ids(processor,  texts, batch_size=32, max_len=50):
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
