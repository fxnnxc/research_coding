from transformers import CLIPProcessor, CLIPModel

def get_model(name, cache_dir):
    if name == "openai/clip-vit-base-patch32":
        model = CLIPModel.from_pretrained(name, cache_dir=cache_dir)
        processor = CLIPProcessor.from_pretrained(name, cache_dir=cache_dir)
    elif name == "openai/clip-vit-large-patch14":
        model = CLIPModel.from_pretrained(name, cache_dir=cache_dir)
        processor = CLIPProcessor.from_pretrained(name, cache_dir=cache_dir)
        
    else:
        raise ValueError()
    return model, processor