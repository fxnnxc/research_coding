import matplotlib.pyplot as plt 

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