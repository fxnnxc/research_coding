
data_name='pokemon' #flickr30k 
model_name="openai/clip-vit-large-patch14"  # openai/clip-vit-base-patch32
save_dir="outputs/"$data_name"/"$model_name
python scripts/kakao_captioning.py \
    --save_dir $save_dir \
    --data_name $data_name \
    --model_name $model_name
    