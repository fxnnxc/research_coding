
epochs=10

for seed in 0 1 2 3
do 
save_dir="outputs/omegaconf/"seed_$seed
python scripts/run_omegaconf.py \
    --seed $seed \
    --save_dir $save_dir \
    --epochs $epochs 

done 