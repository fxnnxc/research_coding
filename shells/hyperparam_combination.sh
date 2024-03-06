
# seed epoch
c1=(0 10)
c2=(1 20)
c3=(2 30)
c4=(3 40)
c5=(4 50)

candidates=(
    c1 c2 c3 c4 c5
)

for p in ${candidates[@]}
do 
    declare -n pair=$p 
    args=("${pair[@]}")
    seed=${args[0]}
    epochs=${args[1]}
        
    save_dir="outputs/hyperparam/"$epochs"_seed_"$seed
    python scripts/run_hyperparam.py \
        --seed $seed \
        --save_dir $save_dir \
        --epochs $epochs 

done 