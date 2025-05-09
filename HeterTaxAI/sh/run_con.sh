#!/bin/sh$

dist="maddpg"
n=100
device=1
obj_task="gdp"

for seed in {1,2,3};
do
  let real_seed=$seed
  echo "real_seed is ${real_seed}"
  echo "device_num is ${device}"
  echo "alg is ${dist}"
  echo "task is ${obj_task}"
  echo "n is ${n}"
  CUDA_VISIBLE_DEVICES=-1 python main.py --device-num $device --n_households $n --alg $dist --task $obj_task --seed $real_seed
done

#python main.py --device-num 0 --n_households 100 --alg "maddpg" --task "gini"
#python main.py --device-num 0 --n_households 100 --alg "maddpg" --task "gini"
#python main.py --device-num 0 --n_households 100 --alg "maddpg" --task "gini"