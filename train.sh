#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100000M
#SBATCH --partition=infofil01
#hostname

#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=6 --model=BERT_GRU --pretrain_model=../pretrain/outputs/mwoz_pretrain_BERT/best_pretrain.pt
#CUDA_VISIBLE_DEVICES=2,3 python main.py --gpu='0 1' --batch_size=6 --model=BERT_GRU --data=sgd --pretrain_model=../pretrain/outputs/sgd_pretrain_BERT/best_pretrain.pt
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=6 --model=BERT_new
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --data=mwoz --model=USDA_P
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --data=sgd --model=USDA_P
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --data=mwoz --model=USDA_A
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --data=sgd --model=USDA_A
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --data=mwoz --model=USDA
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --data=sgd --model=USDA
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --data=jddc --model=USDA_P --bert_name=bert-base-chinese
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=16 --data=redial --model=USDA

#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --model=USDA_P --pretrain_model=../pretrain/outputs/mwoz_pretrain_BERT/best_pretrain.pt
#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --model=USDA_P --data=sgd --pretrain_model=../pretrain/outputs/sgd_pretrain_BERT/best_pretrain.pt

#CUDA_VISIBLE_DEVICES=0,1 python main.py --gpu='0 1' --batch_size=24 --model=USDA_P --data=jddc --bert_name=bert-base-chinese --pretrain_model=../pretrain/outputs/jddc_pretrain_BERT/best_pretrain.pt

CUDA_VISIBLE_DEVICES=0 python -u main.py --gpu='0' --batch_size=6 --model=USDA --data=sgd
