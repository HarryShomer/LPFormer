#!/bin/bash

# Cora
python src/run.py --data_name cora --lr 5e-3  --gnn-layers 1 --dim 128  --batch-size 1024  --epochs 100 --eps 1e-7 --gnn-drop 0.1 --dropout 0.1 --pred-drop 0.1 --att-drop 0.1 --num-heads 1  --thresh-1hop 1e-2 --thresh-non1hop 1e-2  --feat-drop 0.1 --l2 0 --eval_steps 1 --decay 0.975  --runs 10 --non-verbose --device 0 

# Citeseer
python src/run.py --data_name citeseer --lr 5e-3  --gnn-layers 1 --dim 256 --batch-size 1024  --epochs 100 --kill_cnt 100 --eps 1e-7 --gnn-drop 0.1 --dropout 0.1 --pred-drop 0.1 --att-drop 0.1  --num-heads 1  --thresh-1hop 1e-2 --thresh-non1hop 1 --feat-drop 0.1 --eval_steps 1 --decay 0.95 --non-verbose --l2 0 --runs 10 --device 0

# Pubmed
python src/run.py --data_name pubmed --lr 1e-3  --gnn-layers 1 --dim 128 --batch-size 1024  --epochs 100 --eps 1e-5 --gnn-drop 0.3 --dropout 0.3 --pred-drop 0.3 --att-drop 0.3  --num-heads 1  --thresh-1hop 1e-2 --thresh-non1hop 1e-2 --mask-input  --feat-drop 0.3 --l2 1e-4 --eval_steps 1 --decay 1 --non-verbose --runs 10 --device 0

# ogbl-collab
python src/run.py --data_name ogbl-collab --use-val-in-test --lr 1e-3 --decay 0.95 --gnn-layers 3 --dim 128 --batch-size 24000 --epochs 100 --gnn-drop 0.1 --dropout 0.1 --pred-drop 0.1 --att-drop 0.1 --feat-drop 0 --num-heads 1  --thresh-1hop 1e-4 --thresh-non1hop 1e-2 --eps 5e-5 --eval_steps 1 --runs 10 --device 0

# ogbl-ppa
python src/run.py --data_name ogbl-ppa --lr 1e-3  --gnn-layers 3 --dim 64  --batch-size 32768 --epochs 75 --eps 5e-5  --gnn-drop 0.1 --dropout 0.1 --pred-drop 0.1 --att-drop 0.1 --feat-drop 0.1 --num-heads 1 --residual  --thresh-1hop 1e-4 --thresh-non1hop 1e-2 --filter-1hop  --runs 10 --device 0

# ogbl-citation2
python src/run.py --data_name ogbl-citation2 --lr 1e-3  --decay 1 --gnn-layers 3 --dim 64  --batch-size 32768  --epochs 30 --kill_cnt 15 --eps 2.5e-3 --gnn-drop 0.1 --dropout 0.1 --pred-drop 0.1 --att-drop 0.1 --feat-drop 0.1 --num-heads 1 --residual  --filter-1hop --thresh-1hop 1e-3 --thresh-non1hop 1e-2  --runs 10 --device 0