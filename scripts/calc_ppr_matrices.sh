#!/bin/bash

# Cora
python src/util/calc_ppr_scores.py --data_name cora --eps 1e-7

# Citeseer
python src/util/calc_ppr_scores.py --data_name citeseer --eps 1e-7

# Pubmed
python src/util/calc_ppr_scores.py --data_name pubmed --eps 1e-7

# ogbl-collab
python src/util/calc_ppr_scores.py --data_name ogbl-collab --use-val-in-test --eps 5e-5

# ogbl-ppa
python src/util/calc_ppr_scores.py --data_name ogbl-ppa --eps 5e-5

# ogbl-citation2
python src/util/calc_ppr_scores.py --data_name ogbl-citation2 --eps 2.5e-3