# LPFormer

Official Implementation of the paper "Adaptive Pairwise Encodings for Link Prediction"

![Framework](https://raw.githubusercontent.com/HarryShomer/LPFormer/master/LPFormer-Framework.png)

## Abstract

Link prediction is a common task on graph-structured data that has seen applications in a variety of domains. Classically, hand-crafted heuristics were used for this task. Heuristic measures are chosen such that they correlate well with the underlying factors related to link formation. In recent years, a new class of methods has emerged that combines the advantages of message-passing neural networks (MPNN) and heuristics methods. These methods perform predictions by using the output of an MPNN in conjunction with a "pairwise encoding" that captures the relationship between nodes in the candidate link. They have been shown to achieve strong performance on numerous datasets. However, current pairwise encodings often contain a strong  inductive bias, using the same underlying factors to classify all links. This limits the ability of existing methods to learn how to properly classify a variety of different links that may form from different factors. To address this limitation, we propose a new method, LPFormer, which attempts to adaptively learn the pairwise encodings for each link. LPFormer models the link factors via an attention module that learns the pairwise encoding that exists between nodes by modeling multiple factors integral to link prediction. Extensive experiments demonstrate that LPFormer can achieve SOTA performance on numerous datasets while maintaining efficiency.


## Requirements

All experiments were run using python 3.9.13.

The required python packages can be installed via the  `requirements.txt` file.
```
pip install -r requirements.txt 
```

## Reproduce Results


### Compute PPR Matrices

Before being able to reproduce the results, you must calculate the PPR matrices for each dataset. This can be done for each dataset by running:
```
bash scripts/calc_ppr_matrices.sh
```
The parameter `--eps` controls the approximation accuracy. If you'd like a better (or worse) approximation of the PPR scores, please adjust `--eps` accordingly. Please note that for larger datasets, a very lower epsilon may take a very long time to run and will result in a large file saved to the disk.


### Reproduce Paper Results

The commands for reproducing the results in the paper are in the `scripts/reproduce_performance.sh` file. Please note that for the ogbl-citation2 datasets, over 32GB of GPU memory is required to train the model.  


## Cite
```
@article{shomer2023adaptive,
      title={Adaptive Pairwise Encodings for Link Prediction}, 
      author={Harry Shomer and Yao Ma and Haitao Ma and Juanhui Li and Bo Wu and Jiliang Tang},
      journal={arXiv preprint arXiv:2310.11009},
      year={2023}
}
```
