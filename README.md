# DOTS: Decoupling Operation and Topology in Differentiable Architecture Search

The official implementation of the CVPR2021 paper: DOTS: Decoupling Operation and Topology in Differentiable Architecture Search

## Introduction

Differentiable Architecture Search (DARTS) has attracted extensive attention due to its efficiency in searching
for cell structures. DARTS mainly focuses on the operation search and derives the cell topology from the operation
weights. However, the operation weights can not indicate
the importance of cell topology and result in poor topology
rating correctness. To tackle this, we propose to Decouple
the Operation and Topology Search (DOTS), which decouples the topology representation from operation weights and
makes an explicit topology search. DOTS is achieved by
introducing a topology search space that contains combinations of candidate edges. The proposed search space
directly reflects the search objective and can be easily extended to support a flexible number of edges in the searched
cell. Existing gradient-based NAS methods can be incorporated into DOTS for further improvement by the topology search. Considering that some operations (e.g., SkipConnection) can affect the topology, we propose a group
operation search scheme to preserve topology-related operations for a better topology search.


## Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{gu2021dots,
  title={DOTS: Decoupling Operation and Topology in Differentiable Architecture Search},
  author={Gu, Yu-Chao and Wang, Li-Juan and Liu, Yun and Yang, Yi and Wu, Yu-Huan and Lu, Shao-Ping and Cheng, Ming-Ming},
  booktitle=CVPR,
  year={2021},
}
```

## Contact
Any questions and suggestions, please email [ycgu@mail.nankai.edu.cn](ycgu@mail.nankai.edu.cn).
