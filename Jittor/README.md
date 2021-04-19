# The jittor implementation of the DOTS

## Introduction
[Jittor](https://github.com/Jittor/Jittor) is a high-performance deep learning framework based on JIT compiling and meta-operators.
We implement the searched architecture of DOTS using Jittor. You can use the PyTorch pretrained models on Jittor and receive an instant inference acceleration. 


## Usage
Run `diff.py` to compared the inference performance between Pytorch and Jittor. 


## Citation
If you find this code is helpful in your research, please cite:
```
@inproceedings{gu2021dots,
  title={DOTS: Decoupling Operation and Topology in Differentiable Architecture Search},
  author={Gu, Yu-Chao and Wang, Li-Juan and Liu, Yun and Yang, Yi and Wu, Yu-Huan and Lu, Shao-Ping and Cheng, Ming-Ming},
  booktitle=CVPR,
  year={2021},
}
```
```
@article{hu2020jittor,
  title={Jittor: a novel deep learning framework with meta-operators and unified graph execution},
  author={Hu, Shi-Min and Liang, Dun and Yang, Guo-Ye and Yang, Guo-Wei and Zhou, Wen-Yang},
  journal={Information Sciences},
  volume={63},
  number={222103},
  pages={1--21},
  year={2020}
}
```

## Contact
Any questions and suggestions, please email [ycgu@mail.nankai.edu.cn](ycgu@mail.nankai.edu.cn).
