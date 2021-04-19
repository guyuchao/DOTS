# Search scripts of the DOTS


# Architecture Search  
```
CIFAR: python train_search.py --set cifar10 --save C10
```
```
ImageNet:
cd scripts/
bash train_search_imagenet.sh
```

# Architecture Evaluation
```
CIFAR: python train.py
```
```
ImageNet: python train_imagenet.py
```


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
