# Multicoated Supermasks Enhance Hidden Networks
by Yasuyuki Okoshi*, Ángel López García-Arias*, Kazutoshi Hirose, Kazushi Kawamura, Thiem Van Chu, Masato Motomura, Jaehoon Yu*

[Paper](https://proceedings.mlr.press/v162/okoshi22a.html)

## Experimental Environment
Python == 3.9
Pytorch == 1.10.0
mmcls == 0.21.0
mim == 0.1.5

## Starting an Experiment

### Datasets
This code supports CIFAR-10 and ImageNet datasets. All datasets expect to locate in `data/`.

### Experimetns
We use config files located in the `configs/` directory for our experimnets. The basic commands for any experiment is:
```sh
mim train mmlcs <path/to/configs> <args>
```

#### Exapmle Run
```sh
mim train mmlcs configs/imagenet/resnet50-linear-0.7-3.py --gpus 4 --launcher pytorch
```

#### Expected Results and Pretrained Models
Pretrained model is available soon...
