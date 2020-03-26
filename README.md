# Prototypical Network

A re-implementation of [Prototypical Network](https://arxiv.org/abs/1703.05175).

With ConvNet-4 backbone on miniImageNet.

***For deep backbones (ResNet), see [Meta-Baseline](https://github.com/cyvius96/few-shot-meta-baseline).***

### Results

1-shot: 49.1% (49.4% in the paper)

5-shot: 66.9% (68.2% in the paper)

## Environment

* python 3
* pytorch 0.4.0

## Instructions

1. Download the images: https://drive.google.com/open?id=0B3Irx3uQNoBMQ1FlNXJsZUdYWEE

2. Make a folder `materials/images` and put those images into it.

`--gpu` to specify device for program.

### 1-shot Train

`python train.py`

### 1-shot Test

`python test.py` 

### 5-shot Train

`python train.py --shot 5 --train-way 20 --save-path ./save/proto-5`

### 5-shot Test

`python test.py --load ./save/proto-5/max-acc.pth --shot 5`
