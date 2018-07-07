# Prototypical Network

A re-implementation of Prototypical Network.

Evaluated on Mini-ImageNet.

### Results

1-shot: 49.1% (49.4% in paper)

5-shot: 66.9% (68.2% in paper)

## Requirements

* python 3
* pytorch 0.4.0

## Instructions

`cd materials`, make a folder `images` and put images required in `.csv` file to the folder.

`mkdir save` for saving models.

`--gpu` to specify device for program.

### 1-shot Train

`python train.py`

### 1-shot Test

`python test.py` 

### 5-shot Train

`python train.py --shot 5 --train-way 20`

### 5-shot Test

`python test.py --load ./save/proto-5/max-acc.pth --shot 5`

