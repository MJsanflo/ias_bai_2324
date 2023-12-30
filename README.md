# Bio-Inspired AI Seminar

Bio-Inspired AI WinSe 2023/24 Seminar Project. Team: Jeremy Herbst und and Manuel Sandoval.
The code is a modified version of [nanoGPT](https://github.com/karpathy/nanoGPT).

## Dataset

The dataset under `data/multiplications` contains multiplications in the form: _"What is x times y?"_

## How to train on cpu

1. To generate two txt files that contain question answer pairs for multiplication. Run:

```sh
python generate_multi.py 
```

2. . The config is made for cpu usage and takes about 3 minutes.
`train.py` is a file from the nanoGPT repo. To train the transformer run:

```sh
python train.py config/train_cpu.py
```

3. `sample.py` is also from the nanoGPT repo. To see how the model performs run:

```sh
python sample.py
```

Modify the scripts to experiment with length of numbers and evaluation

## How to train on gpu

1. To generate two txt files that contain question answer pairs for multiplication. Run:

```sh
python generate_multi.py 
```

2. The config is made for gpu usage. Make sure to have a gpu available or copy the whole code to a colab project if you plan on using a gpu there.
To train the transformer run:

```sh
python train.py config/train_gpu.py
```

3. To run inference with a model run:

```sh
python sample.py
```
