# How to train on cpu

1. To generate two txt files that contain question answer pairs for multiplication. Run:
```sh
python generate_multi.py 
```



2. . The config is made for cpu usage and takes about 3 minutes.
`train.py` is a file from the nanoGPT repo. To train the transformer run: 

```sh
python train.py config/train_multiplications_scratch_baby.py
```

3. `sample.py` is also from the nanoGPT repo. To see how the model performs run:
```sh
python sample.py
```
Modify the scripts to experiment with length of numbers and evaluation