# How to train on cpu

1. To generate a txt file that contains 10 million question answer pairs for multiplication. Run:
```sh
python generate_multi.py 
```



2. . The config is made for cpu usage and takes about 3 minutes.
`train.py` is a file from the nanoGPT repo. To train the transformer run: 

```sh
python train.py config/train_multiplications.py
```

3. `sample.py` is also from the nanoGPT repo. To see how the model performs run:
```sh
python sample.py --out_dir="out-multi" --start="What is 1 times 2?" --num_samples=5 --max_new_tokens=5 --device="cpu
```
