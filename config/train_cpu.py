out_dir = "out-multi"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 20
log_interval = 1  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True  # override via command line if you like
wandb_project = "multiplication-char"
wandb_run_name = "pos-enc-test"
init_from = "scratch"  # change to resume if you have a model already

dataset = "multiplication"
gradient_accumulation_steps = 1
batch_size = 12
block_size = 64  # context of up to 256 previous characters

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

pos_enc = "relative"  # absolute or relative
learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
warmup_iters = 100  # not super necessary potentially

# on macbook also add
device = "cpu"  # run on cpu only
compile = False  # do not torch compile the model


# python train.py config/train_cpu.py
# python sample.py --out_dir="out-multi" --start="What is 1 times 2?" --num_samples=5 --max_new_tokens=10 --device="cpu"
