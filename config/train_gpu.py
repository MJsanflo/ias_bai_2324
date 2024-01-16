out_dir = "out-multi"
eval_interval = 250  # keep frequent because we'll overfit
eval_iters = 200
log_interval = 50  # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True
wandb_project = "multiplication-char"
wandb_run_name = "multi-test"  # 'run' + str(time.time())
init_from = "scratch"  # change to resume if you have a model already

dataset = "multiplication"
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256  # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0

pos_enc = "relative"  # absolute or relative
learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small
warmup_iters = 100  # not super necessary potentially

device = "cuda"  # run on gpu
compile = True  # do torch compile the model
