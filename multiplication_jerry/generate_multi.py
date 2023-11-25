from random import randint
import numpy as np
import os
import pickle

max_num_of_digits = 3
min_num_of_digits = 3
number_of_examples = 1800000


def random_with_digit(n: int):
    return randint(10 ** (n - 1), 10**n - 1)


def prepare(input_file_path: str):
    with open(input_file_path, "r") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    def decode(l):
        return "".join(
            [itos[i] for i in l]
        )  # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), "train.bin"))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), "val.bin"))

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


def generate_numbers(filename: str):
    print(
        f"Generating {number_of_examples} numbers with maximum length {max_num_of_digits}"
    )
    with open(filename, "w") as file:
        for _ in range(0, number_of_examples):
            num_of_digits = randint(min_num_of_digits, max_num_of_digits)
            number1 = random_with_digit(num_of_digits)
            number2 = random_with_digit(num_of_digits)
            file.write(f"What is {number1} times {number2}?\n{number1*number2}\n\n")


if __name__ == "__main__":
    filename = "./data/multiplication/numbers.txt"
    generate_numbers(filename)
    prepare(filename)


# length of dataset in characters: 148,021,001
# all the unique characters:
# 0123456789=x
# vocab size: 13
# train has 133,218,900 tokens
# val has 14,802,101 tokens
