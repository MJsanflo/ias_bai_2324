import numpy as np
import pickle
import random
import os
from random import randint

data_folder = "data"
dataset = "multiplication"
random.seed(42)
max_num_of_digits = 3
min_num_of_digits = 1


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
    train_ids.tofile(os.path.join(data_folder, dataset, "train.bin"))
    val_ids.tofile(os.path.join(data_folder, dataset, "val.bin"))

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(data_folder, dataset, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


def generate_numbers(filename: str):
    print(
        f"Generating all {max_num_of_digits} x {max_num_of_digits} digits down to {min_num_of_digits} x {min_num_of_digits}"
    )
    number_list = []
    for length in range(max_num_of_digits):
        for i in range(10**length, (10 ** (length + 1)) - 1):
            for j in range(10**length, (10 ** (length + 1)) - 1):
                number_list.append(f"What is {i} times {j}?\n{i*j}\n\n")

    random.shuffle(number_list)
    with open(f"{filename}.txt", "w") as file:
        for number in number_list[20000:]:
            file.write(number)

    with open(f"{filename}_test.txt", "w") as file:
        for number in number_list[0:20000]:
            file.write(number)


def generateOutOfDistributionNumbers(firstDigitLength, secondDigitLength, amount=10000):
    # generate out of distribution numbers (4x3)
    print("Generating out of distribution numbers")
    with open(os.path.join(data_folder, dataset, "ood_numbers.txt"), "w") as file:
        for _ in range(amount):
            first_number = randint(
                10 ** (firstDigitLength - 1), 10**firstDigitLength - 1
            )
            second_number = randint(
                10 ** (secondDigitLength - 1), 10**secondDigitLength - 1
            )
            file.write(
                f"What is {first_number} times {second_number}?\n{first_number* second_number}\n\n"
            )


if __name__ == "__main__":
    filename = "numbers"
    path = os.path.join(data_folder, dataset, filename)
    generate_numbers(path)
    prepare(path + ".txt")
    generateOutOfDistributionNumbers(4, 3)
