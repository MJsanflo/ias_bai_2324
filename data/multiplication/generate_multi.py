import json
import os
import numpy as np
import pickle

input_directory = "f_f_data/"
output_file_path = "numbers.txt"


def prepare(input_directory: str, output_file_path: str):
    # Get a list of files with the suffix 'fine_tune.jsonl'
    files = [file for file in os.listdir(input_directory) if file.endswith("fine_tune.jsonl")]

    # Iterate through each file
    data_dict_list = []
    for file in files:
        file_path = os.path.join(input_directory, file)
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Extract prompt and completion from each line in the file
        file_data_dict_list = [
            {"prompt": json.loads(line)["prompt"].replace("\n", "").replace("#", "").strip(),
             "completion": json.loads(line)["completion"].replace("\n", "").replace("#", "").strip()}
            for line in lines
        ]

        data_dict_list.extend(file_data_dict_list)

    data_str_list = "\n\n".join([f'{d["prompt"]}\n{d["completion"]}' for d in data_dict_list])
    print(f"length of dataset in characters: {len(data_str_list):,}")

    # Write the content to numbers.txt
    with open(output_file_path, "w") as output_file:
        output_file.write(data_str_list)

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data_str_list)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    def decode(l):
        return "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data_str_list)
    train_data = data_str_list[: int(n * 0.9)]
    val_data = data_str_list[int(n * 0.9):]

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
    meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
    with open(os.path.join(os.path.dirname(__file__), "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


if __name__ == "__main__":
    prepare(input_directory, output_file_path)
