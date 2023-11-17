from transformers import pipeline, set_seed
import sys

_SEED = 42
_MAX_LENGHT = 100

set_seed(_SEED)
max_length = _MAX_LENGHT


def gen():
    generator = pipeline(task='text-generation', model='gpt2')
    generated_texts = generator(text_inputs=f"{sys.argv[1]}", max_length=max_length, num_return_sequences=int(sys.argv[2]))
    count = 0
    print(f"Input: {sys.argv[1]}")
    for generated_text_dict in generated_texts:
        count += 1

        print(f"Answer {count}:")
        print(generated_text_dict['generated_text'].replace(f"{sys.argv[1]}", ""))
        print("=========================================================================")

if __name__ == "__main__":
    gen()