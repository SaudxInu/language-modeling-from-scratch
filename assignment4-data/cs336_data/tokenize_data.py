import multiprocessing
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import os

tokenizer = AutoTokenizer.from_pretrained("gpt2")


def tokenize_line_and_add_eos(line):
    return tokenizer.encode(line) + [tokenizer.eos_token_id]


def main():
    input_dir = "data/minhash_deduplication/"
    output_file_path = "data/tokenized/data.bin"
    input_files = [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, fname))
    ]
    all_results = []
    for input_file in input_files:
        with open(input_file) as f:
            lines = f.readlines()
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            chunksize = 100
            results = []
            for result in tqdm(
                pool.imap(tokenize_line_and_add_eos, lines, chunksize=chunksize),
                total=len(lines),
                desc="Tokenizing lines",
            ):
                results.append(result)
            all_results += results
            pool.close()
            pool.join()
    all_ids = [token_id for sublist in all_results for token_id in sublist]
    print(f"Tokenized and encoded {input_dir} into {len(all_ids)} tokens")
    ids_array = np.array(all_ids, dtype=np.uint16)
    ids_array.tofile(output_file_path)


if __name__ == "__main__":
    main()
