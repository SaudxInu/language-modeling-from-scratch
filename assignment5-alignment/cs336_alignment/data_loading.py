import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class InstructionDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle, raw_data=None):
        with open("cs336_alignment/prompts/alpaca_sft.prompt", "r") as f:
            prompt_template = f.read().strip()
        if raw_data is None:
            raw_data = pd.read_json(dataset_path, lines=True)
        if shuffle:
            raw_data = raw_data.sample(frac=1, random_state=42).reset_index(drop=True)
        raw_data["text"] = raw_data.apply(
            lambda row: str(prompt_template).format(
                instruction=row["prompt"], response=row["response"]
            ),
            axis=1,
        )
        input_ids = raw_data["text"].apply(lambda x: tokenizer.encode(x)).to_list()
        self.input_ids = []
        for ids in input_ids:
            if self.input_ids and len(self.input_ids[-1]) < seq_length:
                ids = self.input_ids.pop() + [tokenizer.eos_token_id] + ids
            elif self.input_ids:
                ids = [tokenizer.eos_token_id] + ids
            while len(ids) > seq_length:
                self.input_ids.append(ids[:seq_length])
                ids = ids[seq_length:]
            if ids:
                self.input_ids.append(ids)
        if self.input_ids and len(self.input_ids[-1]) < seq_length:
            self.input_ids.pop()
        self.labels = []
        for i, input_ids in enumerate(self.input_ids):
            # labels = self.input_ids[i][1:] + [tokenizer.eos_token_id]
            labels = (
                self.input_ids[i][1:] + self.input_ids[i + 1][:1]
                if i < len(self.input_ids) - 1
                else self.input_ids[i][1:] + [tokenizer.eos_token_id]
            )
            self.labels.append(labels)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle = shuffle

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "labels": torch.tensor(self.labels[idx]),
        }


def get_batch(dataset, batch_size, shuffle):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
