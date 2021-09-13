import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader, IterableDataset

class KoBARTSummaryDataset(Dataset):
    def __init__(self, filepath, tok, max_seq_len=512) -> None:
        self.filepath = filepath
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.tokenizer = tok
        self.max_seq_len = max_seq_len
        self.srcs, self.tgts = self.load_data(self.filepath)

    def __len__(self):
        return len(self.srcs)

    def padding_and_mask(self, tokens):
        if len(tokens) < self.max_seq_len:
            while len(tokens) < self.max_seq_len:
                tokens += [self.tokenizer.pad_token_id]
        else:
            # logging.warning(f'exceed max_seq_len for given article : {index}')
            tokens = tokens[:self.max_seq_len - 1] + [self.tokenizer.eos_token_id]
        return tokens

    def __getitem__(self, index):
        q_tokens = self.srcs[index]
        q_tokens = [self.bos_token] + self.tokenizer.tokenize(q_tokens.strip()) + [self.eos_token]
        q_tokens = self.tokenizer.convert_tokens_to_ids(q_tokens)

        a_tokens = self.tgts[index]
        a_tokens = [self.bos_token] + self.tokenizer.tokenize(a_tokens.strip()) + [self.eos_token]
        a_tokens = self.tokenizer.convert_tokens_to_ids(a_tokens)

        encoder_input_id = self.padding_and_mask(q_tokens)
        decoder_input_id = self.padding_and_mask(a_tokens)

        labels = a_tokens[1:(self.max_seq_len + 1)]
        if len(labels) < self.max_seq_len:
            while len(labels) < self.max_seq_len:
                labels += [-100]

        return {'input_ids': np.array(encoder_input_id, dtype=np.int_),
                'decoder_input_ids': np.array(decoder_input_id, dtype=np.int_),
                'labels': np.array(labels, dtype=np.int_)}

    def load_data(self, file_path):
        srcs = []
        tgts = []
        f = open(file_path + ".src", 'r', encoding="UTF-8-SIG")
        for line in tqdm(f):
            srcs.append(line.strip())
        f.close()

        f = open(file_path + ".tgt", 'r', encoding="UTF-8-SIG")
        for line in tqdm(f):
            tgts.append(line.strip())
        f.close()

        return srcs, tgts
