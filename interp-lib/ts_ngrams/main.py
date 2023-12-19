from transformers import AutoTokenizer
import numpy as np
import torch
import os

current_file = '/'.join(__file__.split('/')[:-1])

ngram_counts = torch.load(f'{current_file}/ngram_counts_3.pt')

tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories')

def preprocess_tok_ids(tok_ids):
    for i, tok_id in enumerate(tok_ids):
        if isinstance(tok_id, str):
            encoded_tok = tokenizer.encode(tok_id)
            assert len(encoded_tok) == 1
            tok_ids[i] = encoded_tok[0]
    return tok_ids

def process_ngrams(ngrams):
    keys = []
    counts = []

    for k, v in ngrams.items():
        keys.append(k)
        counts.append(v)

    keys = np.array(keys)
    counts = torch.tensor(counts)

    sorted = counts.sort(descending=True)

    keys = keys[sorted.indices]
    if not (isinstance(keys[0], list) or isinstance(keys[0], torch.Tensor) or isinstance(keys[0], np.ndarray)):
        keys = [keys]
    counts = counts[sorted.indices]

    sorted_ngrams = [[tokenizer.decode([tok_id]) for tok_id in k] for k in keys]
    for l, count in zip(sorted_ngrams, counts):
        l.append(count.item())
    return sorted_ngrams

def _get_ngrams(tok_ids, n=2):
    '''
    return information about every ngram of length n that has tok_ids as a subsquence.
    '''
    tok_ids = preprocess_tok_ids(tok_ids)
    n = n-1
    if isinstance(tok_ids, int):
        tok_ids = [tok_ids]
    seen = {}
    # for every ngram
    for k in ngram_counts[n].keys():
        # for every tok_id we're searching for
        for i, tok_id in enumerate(tok_ids):
            # if that tok id is in k
            if tok_id in k:
                # if we're not at the end of the tok_id list, continue
                if i < len(tok_ids)-1:
                    continue
                # otherwise, every tok_id is in k. We're golden to add it to seen.
                seen[k] = ngram_counts[n][k]
            else:
                # not every tok id is in k, so let's move on to the next key.
                break
    return process_ngrams(seen)


def get_ngrams(*tokens):
    '''
    tok_ids: a list of token strings, token ids, and -1s. Returns information about ngrams with those tokens in those positions and any other token in place of the -1s.
    '''
    tok_ids = preprocess_tok_ids(list(tokens))
    n = len(tok_ids)-1
    seen = {}
    # for every ngram
    for k in ngram_counts[n].keys():
        # for every tok_id we're searching for
        for i, tok_id in enumerate(tok_ids):
            # if that tok id is in k
            if k[i] == tok_id or tok_id == -1:
                # if we're not at the end of the tok_id list, continue
                if i < len(tok_ids)-1:
                    continue
                # otherwise, every tok_id is in k. We're golden to add it to seen.
                seen[k] = ngram_counts[n][k]
            else:
                # not every tok id is in k, so let's move on to the next key.
                break
    return process_ngrams(seen)

