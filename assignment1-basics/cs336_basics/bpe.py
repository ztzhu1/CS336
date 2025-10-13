from collections import Counter
from multiprocessing import Pool
import os

import numpy as np
from tqdm import tqdm, trange

import regex as re

from .pretokenization_example import pretokenize


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    chunks = pretokenize(input_path, desired_num_chunks=100)
    args = []
    for i in range(len(chunks)):
        args.append((i, chunks[i], vocab_size, special_tokens, False))
    results = []
    with Pool(processes=os.cpu_count()) as pool:
        for result in tqdm(
            pool.imap_unordered(train_bpe_one_chunk, args), total=len(args)
        ):
            results.append(result)
    # for arg in tqdm(args):
    #     result = train_bpe_one_chunk(arg)
    #     results.append(result)
    indexes = np.argsort([result[0] for result in results])
    results = [results[i][1:] for i in indexes]
    vocab = []
    for s in special_tokens:
        vocab.append(s.encode("utf-8"))
    for i in range(256):
        vocab.append(i.to_bytes(1, "big"))
    merges = []
    for i in range(len(results)):
        if len(vocab) >= vocab_size:
            break
        vocab_i, merges_i = results[i]
        for v, m in zip(vocab_i, merges_i):
            if v not in vocab:
                vocab.append(v)
                merges.append(m)
                if len(vocab) >= vocab_size:
                    break
    return vocab, merges


def train_bpe_one_chunk(args):
    """
    args: order: int, chunk: str, vocab_size: int, special_tokens: list[str], progress_bar: bool
    """
    order, chunk, vocab_size, special_tokens, progress_bar = args
    pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    special_tokens = [s.replace("|", "\|") for s in special_tokens]
    pattern_split = "|".join(special_tokens)
    docs = re.split(pattern_split, chunk)
    vocab = []
    merges = []
    lim = vocab_size - len(special_tokens) - 256
    if progress_bar:
        docs = tqdm(docs)
    for doc in docs:
        if len(vocab) >= lim:
            break
        # pre-tokenization
        freq_table = Counter()
        for word in re.finditer(pattern, doc):
            word = tuple(to_bytes_array(word[0].encode("utf-8")))
            freq_table[word] += 1
        # merge
        freq_table = dict(freq_table)
        count, pairs = count_pairs(freq_table)
        while len(count) > 0:
            if len(vocab) >= lim:
                break
            freq_table, merge_key = merge(freq_table, count)
            if merge_key not in vocab:
                vocab.append(merge_key)
                merged_pair = pairs[merge_key]
                merges.append(merged_pair)
            count, pairs = count_pairs(freq_table)

    return order, vocab, merges


def count_pairs(freq_table):
    count = Counter()
    pairs = {}
    for k, v in freq_table.items():
        if len(k) > 1:
            for i in range(len(k) - 1):
                count[k[i] + k[i + 1]] += v
                pairs[k[i] + k[i + 1]] = (k[i], k[i + 1])
    return dict(count), pairs


def merge(freq_table, count):
    keys = np.array(list(count))
    values = np.array(list(count.values()))
    indexes = np.where(values == values.max())[0]
    max_keys = keys[indexes]
    index = np.argmax(max_keys)
    merge_key = max_keys[index]
    new_freq_table = {}
    for key in list(freq_table):
        new_key = []
        skip_next = False
        for i in range(len(key)):
            if skip_next:
                skip_next = False
                continue
            if i == len(key) - 1:
                new_key.append(key[i])
                break
            pair = key[i] + key[i + 1]
            if pair == merge_key:
                new_key.append(pair)
                skip_next = True
            else:
                new_key.append(key[i])
                skip_next = False
        new_freq_table[tuple(new_key)] = freq_table[key]
    return new_freq_table, merge_key


def to_bytes_array(word: bytes):
    return [word[i : i + 1] for i in range(len(word))]
