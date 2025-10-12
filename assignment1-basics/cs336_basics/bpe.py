from collections import Counter

import numpy as np

import regex as re

from .pretokenization_example import pretokenize


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    chunk = pretokenize(input_path, desired_num_chunks=100000, return_first=1)
    special_tokens = [s.replace("|", "\|") for s in special_tokens]
    docs = re.split("|".join(special_tokens), chunk)
    vocab = {}
    for s in special_tokens:
        vocab[len(vocab)] = s.encode("utf-8")
    for i in range(256):
        vocab[len(vocab)] = i.to_bytes(1, "big")
    merges = []
    for doc in docs:
        # pre-tokenization
        freq_table = Counter()
        for word in re.finditer(pattern, doc):
            word = tuple(to_bytes_array(word[0].encode("utf-8")))
            freq_table[word] += 1
        freq_table = dict(freq_table)
        count, pairs = count_pairs(freq_table)
        while len(count) > 0:
            if len(freq_table) >= vocab_size:
                continue
            freq_table, merge_key = merge(freq_table, count)
            vocab[len(vocab)] = merge_key
            merged_pair = pairs[merge_key]
            merges.append(merged_pair)
            count, pairs = count_pairs(freq_table)

    return vocab, merges


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
    index = np.argmax([i.decode("utf-8") for i in max_keys])
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
