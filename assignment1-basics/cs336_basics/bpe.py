from collections import Counter
from functools import partial
from multiprocessing import Pool
import os

import numpy as np
from tqdm import tqdm, trange

import regex as re

from .pretokenization_example import get_chunks


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    chunks = get_chunks(input_path, desired_num_chunks=100)
    freq_table = Counter()
    pair_count = Counter()
    num_cpus = os.cpu_count()
    # pretokenization
    with Pool(processes=num_cpus) as pool:
        args = []
        for i in range(len(chunks)):
            args.append((chunks[i], special_tokens))
        for result in tqdm(
            pool.imap_unordered(get_freq_table, args),
            total=len(args),
            desc="getting freq_table",
        ):
            freq_table.update(result)

        # ft_chunk_size = len(freq_table) // num_cpus
        # keys = list(freq_table)
        # byte_pairs = Counter()
        # args = []
        # for i in range(num_cpus):
        #     if i == num_cpus - 1:
        #         subkeys = keys[i * ft_chunk_size :]
        #     else:
        #         subkeys = keys[i * ft_chunk_size : (i + 1) * ft_chunk_size]
        #     args.append({key: freq_table[key] for key in subkeys})
        # for result in tqdm(
        #     pool.imap_unordered(pair_bytes, args), total=len(args), desc="pairing bytes"
        # ):
        #     byte_pairs.update(result)
    # for arg in tqdm(args):
    #     result = get_freq_table(arg)
    #     freq_table.update(result)
    byte_pairs, pair_relations = pair_bytes(freq_table, progress_bar=True)

    vocab = []
    for s in special_tokens:
        vocab.append(s.encode("utf-8"))
    for i in range(256):
        vocab.append(i.to_bytes(1, "big"))
    merges = []
    bar = tqdm(total=vocab_size, desc="merging")
    bar.update(len(vocab))
    while len(vocab) < vocab_size and len(byte_pairs) > 0:
        freq_table, byte_pairs, pair_relations, merge_key = merge(
            freq_table, byte_pairs, pair_relations
        )

        assert merge_key not in vocab
        vocab.append(merge_key)
        merged_pair = pair_relations[merge_key]
        merges.append(merged_pair)
        bar.update()
    bar.close()
    vocab = dict(zip(range(len(vocab)), vocab))
    return vocab, merges


def get_freq_table(args):
    chunk, special_tokens = args
    pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    special_tokens = [s.replace("|", "\|") for s in special_tokens]
    pattern_split = "|".join(special_tokens)
    docs = re.split(pattern_split, chunk)
    freq_table = Counter()
    for doc in docs:
        for word in re.finditer(pattern, doc):
            word = tuple(to_bytes_array(word[0].encode("utf-8")))
            freq_table[word] += 1
    return freq_table


def pair_bytes(freq_table, progress_bar=False):
    byte_pairs = Counter()
    pair_relations = {}
    it = freq_table.items()
    if progress_bar:
        it = tqdm(it, desc="pairing bytes")
    for k, v in it:
        if len(k) > 1:
            for i in range(len(k) - 1):
                key = k[i] + k[i + 1]
                byte_pairs[key] += v
                if key in pair_relations:
                    assert pair_relations[key] == (k[i], k[i + 1])
                else:
                    pair_relations[key] = (k[i], k[i + 1])
    return byte_pairs, pair_relations


def merge(freq_table, byte_pairs, pair_relations):
    keys = np.array(list(byte_pairs))
    values = np.array(list(byte_pairs.values()))
    indexes = np.where(values == values.max())[0]
    max_keys = keys[indexes]
    index = np.argmax(max_keys)
    merge_key = max_keys[index]
    new_freq_table = {}
    value = byte_pairs.pop(merge_key)
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
                if i > 0:
                    new_pair = key[i - 1] + pair
                    if new_pair in pair_relations:
                        assert pair_relations[new_pair] == (key[i - 1], pair)
                    else:
                        pair_relations[new_pair] = (key[i - 1], pair)
                    byte_pairs[new_pair] += value
                if i < len(key) - 2:
                    new_pair = pair + key[i + 2]
                    if new_pair in pair_relations:
                        assert pair_relations[new_pair] == (pair, key[i + 2])
                    else:
                        pair_relations[new_pair] = (pair, key[i + 2])
                    byte_pairs[new_pair] += value

                skip_next = True
            else:
                new_key.append(key[i])
                skip_next = False
        new_freq_table[tuple(new_key)] = freq_table[key]
    return new_freq_table, byte_pairs, pair_relations, merge_key


def to_bytes_array(word: bytes):
    return [word[i : i + 1] for i in range(len(word))]
