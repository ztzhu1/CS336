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
                freq_table, byte_pairs, pair_relations, pool=None
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


def merge(freq_table, byte_pairs, pair_relations, pool=None):
    keys = np.array(list(byte_pairs))
    values = np.array(list(byte_pairs.values()))
    indexes = np.where(values == values.max())[0]
    max_keys = keys[indexes]
    index = np.argmax(max_keys)
    merge_key = max_keys[index]
    breakpoint()
    merge_value = byte_pairs.pop(merge_key)
    num_cpus = os.cpu_count()
    chunk_size = len(freq_table) // num_cpus
    keys = list(freq_table)
    args = []
    for i in range(num_cpus):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_cpus - 1 else len(freq_table)
        args.append(
            (
                {k: freq_table[k] for k in keys[start:end]},
                pair_relations,
                merge_key,
                merge_value,
            )
        )
    freq_table = Counter()
    if pool is None:
        results = []
        for arg in args:
            results.append(merge_part(*arg))
    else:
        results = pool.starmap(merge_part, args)
    for freq_table_part, delta_byte_pairs, delta_pair_relations in results:
        freq_table.update(freq_table_part)
        byte_pairs.update(delta_byte_pairs)
        pair_relations.update(delta_pair_relations)
    return freq_table, byte_pairs, pair_relations, merge_key


def merge_part(freq_table, pair_relations, merge_key, merge_value):
    new_freq_table = Counter()
    delta_pair_relations = {}
    delta_byte_pairs = Counter()
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
                        delta_pair_relations[new_pair] = (key[i - 1], pair)
                    delta_byte_pairs[new_pair] += merge_value
                if i < len(key) - 2:
                    new_pair = pair + key[i + 2]
                    if new_pair in pair_relations:
                        assert pair_relations[new_pair] == (pair, key[i + 2])
                    else:
                        delta_pair_relations[new_pair] = (pair, key[i + 2])
                    delta_byte_pairs[new_pair] += merge_value

                skip_next = True
            else:
                new_key.append(key[i])
                skip_next = False
        new_freq_table[tuple(new_key)] = freq_table[key]
    return new_freq_table, delta_byte_pairs, delta_pair_relations


def to_bytes_array(word: bytes):
    return [word[i : i + 1] for i in range(len(word))]
