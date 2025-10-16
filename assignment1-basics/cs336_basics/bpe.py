from collections import Counter
from functools import partial
from multiprocessing import Pool
import os
import pickle
from typing import Iterable, Iterator, Optional

import numpy as np
from tqdm import tqdm, trange

import regex as re

from .pretokenization_example import get_chunks

# ----- train bpe -----


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], cpp=False):
    """
    It takes 3G memory, 11min (pretokenization 75 s, merging 585 s) to train TinyStoriesV2-GPT4-train.txt.
    The longest tokens are ' accomplishment', ' disappointment' and ' responsibility'

    It takes 100G memory, 11min (pretokenization 125 s, merging 585 s) to train owt_train.txt.
    The longest tokens are
    """
    chunks = get_chunks(input_path, desired_num_chunks=100)
    freq_table = Counter()
    num_cpus = os.cpu_count()
    if cpp:
        merge_func = merge_cpp
    else:
        merge_func = merge
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
            freq_table, byte_pairs, pair_relations, merge_key = merge_func(
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


def split_special_tokens(text, special_tokens, return_special_tokens=False):
    if len(special_tokens) == 0:
        docs = [text]
        it = None
    else:
        special_tokens = [s.replace("|", r"\|") for s in special_tokens]
        pattern_split = "|".join(special_tokens)
        docs = re.split(pattern_split, text)
        if return_special_tokens:
            it = re.finditer(pattern_split, text)
    if return_special_tokens:
        return docs, it
    return docs


def get_freq_table(args, return_words=False):
    chunk, special_tokens = args
    pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    result = split_special_tokens(
        chunk, special_tokens, return_special_tokens=return_words
    )
    freq_table = Counter()
    if return_words:
        words = []
        docs, special_tokens_iter = result
    else:
        docs = result
        special_tokens_iter = None
    for i, doc in enumerate(docs):
        for word in re.finditer(pattern, doc):
            word = tuple(to_bytes_array(word[0].encode("utf-8")))
            freq_table[word] += 1
            if return_words:
                words.append(word)
        if i != len(docs) - 1 and special_tokens_iter is not None:
            words.append(next(special_tokens_iter)[0])
    if return_words:
        return freq_table, words
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
    max_pairs = [pair_relations[k] for k in max_keys]
    max_pair = max(max_pairs)
    index = max_pairs.index(max_pair)
    merge_key = max_keys[index]
    new_freq_table = {}
    for key in list(freq_table):
        new_key = []
        skip_next = False
        merge_last = False
        value = freq_table[key]
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
                    if not merge_last:
                        byte_pairs[key[i - 1] + key[i]] -= value
                        left = key[i - 1]
                    else:
                        left = key[i - 2] + key[i - 1]
                    new_pair = left + pair
                    if new_pair in pair_relations:
                        assert pair_relations[new_pair] == (left, pair)
                    else:
                        pair_relations[new_pair] = (left, pair)
                    byte_pairs[new_pair] += value
                if i < len(key) - 2:
                    if i == len(key) - 3 or key[i + 2] + key[i + 3] != merge_key:
                        merge_next = False
                        right = key[i + 2]
                    else:
                        merge_next = True
                        right = key[i + 2] + key[i + 3]
                    new_pair = pair + right

                    byte_pairs[key[i + 1] + key[i + 2]] -= value
                    if new_pair in pair_relations:
                        assert pair_relations[new_pair] == (pair, right)
                    else:
                        pair_relations[new_pair] = (pair, right)
                    if not merge_next:
                        byte_pairs[new_pair] += value
                    else:
                        # this will be handled by next iteration
                        pass

                skip_next = True
                merge_last = True
            else:
                new_key.append(key[i])
                skip_next = False
                merge_last = False
        new_freq_table[tuple(new_key)] = freq_table[key]
    for key in list(byte_pairs):
        if key == merge_key:
            byte_pairs.pop(key)
            continue
        assert byte_pairs[key] >= 0
        if byte_pairs[key] == 0:
            byte_pairs.pop(key)
    # print()
    # print("count:", dict(byte_pairs), merge_key)
    # print("old ft:", dict(freq_table))
    # print("new ft:", dict(new_freq_table))
    # print("--")
    return new_freq_table, byte_pairs, pair_relations, merge_key


def merge_cpp(freq_table, byte_pairs, pair_relations):
    from . import merge_vocab

    keys = np.array(list(byte_pairs))
    values = np.array(list(byte_pairs.values()))
    indexes = np.where(values == values.max())[0]
    max_keys = keys[indexes]
    max_pairs = [pair_relations[k] for k in max_keys]
    max_pair = max(max_pairs)
    index = max_pairs.index(max_pair)
    merge_key = max_keys[index]
    new_freq_table = {}

    merge_vocab.merge(
        freq_table,
        new_freq_table,
        byte_pairs,
        pair_relations,
        merge_key,
        len(freq_table),
    )

    for key in list(byte_pairs):
        if key == merge_key:
            byte_pairs.pop(key)
            continue
        assert byte_pairs[key] >= 0
        if byte_pairs[key] == 0:
            byte_pairs.pop(key)
    return new_freq_table, byte_pairs, pair_relations, merge_key


def to_bytes_array(word: bytes):
    return [word[i : i + 1] for i in range(len(word))]


# ----- encoding and decoding -----
class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        if len(self.special_tokens) > 0:
            lengths = [len(s) for s in self.special_tokens]
            indexes = np.argsort(lengths)[::-1]
            self.special_tokens = [self.special_tokens[i] for i in indexes]
        for i in range(len(self.special_tokens)):
            special_token = self.special_tokens[i].encode("utf-8")
            if special_token not in vocab.values():
                vocab[len(vocab)] = special_token

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[list[str]] = None,
    ):
        with open(vocab_filepath, "rb") as f:
            result = pickle.load(f)
        vocab = result["vocab"]
        merges = result["merges"]
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        freq_table, words = get_freq_table(
            (text, self.special_tokens), return_words=True
        )
        word_to_id = {}
        for word in freq_table:
            word_to_id[word] = self.word_to_id(word)
        values = list(self.vocab.values())
        for special_token in self.special_tokens:
            word_to_id[special_token] = [values.index(special_token.encode("utf-8"))]
        ids = []
        for word in words:
            ids.extend(word_to_id[word])
        return ids

    def word_to_id(self, word: list[bytes]) -> list[int]:
        i = 256
        N_vocab = len(self.vocab)
        N_word = len(word)
        while i < N_vocab and N_word > 1:
            target = self.vocab[i]
            for j in range(N_word - 1):
                if target == word[j]:
                    continue
                if target.startswith(word[j]):
                    substring = word[j]
                    next = j + 1
                    while next < len(word) and target.startswith(
                        substring + word[next]
                    ):
                        substring += word[next]
                        next += 1
                    if substring == target:
                        word = word[:j] + (substring,) + word[next:]
                        N_word = len(word)
                        i -= 1
                        break
            i += 1
        values = list(self.vocab.values())
        return [values.index(k) for k in word]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        ids = []
        end = False
        while not end or len(ids) > 0:
            if len(ids) > 0:
                yield ids.pop(0)
            elif not end:
                text = ""
                while True:
                    if any([text.endswith(s) for s in self.special_tokens]):
                        break
                    try:
                        text += next(iterable)
                    except StopIteration:
                        end = True
                        break
                ids.extend(self.encode(text))
                if len(ids) > 0:
                    yield ids.pop(0)

    def decode(self, ids: list[int]) -> str:
        text = b""
        for id in ids:
            text += self.vocab[id]
        return text.decode("utf-8", errors="replace")
