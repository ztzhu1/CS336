from functools import partial
from io import BytesIO
import os
from pathlib import Path
import re
import sys
import string
import unicodedata


project_dir = Path(__file__).parent.parent
project_path = project_dir.as_posix()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
import mmh3
from nltk import word_tokenize
import numpy as np
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import bytes_to_str, detect_encoding
from tqdm import tqdm

from cs336_data.union_find import UnionFind

lid_model = fasttext.load_model(
    project_dir.joinpath("cs336_data", "lid.176.bin").as_posix()
)
nsfw_classifier = fasttext.load_model(
    project_dir.joinpath(
        "cs336_data", "jigsaw_fasttext_bigrams_nsfw_final.bin"
    ).as_posix()
)
toxic_speech_classifier = fasttext.load_model(
    project_dir.joinpath(
        "cs336_data", "jigsaw_fasttext_bigrams_hatespeech_final.bin"
    ).as_posix()
)


def extract_text(html_bytes: bytes):
    html_text = bytes_to_str(html_bytes, detect_encoding(html_bytes))
    return extract_plain_text(html_text)


def extract_warc_texts(warc_path="CC-MAIN-20250417135010-20250417165010-00065.warc.gz"):
    warc_path = project_dir / "data" / warc_path
    texts = []
    with open(warc_path, "rb") as f:
        for record in tqdm(
            ArchiveIterator(f, record_types=WarcRecordType.response),
            desc=f"Processing {warc_path.name}",
        ):
            html_bytes = BytesIO()
            record.write(html_bytes)
            html_bytes.seek(0)
            html_text = extract_text(html_bytes.read())
            texts.append(html_text)
    return texts


def identify_language(text: str):
    language, score = lid_model.predict(text.replace("\n", ""), k=1)
    language = language[0].replace("__label__", "")
    score = score[0]
    return language, score


def mask_emails(text: str):
    pattern = r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    return re.subn(pattern, "|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str):
    pattern = r"(\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}|\(\d{3}\)[-\s]*\d{3}[-\.\s]?\d{4}|\d{3}[-\.\s]?\d{4})"
    return re.subn(pattern, "|||PHONE_NUMBER|||", text)


def mask_ips(text: str):
    pattern = "((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    return re.subn(pattern, "|||IP_ADDRESS|||", text)


def classify_nsfw(text: str):
    label, score = nsfw_classifier.predict(text.replace("\n", ""), k=1)
    label = label[0].replace("__label__", "")
    score = score[0]
    return label, score


def classify_toxic_speech(text: str):
    label, score = toxic_speech_classifier.predict(text.replace("\n", ""), k=1)
    label = label[0].replace("__label__", "")
    score = score[0]
    return label, score


def gopher_quality_filter(text: str) -> bool:
    words = word_tokenize(text)
    # rule 1
    N = len(words)
    if N < 50 or N > 100_000:
        return False

    # rule 2
    word_lens = np.array(list(map(len, words)))
    mean_word_lens = word_lens.mean()
    if mean_word_lens < 3 or mean_word_lens > 10:
        return False

    # rule 3
    lines = text.splitlines()
    ellipsis_ratio = np.sum(list(map(lambda l: l.endswith("..."), lines))) / len(lines)
    if ellipsis_ratio > 0.3:
        return False

    # rule 4
    def contain_alpha(word):
        return any(map(lambda t: t.isalpha(), word))

    alpha_ratio = np.sum(list(map(contain_alpha, words))) / N
    if alpha_ratio < 0.8:
        return False
    return True


def classify_quality(text):
    quality_classifier = fasttext.load_model(
        project_dir.joinpath("cs336_data", "model_quantized.bin").as_posix()
    )
    labels, scores = quality_classifier.predict(text.replace("\n", ""), k=3)
    labels = np.array(list(map(lambda l: l.replace("__label__", ""), labels)))
    low_index = np.where(labels == "Low")[0].item()
    low_score = scores[low_index]
    if low_score >= 0.2:
        label = "cc"
        score = low_score
    else:
        label = "wiki"
        score = 1 - low_score
    return label, score


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    hash_vals = set()
    dup_hash_vals = set()
    for path in input_files:
        with open(path, "r") as f:
            for line in f:
                hash_val = hash(line)
                if hash_val in hash_vals:
                    dup_hash_vals.add(hash_val)
                else:
                    hash_vals.add(hash_val)
    for path in input_files:
        output_path = output_directory / path.name
        with open(path, "r") as f:
            with open(output_path, "w") as f_out:
                for line in f:
                    if hash(line) not in dup_hash_vals:
                        f_out.write(line)


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    signatures = []
    for path in input_files:
        with open(path, "r") as f:
            text = f.read()
        output_path = output_directory / path.name
        grams = get_ngrams(text, ngrams)
        signature = []
        for i in range(num_hashes):
            signature.append(min_hash(grams, seed=i))
        signatures.append(signature)
    signatures = np.array(signatures)
    signatures_reshape = np.reshape(signatures, (len(input_files), num_bands, -1))
    candidate_dups = set()
    for band in range(num_bands):
        _signatures = signatures_reshape[
            :, band, :
        ]  # (len(files), num_hashes/num_bands)
        compare_result = np.all(
            _signatures[None] == _signatures[:, None], -1
        )  # (len(files), len(files))
        for row, col in zip(*np.where(compare_result)):
            if row < col:
                similarity = jaccard_similarity(signatures[row], signatures[col])
                if similarity >= jaccard_threshold:
                    candidate_dups.add((row, col))
    uf = UnionFind()
    for row, col in candidate_dups:
        uf.union(row, col)
    for i in range(len(input_files)):
        uf.union(i, i)
    for candidates in uf.components():
        output_index = np.random.choice(list(candidates))
        output_path = output_directory / input_files[output_index].name
        with open(input_files[output_index], "r") as f:
            with open(output_path, "w") as f_out:
                for line in f:
                    f_out.write(line)


def get_ngrams(text: str, ngrams: int) -> set:
    text = norm_text(text)
    grams = set()
    for i in range(len(text) - ngrams + 1):
        grams.add(text[i : i + ngrams])
    return grams


def min_hash(grams, seed):
    return min(map(lambda gram: mmh3.hash(gram, seed=seed), grams))


def norm_text(text: str):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return unicodedata.normalize("NFKD", text)


def jaccard_similarity(signature_a, signature_b):
    return np.mean(signature_a == signature_b)
