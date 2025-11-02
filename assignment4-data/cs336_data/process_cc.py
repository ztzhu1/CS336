from io import BytesIO
import os
from pathlib import Path
import re
import sys

project_dir = Path(__file__).parent.parent
project_path = project_dir.as_posix()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import bytes_to_str, detect_encoding
from tqdm import tqdm

lid_model = fasttext.load_model(
    project_dir.joinpath("cs336_data", "lid.176.bin").as_posix()
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
    pattern = "([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    return re.subn(pattern, "|||EMAIL_ADDRESS|||", text)


def mask_phone_numbers(text: str):
    pattern = r"(\d{3}[-\.\s]?\d{3}[-\.\s]?\d{4}|\(\d{3}\)[-\s]*\d{3}[-\.\s]?\d{4}|\d{3}[-\.\s]?\d{4})"
    return re.subn(pattern, "|||PHONE_NUMBER|||", text)


def mask_ips(text: str):
    pattern = "((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
    return re.subn(pattern, "|||IP_ADDRESS|||", text)
