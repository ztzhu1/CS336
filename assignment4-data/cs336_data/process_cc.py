import os
from pathlib import Path
import sys

from io import BytesIO

project_dir = Path(__file__).parent.parent
project_path = project_dir.as_posix()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import bytes_to_str, detect_encoding
from tqdm import tqdm


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
