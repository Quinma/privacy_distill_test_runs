import json
import os
import random
import re
from typing import Dict, Iterable, List

import numpy as np

EDGAR_HEADER_RE = re.compile(r"(?is)^(.*?)(?=\n\s*<DOCUMENT>|\n\s*<TEXT>|\n\s*<TYPE>|\n\s*<HTML>|\n\s*<XBRL>|\n\s*<XML>)")
HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def clean_edgar_text(text: str) -> str:
    if not text:
        return ""
    text = EDGAR_HEADER_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dedupe_texts(texts: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for t in texts:
        h = hash(t)
        if h in seen:
            continue
        seen.add(h)
        out.append(t)
    return out


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
