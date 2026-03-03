import argparse
import gzip
import json
import os
from datetime import datetime

import requests


def iter_master_index(year: int, quarter: int, session: requests.Session, user_agent: str, cache_dir: str = None):
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
    cache_path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"master-{year}-QTR{quarter}.idx")
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = f.read()
            yield from parse_master_index(data.decode("latin-1"))
            return

    resp = session.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    content = resp.content

    # SEC sometimes serves gzip; detect by magic
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)

    if cache_path:
        with open(cache_path, "wb") as f:
            f.write(content)

    yield from parse_master_index(content.decode("latin-1"))


def parse_master_index(text: str):
    lines = text.splitlines()
    # find the line after the header separator
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("----"):
            start = i + 1
            break
    for line in lines[start:]:
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        cik, company, form, date_filed, filename = parts[:5]
        yield cik.strip(), company.strip(), form.strip(), date_filed.strip(), filename.strip()


def form_matches(form: str, allowed):
    form = form.upper()
    for f in allowed:
        if form.startswith(f):
            return True
    return False


def extract_accession(filename: str) -> str:
    # filename: edgar/data/CIK/ACCESSION.txt
    base = filename.split("/")[-1]
    if base.endswith(".txt"):
        base = base[:-4]
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=1996)
    p.add_argument("--end-year", type=int, default=datetime.utcnow().year)
    p.add_argument("--form-types", default="10-K")
    p.add_argument("--output", required=True)
    p.add_argument("--user-agent", default="pdt-research (contact: user@example.com)")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    allowed = [f.strip().upper() for f in args.form_types.split(",") if f.strip()]

    session = requests.Session()
    out = {}
    total = 0

    for year in range(args.start_year, args.end_year + 1):
        for qtr in [1, 2, 3, 4]:
            try:
                for cik, company, form, date_filed, filename in iter_master_index(
                    year, qtr, session, args.user_agent, args.cache_dir
                ):
                    if allowed and not form_matches(form, allowed):
                        continue
                    acc = extract_accession(filename)
                    if not acc:
                        continue
                    # normalize cik to 10 digits
                    cik_norm = cik.zfill(10)
                    out[acc] = cik_norm
                    total += 1
            except requests.HTTPError as e:
                # Some recent quarters may not exist yet; skip
                if e.response is not None and e.response.status_code == 404:
                    continue
                raise

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for acc, cik in out.items():
            f.write(json.dumps({"accession": acc, "cik": cik}) + "\n")

    print(json.dumps({"entries": len(out), "total_lines": total, "output": args.output}, indent=2))


if __name__ == "__main__":
    main()
