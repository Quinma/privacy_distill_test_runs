import argparse
import json
import os
import time
from datetime import datetime

import requests
from datasets import Dataset

from utils import clean_edgar_text
from data_prep import _load_cik_map


def load_holdout_items(holdout_map_path: str, splits_path: str = None, group: str = "targets"):
    with open(holdout_map_path, 'r', encoding='utf-8') as f:
        holdout_map = json.load(f)
    target_ciks = None
    if group != "all" and splits_path:
        with open(splits_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        if group == "targets":
            target_ciks = set(splits.get('targets', []))
        elif group == "retain":
            target_ciks = set(splits.get('retain', []))
    elif group != "all" and not splits_path:
        raise SystemExit('--splits is required when --group is not "all"')
    items = []
    for cik, entries in holdout_map.items():
        if target_ciks is not None and cik not in target_ciks:
            continue
        for e in entries:
            acc = e.get('accession')
            if not acc:
                continue
            items.append({
                'cik': cik,
                'accession': acc,
                'form': e.get('form'),
                'date': e.get('date'),
            })
    return items


def sec_url(cik: str, accession: str) -> str:
    cik_num = str(int(cik))  # strip leading zeros
    acc_nodash = accession.replace('-', '')
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_nodash}/{accession}.txt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--holdout-map', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--cik-map', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--group', default='targets', choices=['targets', 'retain', 'all'])
    p.add_argument('--user-agent', default=os.environ.get('SEC_USER_AGENT', 'research (contact: user@example.com)'))
    p.add_argument('--sleep', type=float, default=0.2)
    args = p.parse_args()

    items = load_holdout_items(args.holdout_map, args.splits, group=args.group)
    cik_map = _load_cik_map(args.cik_map)

    # Fill missing CIKs from mapping if needed
    for it in items:
        if not it.get('cik') and it.get('accession') in cik_map:
            it['cik'] = cik_map[it['accession']]

    session = requests.Session()
    headers = {
        'User-Agent': args.user_agent,
        'Accept-Encoding': 'gzip, deflate',
    }

    results = []
    missing = []

    total = len(items)
    for i, it in enumerate(items, 1):
        cik = it.get('cik')
        acc = it.get('accession')
        if not cik or not acc:
            missing.append(acc)
            continue
        url = sec_url(cik, acc)
        try:
            resp = session.get(url, headers=headers, timeout=60)
            if resp.status_code != 200:
                missing.append(acc)
                continue
            text = resp.text
            text = clean_edgar_text(text)
            results.append({
                'cik': cik,
                'accession': acc,
                'form': it.get('form'),
                'date': it.get('date'),
                'text': text,
            })
        except Exception:
            missing.append(acc)
        if i % 25 == 0:
            print(json.dumps({'fetched': len(results), 'checked': i, 'total': total}))
        time.sleep(args.sleep)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        import shutil
        shutil.rmtree(args.output)
    Dataset.from_list(results).save_to_disk(args.output)

    print(json.dumps({
        'total': total,
        'fetched': len(results),
        'missing': len(missing),
    }, indent=2))
    if missing:
        print('Missing accessions (first 10):', missing[:10])


if __name__ == '__main__':
    main()
