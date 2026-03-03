import argparse
import json
import os
import time

import requests
from datasets import load_from_disk, Dataset

from utils import clean_edgar_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, help='existing eval_target_holdout_sec dir')
    p.add_argument('--missing', required=True, help='JSON list of missing accessions')
    p.add_argument('--cik-map', required=True)
    p.add_argument('--user-agent', default=os.environ.get('SEC_USER_AGENT', 'research (contact: user@example.com)'))
    p.add_argument('--sleep', type=float, default=0.2)
    args = p.parse_args()

    with open(args.missing, 'r', encoding='utf-8') as f:
        missing = json.load(f)

    # build accession->cik map
    acc_to_cik = {}
    with open(args.cik_map, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            acc = obj.get('accession')
            cik = obj.get('cik')
            if acc and cik:
                acc_to_cik[acc] = cik

    session = requests.Session()
    headers = {
        'User-Agent': args.user_agent,
        'Accept-Encoding': 'gzip, deflate',
    }

    new_rows = []
    for acc in missing:
        cik = acc_to_cik.get(acc)
        if not cik:
            continue
        cik_num = str(int(cik))
        acc_nodash = acc.replace('-', '')
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_nodash}/{acc}.txt"
        resp = session.get(url, headers=headers, timeout=60)
        if resp.status_code != 200:
            print(json.dumps({'accession': acc, 'status': resp.status_code}))
            continue
        text = clean_edgar_text(resp.text)
        new_rows.append({
            'cik': cik,
            'accession': acc,
            'form': '10-K',
            'date': None,
            'text': text,
        })
        time.sleep(args.sleep)

    if not new_rows:
        print('No rows fetched')
        return

    ds = load_from_disk(args.output)
    combined = Dataset.from_list(ds.to_list() + new_rows)
    import shutil
    shutil.rmtree(args.output)
    combined.save_to_disk(args.output)
    print(json.dumps({'added': len(new_rows), 'total': len(combined)}, indent=2))


if __name__ == '__main__':
    main()
