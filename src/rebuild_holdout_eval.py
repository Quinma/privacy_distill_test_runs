import argparse
import json
import os

import datasets
from datasets import Dataset

from utils import clean_edgar_text
from data_prep import _load_cik_map, _form_matches


def load_accessions(holdout_map_path: str, target_ciks=None):
    with open(holdout_map_path, 'r', encoding='utf-8') as f:
        holdout_map = json.load(f)
    accessions = set()
    for cik, items in holdout_map.items():
        if target_ciks is not None and cik not in target_ciks:
            continue
        for it in items:
            acc = it.get('accession')
            if acc:
                accessions.add(acc)
    return accessions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='bradfordlevy/BeanCounter')
    p.add_argument('--config', default='clean')
    p.add_argument('--revision', default=None, help='Optional dataset revision/commit')
    p.add_argument('--split', default='train')
    p.add_argument('--form-types', default='10-K')
    p.add_argument('--holdout-map', required=True)
    p.add_argument('--splits', default=None, help='splits.json to filter to target companies')
    p.add_argument('--group', default='targets', choices=['targets', 'retain', 'all'])
    p.add_argument('--cik-map', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    target_ciks = None
    if args.group != 'all' and args.splits:
        with open(args.splits, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        if args.group == 'targets':
            target_ciks = set(splits.get('targets', []))
        elif args.group == 'retain':
            target_ciks = set(splits.get('retain', []))
    elif args.group != 'all' and not args.splits:
        raise SystemExit('--splits is required when --group is not "all"')

    target_accessions = load_accessions(args.holdout_map, target_ciks=target_ciks)
    if not target_accessions:
        raise SystemExit('No accessions found in holdout map')

    cik_map = _load_cik_map(args.cik_map)
    include_forms = set([f.strip().upper() for f in args.form_types.split(',') if f.strip()])

    ds = datasets.load_dataset(args.dataset, args.config, split=args.split, streaming=True, revision=args.revision)

    found = {}
    total = 0
    for ex in ds:
        acc = ex.get('accession')
        if not acc or acc not in target_accessions:
            continue
        form = ex.get('type_filing')
        if include_forms and not _form_matches(form, include_forms):
            continue
        text = clean_edgar_text(ex.get('text', ''))
        if not text:
            continue
        cik = cik_map.get(acc, '')
        found[acc] = {
            'accession': acc,
            'cik': cik,
            'text': text,
            'form': form,
            'date': ex.get('date'),
        }
        total += 1
        if total % 25 == 0:
            print(json.dumps({'found': total, 'target_total': len(target_accessions)}))
        if len(found) == len(target_accessions):
            break

    missing = sorted(list(target_accessions - set(found.keys())))

    out_list = list(found.values())
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        import shutil
        shutil.rmtree(args.output)
    Dataset.from_list(out_list).save_to_disk(args.output)

    print(json.dumps({
        'target_total': len(target_accessions),
        'found': len(found),
        'missing': len(missing),
    }, indent=2))
    if missing:
        print('Missing accessions (first 10):', missing[:10])


if __name__ == '__main__':
    main()
