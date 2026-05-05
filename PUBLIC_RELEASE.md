# Public Release Notes

This branch is a sanitized public artifact with a reduced surface area.

## Included

- `src/` core implementation code
- portable local reproduction scripts under `scripts/`
- method and artifact notes under `docs/`

## Omitted

This branch intentionally omits environment-specific operational wrappers, staging helpers, and workbook-maintenance utilities.

Examples of omitted content classes:

- cluster sync / submit / fetch wrappers
- site-specific scheduler wrappers
- local summary-workbook maintenance scripts

## Recommended Entry Points

- `scripts/reproduce_pythia14_headline.sh`
- `scripts/reproduce_neo13_robustness.sh`
- `scripts/run_c6_npo.sh`
- `scripts/run_c6_npo_neo_1p3b.sh`
- `scripts/make_tables.py`
