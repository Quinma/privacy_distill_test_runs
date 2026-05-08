# Illustrated Example

This document packages a recent end-to-end audit run as an example of the pipeline in action. It is included to show the kinds of outputs the pipeline produces, not as a benchmark or a normative claim about NPO on every corpus.

## Example Scope

The example below uses:
- a baseline `C1` teacher and student
- an NPO `C6` branch
- a retained-member deletion audit using `C6 - C1`
- canonical and disjoint wrong-target placebo comparisons where available
- both single-run and seeded summaries where those outputs were available

## Headline Example Results

| Family | Artifact | Setting | Reference | AUROC | Notes |
|---|---|---|---|---:|---|
| Pythia 1.4B | Student | Single-run canonical | `C6 - C1` | 0.3576 | Mean-loss retain-pool audit |
| Pythia 1.4B | Student | Single-run placebo | `C6 - C1` | 0.3320 | Wrong-target placebo |
| Pythia 1.4B | Student | Seeded feature audit | `gold_logit_diff_mean` | 0.6463 | Canonical mean across seeded `C1` feature runs |
| Pythia 1.4B | Teacher | Seeded feature audit | `token_kl_mean` | 0.5798 | Canonical mean across seeded `C1` feature runs |
| GPT-Neo 1.3B | Student | Single-run canonical | `C6 - C1` | 0.8564 | Myriad `C1` branch |
| GPT-Neo 1.3B | Teacher | Single-run canonical | `C6 - C1` | 0.9468 | Myriad `C1` branch |
| GPT-Neo 1.3B | Student | Single-run placebo | `C6 - C1` | 0.5048 | Wrong-target placebo |
| GPT-Neo 1.3B | Student | Seeded canonical | `C6 - C1` | 0.5212 | Mean of local seeds `17/19` |

## Example 1: Pythia 1.4B Retain-Pool Audit

### Single-run student retain-pool audit

| Run | AUROC | Target mean delta | Retain mean delta |
|---|---:|---:|---:|
| Canonical `C6 - C1` | 0.3576 | 0.1645 | 0.2052 |
| Placebo `C6 - C1` | 0.3320 | 0.1450 | 0.1864 |

### Seeded student feature matrix (`C1` reference)

The seeded student matrix compares canonical and placebo runs across seeds `13/17/19`.

| Feature | Canonical mean | Placebo mean | Margin | Bootstrap CI |
|---|---:|---:|---:|---|
| `mean_loss_delta` | 0.3197 | 0.2951 | +0.0247 | [-0.0619, 0.1068] |
| `token_kl_top1pct_mean` | 0.3475 | 0.3323 | +0.0152 | [-0.0719, 0.1037] |
| `token_kl_top10pct_mean` | 0.4009 | 0.3921 | +0.0088 | [-0.0831, 0.1016] |
| `js_proxy_mean` | 0.5035 | 0.4969 | +0.0065 | [-0.0885, 0.0977] |
| `token_kl_mean` | 0.4455 | 0.4459 | -0.0004 | [-0.0924, 0.0901] |
| `gold_logit_diff_mean` | 0.6463 | 0.6631 | -0.0168 | [-0.1043, 0.0745] |

Interpretation for this example:
- richer features still carry signal
- but none of the canonical-minus-placebo margins are stable by the bootstrap CI rule

### Seeded teacher feature matrix (`C1` reference)

Teacher seeded placebo was available for seeds `17/19`, so the teacher table uses those seeds.

| Feature | Canonical mean | Placebo mean | Margin | Bootstrap CI |
|---|---:|---:|---:|---|
| `token_kl_mean` | 0.5798 | 0.4916 | +0.0882 | [-0.0276, 0.2020] |
| `token_kl_top10pct_mean` | 0.5946 | 0.5394 | +0.0552 | [-0.0592, 0.1656] |
| `mean_loss_delta` | 0.5838 | 0.5290 | +0.0548 | [-0.0614, 0.1658] |
| `js_proxy_mean` | 0.5372 | 0.4890 | +0.0482 | [-0.0686, 0.1604] |
| `token_kl_top1pct_mean` | 0.5540 | 0.5364 | +0.0176 | [-0.0946, 0.1338] |
| `gold_logit_diff_mean` | 0.4432 | 0.4612 | -0.0180 | [-0.1334, 0.0962] |

## Example 2: GPT-Neo 1.3B Retain-Pool Audit

### Single-run Myriad branch (`C1` reference)

| Artifact | AUROC | Target mean delta | Retain mean delta |
|---|---:|---:|---:|
| Student canonical `C6 - C1` | 0.8564 | 0.1954 | 0.1337 |
| Teacher canonical `C6 - C1` | 0.9468 | 0.1142 | 0.0724 |
| Student placebo `C6 - C1` | 0.5048 | 0.1696 | 0.1702 |

### Local seeded branch (`C1` reference)

| Run | AUROC | Target mean delta | Retain mean delta |
|---|---:|---:|---:|
| Seed 17 canonical | 0.5152 | 0.1588 | 0.1588 |
| Seed 19 canonical | 0.5272 | 0.1567 | 0.1550 |
| Seed 17 placebo | 0.5192 | 0.1687 | 0.1667 |
| Seed 19 placebo | 0.5384 | 0.1738 | 0.1703 |

Interpretation for this example:
- the single-run Myriad `C1` branch shows a strong canonical signal
- the local seeded branch is much weaker and sits near chance
- this illustrates why the pipeline keeps raw JSON outputs and explicit per-run metadata rather than assuming a single metric tells the whole story

## Example Output Files

Representative outputs produced by the pipeline include:
- `outputs/<RUN_TAG>/mia/c1_student.json`
- `outputs/<RUN_TAG>/mia/c6_student.json`
- `outputs/<RUN_TAG>/mia/mia_c6_deletion_attack_target_vs_retain_c1ref.json`
- `outputs/<RUN_TAG>/mia/mia_c6_deletion_attack_target_vs_retain.json`
- optional feature summaries such as seeded canonical-versus-placebo comparison tables

For a real use case, the recommended workflow is:
1. keep the raw JSON outputs for each stage
2. summarize them into small tables like the ones above
3. interpret those summaries in the context of your own corpus, holdout construction, and threat model
