# Sibling Captures and Session Leakage in ISCXVPN2016 — artefacts

Artefacts for the paper *Sibling Captures and Session Leakage in ISCXVPN2016: How Split Choices Change
Encrypted-Traffic Model Comparisons* (under review).

This repository lets readers check the paper's numbers without the raw captures. It contains:

- the file-to-session manifest of all 140 ISCXVPN2016 capture files;
- the pair-level evidence of the sibling-capture audit;
- the final train/validation/test index arrays of every split;
- the per-flow predictions and run records of every model run the paper uses;
- the scripts that regenerate every table, confidence interval, and number in the text.

It does **not** contain raw packets, payload bytes, flow features, IP addresses, or ports. Five-tuples appear only as
counts; the salt of the HMAC used to join connections during the audit is not released.

## Reproduce the paper's numbers

Requirements: Python 3.12 with the versions in `requirements.txt`. No GPU and no raw data are needed. The run takes
about two minutes on one CPU core.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHON=.venv/bin/python ./reproduce.sh
```

`reproduce.sh` runs the seven scripts in `analysis/` in order and writes their results to `outputs/`. It then compares
each file byte for byte with `expected/`, the files the paper was built from.

| Script | What it checks or produces | Output |
|---|---|---|
| `fixed_split_plan.py` | Recomputes every session-grouped split with the official scikit-learn 1.9.1 `StratifiedGroupKFold` (the corrected splitter) | `fixed_split_plan.json` |
| `check_g03.py`, `check_g04.py` | Consistency checks of the earlier runs made with the defective splitter (the paper reports them only for comparison) | `g03.json`, `g04.json` |
| `split_bug_effect.py` | How often the scikit-learn 1.1.3 splitter leaves a class without training data (Section 4 of the paper) | `split_bug.json` |
| `check_g05.py` | Checks every final grouped run: split arrays equal the official splitter, each flow is tested once, every class has training data in every fold, metrics recompute from predictions, early-stopping prefixes are identical | `g05.json` (1814 checks) |
| `check_leak.py` | The same checks for the flow-random runs | `leak_check.json` (105 checks) |
| `make_numbers.py` | All tables, macros, and bootstrap confidence intervals of the paper | `numbers.tex`, `tab_*.tex`, `ladder.csv`, `facts.json` |

`numbers.tex` holds one LaTeX macro per number in the paper text, and `facts.json` states what each macro means.
`tab_*.tex` holds Tables 1–6; `tab_perclass.tex` is a per-class table that the paper does not print. The bootstrap is a class-stratified paired cluster bootstrap over
E3 groups with 10,000 resamples and a fixed seed, so its intervals are reproduced exactly.

## Layout

```
manifest/   files.csv, sibling_pairs.csv, evidence_sessions.json, E2/E3_file_to_group.csv
runs/G01..G05/
            results/   per-flow predictions (*_predictions.csv.gz) and a run record (*.json) per run
            data/splits, results/*_splits.npz   train/validation/test indices
            audit/     row manifests, per-fold class coverage, near-duplicate exposure per test flow
analysis/   the scripts above (paths relative to the repository root)
expected/   the outputs used to build the paper
records/    the audit and G05 rules (Chinese originals), the G05 protocol, environment and package versions
pipeline/   the server-side scripts that extracted flows, ran the audit and trained the models (for inspection only;
            they need the raw ISCXVPN2016 captures and the local configuration, which are not included)
```

### Manifest

`manifest/files.csv` has one row per capture file. Columns:

- `class`: activity label inherited from the file;
- `in_9class_task`, `flows_9class_task`: whether the file is in the nine-class task, and how many flows it contributes after cleaning;
- `suffix_session`: file stem without one trailing `a`/`b`/`A`/`B` (and a preceding underscore);
- `evidence_session`: suffix sessions merged along *confirmed* pairs;
- `E2_group`, `E3_group`: the broader merges of the sensitivity analysis;
- the remaining columns: per-file sums of the pair evidence.

`manifest/sibling_pairs.csv` has one row per candidate pair (130 pairs). Columns:

- time relation and gap;
- shared TCP and UDP five-tuples (counts);
- split TCP connections;
- identical packets (count and shares);
- identical feature flows;
- near-duplicate matches;
- the verdict.

The verdict `shared_connection_insufficient` is called *shared five-tuple only* in the paper.

### Splits

Each `*_splits.npz` has the arrays `fold{k}_fit`, `fold{k}_validation` and `fold{k}_test` for k = 0–4. The values are
row indices into the flow table of the condition:

- **Grouped splits** (final, corrected splitter): `runs/G05/data/splits/G05_{condition}_{seed}_splits.npz`. The indices
  refer to the rows of `runs/G05/audit/{condition}_row_manifest.csv.gz` (columns `flow_id, file, session, true_label`).
- **Flow-random splits**: `runs/G01/results/v3/B_baseline_leak_multi_RF_seq_iat_{seed}_splits.npz` (identical arrays
  for all models; `check_leak.py` verifies this). The indices refer to the rows of the matching predictions file.

Condition names used in file names:

| Name | Split in the paper |
|---|---|
| `leak` | flow-random |
| `no_session` | file-grouped |
| `Pstar` | suffix-session (reference protocol P\*) |
| `evidence` | evidence-session |
| `E2`, `E3_8`, `Pstar_8` | broader session merges (`_8`: eight-class variant) |
| `no_clean`, `no_sni`, `no_norm` | P\* with one choice changed (no background cleaning, no SNI zeroing, no standardisation) |
| `interaction` | file-grouped and no cleaning |

Seeds are 1, 7 and 42. `runs/G05/audit/fold_class_coverage.csv` lists the fit, validation and test flows and groups of
every class in every fold.

### Predictions and run records

The columns of `*_predictions.csv.gz` are:

- `flow_id`: SHA-256 of the file name and flow index, truncated; it carries no address;
- `session`, `file`;
- `fold`: the test fold of the flow;
- `true_label`, `pred_label`;
- `prob_0` … `prob_8`: class scores.

Class indices follow `runs/G01/config/final_classes.json`. Every flow is predicted exactly once, and the paper pools
the predictions of the five test folds.

The run record beside each prediction file (`*.json`) holds:

- the metrics;
- per-fold details (selected epoch, training curve, fit/validation/test sizes);
- the SHA-256 of the split arrays and of the protocol.

Model identifiers in file names:

| Identifier | Model |
|---|---|
| `RF_seq_iat` | random forest on len+IAT (300 trees); "RF" in the paper |
| `RF_stat`, `LR_seq_iat`, `LR_stat` | shallow baselines |
| `8` | 1D-CNN on len+IAT |
| `11` | Fusion (bytes CNN + len+IAT CNN + MLP on statistics) |
| `7` | 1D-CNN on payload bytes |
| `1`, `20`, `2`, `3` | 1D-CNN on len; on the first 10 lengths; CNN-BiLSTM; Transformer |

Training budgets in file names:

- no suffix: 30 epochs;
- `_100epoch`: 100 epochs;
- `_es`: early stopping (up to 300 epochs, patience 30).

### Which runs are final

The scikit-learn 1.1.3 `StratifiedGroupKFold(shuffle=True)` used on the training server does not keep folds
class-stratified. The defect was fixed upstream in 1.8 (PR #32540). Run G05 repeated every grouped run with the
corrected algorithm ported into the 1.1.3 environment (`pipeline/G05/fixed_splitter.py`), and `check_g05.py` verifies
that its split arrays equal those of the official 1.9.1 splitter.

| Runs | Status |
|---|---|
| G05 | All grouped results in the paper |
| G01 / G02 `leak` | Flow-random results (StratifiedKFold, not affected by the defect) |
| G01, G03, G04 grouped | Superseded; included only for the old-splitter comparison in the paper and for the consistency checks |

### Records

`records/G01_RULES.md` contains the audit decision rules. They were written before any capture was opened.
`records/G01_RULES_addendum.md` contains the rules added after the first stage. Both files are in Chinese.

The heading "预注册" means the rules were fixed in advance and recorded internally. They were not filed with a
third-party registry.

`records/G05_RULES.md` and `records/G05_protocol.json` describe the corrected rerun. They were fixed before training.

## Data source

ISCXVPN2016 was created by the Canadian Institute for Cybersecurity, University of New Brunswick. If you use these
artefacts, please cite:

> G. Draper-Gil, A. H. Lashkari, M. S. I. Mamun, A. A. Ghorbani. Characterization of Encrypted and VPN Traffic Using
> Time-Related Features. ICISSP 2016.

and the paper above (the citation will be added after publication).

## Licence

The scripts and the derived data files in this repository are released under the MIT License (`LICENSE`).
ISCXVPN2016 itself is distributed by the Canadian Institute for Cybersecurity under its own terms.
