# Data quality audit

Two defects were found in this project a year after it was submitted as a
course assignment. Both stayed invisible because both made the results look
*better*, and nothing in the original pipeline was built to be suspicious of a
good number.

This page is the record. The README quotes its conclusions; the evidence lives
here.

---

## Finding 1: one subset names its healthy class differently

**What was wrong.** The binary encoder read:

```python
lambda x: 0 if str(x).lower() == 'normal' else 1
```

LP3 names its healthy class `ok` and contains no row labelled `normal` at all.
LP1, LP2, LP4 and LP5 all use `normal`. The 20 healthy executions of LP3 were
therefore encoded as failures, and that subset appeared to fail 100% of the
time.

**What it moved.**

| | Before | After |
| --- | --- | --- |
| Healthy executions | 109 | **129** |
| Failure share | 76.5% | **72.0%** |
| LP3 healthy runs | 0 of 47 | **20 of 47** |

Every score in the original notebooks was computed against those labels, and 20
contradictory examples were forced into every model. Fixing them did not degrade
the results, it improved them, which is how the bug survived.

**Where the fix lives.** `HEALTHY_LABELS` in `src/data/loader.py`, with the
evidence in a comment beside it, and `tests/test_dataset.py` asserting the count
of 129.

---

## Finding 2: 46% of the dataset is the same executions counted twice

**What was wrong.** The project merged all five subsets into 463 rows and split
them at random. The five subsets are not five recording sessions.

```
463 executions
251 distinct sensor traces
 89 traces appear once
116 traces appear twice
 46 traces appear three times or more
```

LP2 and LP3 hold **the same 47 recordings**, annotated under two different fault
taxonomies: LP2 names what the transfer failure was (`front_col`, `left_col`),
LP3 names where the part ended up (`moved`, `lost`). LP4 and LP5 share 116 more
executions the same way. Established by comparing the 90 raw readings of every
row against every other, not by trusting subset names.

**What it moved.** Under the random 80/20 split this project had always used,
**64 of the 93 test executions, 69%, have an exact copy in the training half.**

On a single held-out fold of 93 executions the effect is small, because the
problem is easy: one threshold on one sensor already reaches 0.93 F1. On the
*cross-validated* figure, which is the trustworthy one at this sample size, the
effect is consistent, and it is where the perfect score came from:

| Cross-validated F1, failure class | Random folds | Grouped folds | Cost |
| --- | --- | --- | --- |
| Logistic Regression | 0.979 ± 0.012 | 0.947 ± 0.028 | **-0.032** |
| Random Forest | **1.000 ± 0.000** | 0.979 ± 0.019 | **-0.021** |
| SVM (RBF) | 0.977 ± 0.013 | 0.954 ± 0.032 | **-0.023** |
| Gradient Boosting | 0.994 ± 0.008 | 0.959 ± 0.012 | **-0.036** |

The standard deviation of exactly zero is the tell. A model does not reach
identical perfection on five different folds of 463 samples unless the folds are
not independent.

**What the duplicates also hid, and this is worse than the F1.** Every one-class
detector in this project is thresholded by a rule asking for a 5% false alarm
rate. Measured on healthy executions the detector has not seen:

| Detector | Target | Random folds | Grouped folds |
| --- | --- | --- | --- |
| Isolation Forest | 5% | 6.5% | 6.8% |
| One-Class SVM | 5% | 15.7% | **54.0%** |
| PCA reconstruction | 5% | 23.2% | **65.7%** |
| Mahalanobis | 5% | 16.4% | **60.4%** |

Grouping triples the measured false alarm rate. Two copies of the same recording
on both sides of a split make the held-out healthy runs look like the training
healthy runs, which is exactly the assumption every threshold rule depends on.
The duplicates were not only inflating F1 by two to four points, they were
concealing a threefold calibration failure, and the calibration failure is the
one that decides whether the system is deployable.

**One thing the duplicates did not break.** No trace ever carries a healthy
label in one subset and a failure label in another. The taxonomies differ, the
healthy against failed call does not, so the binary target stays unambiguous and
merging the subsets remains defensible. Asserted in `tests/test_duplicates.py`.

**Where the fix lives.** `trace_ids()` in `src/data/loader.py` gives every
distinct recording an identifier; `src/evaluation/protocol.py` defines the
grouped split; every published headline figure uses it, and the random-split
figure is published beside it so the gap stays visible.

---

## Why the existing control did not catch it

This repository already ran a shuffled-label control: refit on permuted labels,
check that the score falls below the constant baseline. It passed, at 0.760
against a baseline of 0.838, and it kept passing for a year while 46% of the
dataset was duplicated executions.

That is not a broken control. It answers a different question. Permuting the
labels destroys the association a duplicate carries, so a duplicate-driven leak
collapses under permutation exactly like a clean pipeline does. The control
proves the features do not contain the target. It says nothing about whether the
rows are independent.

The lesson worth carrying to any dataset: **a leak control tests one leak.**
Feature leakage, scaler leakage and sample leakage are three different failures
and each needs its own check.

---

## Finding 3, minor: artefacts outlived the fix that invalidated them

Four trained estimators in `models/` and three figures in `docs/` were committed
at 12:21 and 12:29 on 29 July 2026. The labelling fix landed at 14:37 the same
day. Nothing regenerates a committed binary, so for two months the repository
shipped pickles trained on wrong labels and a hero image drawn from them, beside
a corrected results table.

Fixed by deleting both, gitignoring `models/*.pkl`, and adding
`scripts/figures.py`, which redraws every published image from `reports/` in one
command. CI runs it on every push.

---

## What the audit did not check

Stated so that its silence is not read as an endorsement.

- Whether the 15 time steps are uniformly sampled.
- Whether the shipped `.data` files match the original UCI archive byte for
  byte, beyond parsing cleanly into the documented shapes.
- Whether the duplicated traces were intended by the dataset authors as separate
  annotation tasks. The UCI description of the subsets suggests it; nothing in
  the files states it, and the consequence for a random split is the same either
  way.
