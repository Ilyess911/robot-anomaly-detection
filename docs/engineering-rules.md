# Engineering rules for this repository

Read `README.md` first. It is the product: the numbers it publishes are the
contract, and `reports/benchmark.json` and `reports/experiments.json` are their
proof.

## Commands

```
make setup       # venv with the exact versions that produced the published numbers
make verify      # lint, tests, both benchmarks. What CI runs
make reproduce   # benchmark, experiments, figures. Every published artefact
make benchmark   # baselines and supervised models, both split protocols
make experiments # repeated grouped CV, calibration study, transfer
make deployment  # cost-sensitive operating point and inference cost
make figures     # redraws assets/ from reports/
make detect      # inference CLI, see scripts/detect.py --help
make demo        # Streamlit demo, needs requirements-demo.txt
make notebooks   # opens notebooks/, to run in order 01 to 05
```

## Rules that are not up for discussion

1. **No score is published without a baseline next to it.** On this dataset,
   answering "failure" every time already scores 0.838 F1, and one threshold on
   one sensor scores 0.931. A model is judged against those, never against zero.
   A results table that omits them is misleading even when every figure in it is
   correct.

2. **A perfect score is an alarm, and one control is never enough.** Random
   Forest reached 1.000 ± 0.000 here for a year while 46% of the dataset was
   duplicated executions, and the shuffled-label control passed throughout. It
   was not broken; it answers a different question. Feature leakage, scaler
   leakage and sample leakage are three failures and each needs its own check.

3. **The grouped split is the headline protocol.** 463 rows hold 251 distinct
   sensor traces: LP2 and LP3 annotate the same 47 recordings, LP4 and LP5 share
   116 more. Any evaluation that splits at random must be labelled as such and
   published beside the grouped one, never instead of it.

4. **One definition of F1, named.** `f1_anomaly` is the positive class,
   `f1_weighted` is the weighted average. The original notebooks used three
   different definitions across three files and compared them to each other.
   Never report a metric whose averaging is implicit.

5. **The scaler lives inside the Pipeline.** Fitting it on the full dataset
   before splitting leaks the test set. It happens to cost nothing measurable on
   the tree models, which is not a reason to keep doing it.

6. **A metric is reported with an interval or not at all.** Four detectors at
   ROC-AUC 1.000 on one fold of 93 executions was a statement about the fold, not
   about the detectors. Headline figures come from five repeats of grouped
   five-fold, 25 fits, with a confidence interval, and two detectors whose
   intervals overlap are called indistinguishable rather than ranked.

7. **A score is not a decision.** Any detector published here is also reported at
   the threshold a label-free rule actually sets, and the realised false alarm
   rate is measured on held-out healthy runs. Reporting the F1 at the threshold
   that maximises F1 requires the labels and means nothing for deployment; it
   appears only as the ceiling.

8. **Costs are parameters, never folded into a score.** Discrimination is
   measured on held-out data, line economics are imposed from outside and stated.
   The two are never mixed into one number without saying so.

9. **Every published number and figure is reproducible by one command.** If the
   README says 0.931, `make reproduce` must produce 0.931 and redraw the figure
   that shows it. CI replays both reports, compares them to the committed ones,
   and fails on any drift.

10. **No artefact is committed that a command cannot regenerate.** Four pickles
   and three figures once outlived a labelling fix by two months because nothing
   rebuilds a committed binary. `models/*.pkl` is gitignored and every image in
   `assets/` comes from `scripts/figures.py`.

11. **Label semantics belong in `HEALTHY_LABELS`, not in a lambda.** LP3 calls its
   healthy class `ok`; every other subset calls it `normal`. Matching on `normal`
   alone silently turned 20 healthy executions into failures. Any new label
   mapping goes through that constant, with the evidence in a comment.

12. **Every detector obeys the sign convention.** `anomaly_score` is higher for
   more anomalous input, always. scikit-learn is inconsistent about this, an
   inverted detector still produces plausible output, and `tests/test_detectors.py`
   checks it rather than trusting it.

13. **Notebooks keep their original outputs.** They ran before both corrections
    and their figures reflect that. Re-running them would erase the record of
    what was fixed. Each carries a banner saying so, and the README says which of
    their numbers are stale.

## Style

Python 3.14, ruff for lint and format, line length 100. This repository is
written in English throughout, because it is public and its audience is
international. Docstrings explain why a choice was made, not what the next line
does.

Prose avoids em dashes. Commits follow Conventional Commits, with the body in
French explaining the reasoning.

## Definition of done

`make verify` green, `reports/*.json` regenerated and committed if any number
moved, `assets/` redrawn in the same commit as the number it illustrates, the
README updated alongside, and the notebooks left alone.
