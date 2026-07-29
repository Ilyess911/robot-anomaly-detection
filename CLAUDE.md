# CLAUDE.md — working rules for this repository

Read `README.md` first. It is the product: the numbers it publishes are the
contract, and `reports/benchmark.json` is their proof.

## Commands

```
make setup       # venv + the exact versions that produced the published numbers
make verify      # lint, tests, benchmark. What CI runs
make benchmark   # replays the comparison and rewrites reports/benchmark.json
make notebooks   # opens notebooks/, to run in order 01 to 05
```

## Rules that are not up for discussion

1. **No score is published without a baseline next to it.** On this dataset,
   answering "anomaly" every time already scores 0.838 F1, and one threshold on
   one sensor scores 0.924. A model is judged against those, never against
   zero. A results table that omits them is misleading even when every figure
   in it is correct.

2. **A perfect score is an alarm, not an achievement.** Random Forest reaches
   1.0000 here. That claim is only allowed to stand because the shuffled-label
   test proves the pipeline is clean (0.760 on permuted labels, below the
   constant baseline). Any new perfect result requires the same proof.

3. **One definition of F1, named.** `f1_anomaly` is the positive class,
   `f1_weighted` is the weighted average. The original notebooks used three
   different definitions across three files and compared them to each other.
   Never report a metric whose averaging is implicit.

4. **The scaler lives inside the Pipeline.** Fitting it on the full dataset
   before splitting leaks the test set. It happens to cost nothing measurable
   here, which is not a reason to keep doing it.

5. **Every published number is reproducible by one command.** If the README
   says 0.924, `make benchmark` must print 0.924. CI compares the regenerated
   report against the committed one and fails on any drift.

6. **Label semantics belong in `HEALTHY_LABELS`, not in a lambda.** LP3 calls
   its healthy class `ok`; every other subset calls it `normal`. Matching on
   `normal` alone silently turned 20 healthy executions into failures. Any new
   label mapping goes through that constant, with the evidence in a comment.

7. **Notebooks keep their original outputs.** They were run before these
   corrections and their figures reflect that. Re-running them would erase the
   record of what was fixed. The README says which of their numbers are stale.

## Style

Python 3.12, ruff for lint and format, line length 100. Docstrings explain why
a choice was made, not what the next line does.

Commits: Conventional Commits, body in French explaining the reasoning.

## Definition of done

`make verify` green, `reports/benchmark.json` regenerated and committed if any
number moved, the README updated in the same commit as the number it quotes,
and the notebooks left alone.
