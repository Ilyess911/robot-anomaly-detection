<div align="center">

# Robot Anomaly Detection

**A labelling bug made an entire subset look 100% faulty. Finding it was the work.**

Detecting collisions, obstructions and tool faults across 463 industrial robot
executions, from six force and torque sensors sampled fifteen times per run.

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![Lint](https://img.shields.io/badge/lint-ruff-261230?logo=ruff&logoColor=white)
[![verify](https://github.com/Ilyess911/robot-anomaly-detection/actions/workflows/verify.yml/badge.svg)](https://github.com/Ilyess911/robot-anomaly-detection/actions/workflows/verify.yml)
![Licence](https://img.shields.io/badge/licence-MIT-2EA043)

<br />

![Average sensor signal, healthy runs against failed runs](docs/normal-vs-anomaly-pattern.png)

</div>

---

## What this is

A machine learning course project at ESILV, built with Adel Bousri, on the UCI
Robot Execution Failures dataset. Six sensors, fifteen time steps, one label per
run: did the arm complete its motion or hit something.

It would be a forgettable classification exercise, except that auditing it a
year later turned up a data bug that had quietly distorted every published
number. What follows is that audit.

## The promise (and its limits)

| It does | It does not |
| --- | --- |
| **Publish a baseline next to every score**, because 0.96 means nothing on its own | Claim the models are good: most of the performance is in the data |
| **Prove a perfect score is real**, with a shuffled-label control | Run anywhere near a production line |
| **Reproduce every published figure by one command** | Validate over time: executions are shuffled, not ordered |

## The bug

LP3 names its healthy class `ok`. Every other subset names it `normal`. The
binary encoder matched on `normal` alone:

```python
lambda x: 0 if str(x).lower() == 'normal' else 1
```

So the 20 healthy executions of LP3 were counted as failures, and one subset out
of five appeared to fail 100% of the time.

| | Before | After |
| --- | --- | --- |
| Healthy executions | 109 | **129** |
| Anomaly share | 76.5% | **72.0%** |
| LP3 healthy runs | 0 / 47 | **20 / 47** |

Every score in the original notebooks was computed against those wrong labels,
and 20 contradictory examples were being forced into every model. Fixing them
did not degrade the results. It improved them, which is how the bug had stayed
invisible.

## Results

Statistical features per sensor (mean, std, min, max, range, skewness, kurtosis,
linear trend: 48 in total), an 80/20 stratified split, five-fold grid search on
the training half. Test set: 93 executions, 67 of them failures.

The first three rows are not models.

| | F1 anomaly | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- |
| Answer "anomaly" every time | 0.838 | 0.720 | 0.720 | 1.000 |
| **One sensor, one threshold** (`Fx_range`) | **0.924** | 0.893 | 0.939 | 0.910 |
| A depth-2 tree, so three decisions | 0.971 | 0.957 | 0.957 | 0.985 |

| Tuned models | F1 anomaly | Accuracy | ROC AUC | CV F1 |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.977 | 0.968 | 0.995 | 0.979 ± 0.012 |
| SVM (RBF) | 0.977 | 0.968 | 0.986 | 0.977 ± 0.013 |
| Gradient Boosting | 0.993 | 0.989 | 0.992 | 0.994 ± 0.008 |
| **Random Forest** | **1.000** | 1.000 | 1.000 | 1.000 ± 0.000 |

| Unsupervised, trained on 103 healthy runs only | F1 anomaly | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- |
| Isolation Forest | 0.957 | 0.936 | 0.930 | 0.985 |
| One-Class SVM | 0.944 | 0.914 | 0.893 | 1.000 |

Read the two tables together and the conclusion is uncomfortable: **one
threshold on one sensor gets 92% of the way there.** A grid-searched Random
Forest buys the remaining eight points. The difficulty in this problem was never
the model.

`make benchmark` reproduces every figure above. CI regenerates them on each push
and fails if any moves.

## A perfect score is an alarm

Random Forest reaches 1.000 on the test set and 1.000 ± 0.000 across five folds.
On 463 samples, that is a reason to look for a leak, not to celebrate.

The control is to destroy the signal and check the model follows:

| Labels | CV F1 |
| --- | --- |
| Real | 0.993 |
| Shuffled, five draws | **0.760** |

Trained on noise the model lands *below* the constant baseline of 0.838, which
is what a clean pipeline does: it fits patterns that are not there and pays for
it. Had it stayed high, something in the features would have been carrying the
answer. This control is now a test, so it runs on every commit.

## What the scaler leak was worth

The notebooks standardise before they split, so the scaler sees the test set:

```python
X_scaled = scaler.fit_transform(X)                    # the whole dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, ...)
```

Textbook leak, and the first thing to look for in anyone's notebook. The
benchmark runs both protocols in the same environment, same seeds, same grids:

| | Leaky | Clean | Cost |
| --- | --- | --- | --- |
| Logistic Regression | 1.000 | 0.977 | **-0.023** |
| Random Forest | 1.000 | 1.000 | 0.000 |
| SVM (RBF) | 0.977 | 0.977 | 0.000 |
| Gradient Boosting | 0.993 | 0.993 | 0.000 |

Zero on the tree models, which are invariant to scaling and could never have
been affected. On logistic regression the leak buys a fake perfect score.

**An earlier version of this README got that number wrong.** It reported -0.011
by comparing notebook outputs from Python 3.9 against a script run on Python
3.14 with a five-year-newer scikit-learn. Two variables, one conclusion. The
measurement is only meaningful inside one environment, which is why
`requirements-lock.txt` pins it.

## Decisions

**Statistical features instead of raw samples.** 90 raw values on 463 runs
invites overfitting and resists interpretation. Eight statistics per sensor cut
the space to 48 dimensions and produce names a maintenance engineer can argue
with: `Tz_std` is the variability of yaw torque, not "feature 74".

**All five subsets merged.** LP1 to LP5 are phases of one assembly task, holding
between 47 and 164 runs. Kept apart, three of the five are too small to split
meaningfully. The cost is stated in the limits.

**Binary target.** The 16 original labels include classes with 3 and 5 members.
A 16-class model on 463 samples would produce confident nonsense on the rare
ones, and the useful question on a line is binary anyway: stop the arm or not.

**Unsupervised models see only healthy runs.** That mirrors deployment order. A
new cell has months of healthy operation and no catalogue of failures it has not
had yet.

## Limits

**The test set is 93 executions.** One reclassified sample moves accuracy by a
full point. The cross-validated figures are more trustworthy than the single
split, and both are reported.

**The class balance is inverted relative to reality.** 72% failures here; a real
line sees the opposite, and precision on the rare class is exactly what would
degrade. This project does not measure the thing that would matter most.

**There is no temporal validation.** Executions are shuffled at random. A real
deployment trains on the past and tests on the future, including sensor drift
and recalibration, none of which this dataset exposes.

**One robot, one task, one recording session.** Nothing here demonstrates
transfer to another arm or another cell.

**Nothing is deployed.** No inference service, no latency measurement, no
monitoring. The models are pickles on disk.

**The notebooks still show the old numbers.** They ran before the labelling fix
and have not been re-executed, on purpose: rewriting them would erase the record
of what was wrong. Where they disagree with this page, this page is right.

## What I take from it

That the labels deserve more suspicion than the model. Four classifiers within
two points of each other told me nothing; twenty rows named `ok` instead of
`normal` changed every table on this page.

That a baseline is not a formality. The distance between 0.838 and 1.000 is the
whole contribution, and publishing the second number without the first is the
most common way to be technically truthful and practically misleading.

That a perfect score obliges you to prove it. The shuffled-label control took
ten minutes to write and is the only reason the 1.000 in the table above is
allowed to stay there.

## Running it

```bash
git clone https://github.com/Ilyess911/robot-anomaly-detection.git
cd robot-anomaly-detection
make setup       # venv + the exact versions behind the published numbers
make verify      # lint, tests, benchmark
make notebooks   # opens notebooks/, run 01 to 05 in order
```

## Structure

```
notebooks/     01 exploration, 02 preprocessing, 03 supervised,
               04 unsupervised, 05 evaluation
src/           utils.py    parsing, labels, features, plots, metrics
               models.py   trainers with grid search (superseded by the benchmark)
scripts/       benchmark.py        the comparison behind every published figure
               scrub_notebooks.py  strips machine paths and warning noise
tests/         test_dataset.py     the dataset is what the README claims
               test_protocol.py    the protocol, the baseline, the leak control
data/          lp1 to lp5, the UCI subsets, unmodified
models/        trained estimators, one pickle per model
reports/       benchmark.json, regenerated by CI and compared to this file
docs/          figures used by this page
```

## Credits

Built with Adel Bousri as a machine learning course project at ESILV. The
pipeline, the modules and the notebooks are joint work. The audit, the
benchmark, the tests and this page are later additions by Ilyess Assadi.

The dataset belongs to its authors and is redistributed here unmodified for
reproducibility. Please cite the
[UCI entry](https://archive.ics.uci.edu/dataset/138/robot+execution+failures),
donated by Luis Seabra Lopes and Luis M. Camarinha-Matos, rather than this
repository.

## License

MIT for the code. See [LICENSE](LICENSE); the dataset is excluded and keeps its
own terms.
