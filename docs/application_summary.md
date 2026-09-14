# Project summary for applications

Three lengths of the same project, plus the discussion a supervisor is likely to
open with. Every figure quoted here comes from `reports/` and regenerates with
`make reproduce`.

---

## Three lines, for a CV or a signature block

> **Robot Anomaly Detection** — An anomaly detection framework for robotic and
> industrial sensor data, comparing supervised classifiers against one-class
> detectors fitted on healthy operation alone. Found and quantified two data
> defects in the benchmark itself, including duplicate executions that inflated
> every published score on it. Fully reproducible: continuous integration fails
> if any published figure moves.

---

## One hundred words, for a motivation letter

> I built an anomaly detection framework for robotic and industrial sensor data
> on the UCI Robot Execution Failures dataset: 463 executions, six force and
> torque channels, fifteen time steps each. It compares four supervised
> classifiers against four lightweight one-class detectors fitted on healthy
> runs only, which is the deployable setting in predictive maintenance. Auditing
> it revealed that the benchmark holds only 251 distinct recordings behind its
> 463 rows, so a random split hands the model 69% of its test set in advance.
> Under a grouped protocol the perfect scores disappear. The repository publishes
> baselines, leak controls and a reproducibility check for every figure.

---

## Two hundred words, for a research internship application

> This project studies how well lightweight machine learning models detect
> abnormal operating behaviour in a robotic system from multivariate force and
> torque data, and how much of a reported score survives an honest evaluation
> protocol.
>
> Each execution of an industrial manipulator is recorded as fifteen time steps
> of six wrist channels and reduced to 48 interpretable statistical descriptors.
> Four supervised classifiers are compared against four one-class detectors
> (Isolation Forest, One-Class SVM, PCA reconstruction error, Mahalanobis
> distance) fitted on healthy executions alone, which mirrors deployment: a new
> cell has months of healthy operation and no catalogue of the failures it has
> not had yet.
>
> Three results carry the work. One threshold on one sensor reaches 0.93 F1
> against 0.84 for answering "failure" every time, so the difficulty was never
> the model. All four one-class detectors rank failures perfectly on unseen data,
> yet score between 0.66 and 0.92 F1 at a label-free threshold: the operational
> gap is calibration, not discrimination. And the benchmark itself holds 463 rows
> over 251 distinct recordings, so the random split every published result uses
> leaks 69% of the test set, which is what produced a cross-validated F1 of
> 1.000 ± 0.000.
>
> Everything reproduces from one command, and continuous integration fails if a
> published number moves.

---

## Research discussion

### What was technically hard

**Proving a negative about a dataset.** The duplicates were not visible in any
summary statistic. The subsets have different sizes, different label vocabularies
and different class balances; nothing suggested that LP2 and LP3 are the same 47
recordings until every row's 90 raw readings were compared against every other.
The hard part was not the comparison, it was deciding to run it on a dataset that
had been used in published work for two decades.

**Keeping two protocols honest at once.** Measuring what a leak is worth means
running both versions in the same environment with the same seeds and the same
search grids. An earlier version of this repository got the standardisation leak
wrong by 0.012 because it compared notebook outputs from Python 3.9 against a
script run on Python 3.14: two variables, one conclusion. Every leak measurement
here is now bracketed by a pinned environment.

**Choosing a threshold rule that is not cheating.** Reporting the F1 at the
threshold that maximises F1 is an oracle result and it is meaningless for
deployment. The percentile rule, calibrated on the healthy training scores, is
what a real system can do, and the gap between the two is reported as the cost
of not having labels.

### What I learned

That a leak control tests one leak. The shuffled-label control in this repository
passed throughout, and it was right to: permuting labels breaks the association
duplicates carry, so a duplicate-driven leak collapses exactly like a clean
pipeline. Feature leakage, scaler leakage and sample leakage need three separate
checks, and I now write all three by default.

That a perfect score with zero variance is a stronger signal than a perfect score
alone. Random Forest reported 1.000 ± 0.000 across five folds of 463 samples for
a year. The mean invited a shrug; the standard deviation should not have.

That baselines change conclusions rather than decorate them. Between "Random
Forest reaches 1.000" and "one threshold on one sensor reaches 0.93, and Random
Forest buys the last seven points" lies the entire difference between a result
and a claim.

### What I would improve

The evaluation still rests on one held-out fold of 93 executions, where a single
reclassified sample moves accuracy by a point. Repeated grouped cross-validation
with confidence intervals would be more defensible than the single split, and it
is cheap on a dataset this size.

The class balance is inverted relative to reality: 72% failures here, the
opposite on a line. The metric that would degrade first in production, precision
on the rare class, is the one this dataset cannot measure. A rebalanced
resampling study would at least bound it.

The 15 time steps are collapsed into statistics before any model sees them. That
choice is defended in the README on interpretability and sample-size grounds, but
it is a choice, and it has never been tested against a sequential model on the
raw tensor.

### What questions remain open

1. **Can a threshold be set without labels and still hold?** Extreme value theory
   on the healthy score tail and conformal prediction both promise a controlled
   false alarm rate. On this data the ranking is already perfect, so a
   calibration study would isolate exactly the quantity that matters.

2. **Does anything here transfer?** Holding out one phase of the task and
   removing every copy of its traces, mean ROC-AUC falls to 0.92 on the hardest
   phase against a uniform 1.00 under the naive protocol. That is a hint, on one
   robot and one task, not a transfer result.

3. **What does each detector cost at the edge?** Mahalanobis distance is a single
   matrix multiply, Isolation Forest is 200 trees, and on this data their
   accuracy is indistinguishable. Latency, memory and energy would separate them
   where the metrics do not.

4. **Is the sequential structure worth anything at this sample size?** With 129
   healthy executions, a nonlinear autoencoder would carry more parameters than
   it has samples. PCA reconstruction error is included as the linear case. The
   honest question is at what sample size the nonlinear version starts to earn
   its cost, and that is answerable.
