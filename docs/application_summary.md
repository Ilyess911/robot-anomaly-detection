# Project summary for applications

Three lengths of the same project, plus the discussion a supervisor is likely to
open with. Every figure quoted here comes from `reports/` and regenerates with
`make reproduce`.

---

## Three lines, for a CV or a signature block

> **Robot Anomaly Detection**. An anomaly detection framework for robotic and
> industrial sensor data in which one-class detectors fitted on healthy operation
> alone rank failures at ROC-AUC 0.99 yet flag 60% of healthy runs at a threshold
> asking for 5%, destroying 90% of the achievable value under realistic line
> economics. Also found that the benchmark holds 463 rows over 251 distinct
> recordings, which is what produced the perfect scores previously published on
> it. Fully reproducible: continuous integration fails if any figure moves.

---

## One hundred words, for a motivation letter

> I built an anomaly detection framework for robotic and industrial sensor data
> on the UCI Robot Execution Failures dataset: 463 executions, six force and
> torque channels, fifteen time steps each. Four lightweight one-class detectors
> are fitted on healthy runs only, which is the deployable setting in predictive
> maintenance, and evaluated over 25 grouped folds. They rank failures almost
> perfectly and calibrate badly: a threshold asking for a 5% false alarm rate
> delivers 54% to 66%, which under realistic line economics destroys about 90% of
> the value the detector could deliver. Auditing the benchmark also showed 463
> rows over 251 distinct recordings, which is what produced the perfect scores
> previously published on it.

---

## Two hundred words, for a research internship application

> This project studies how well lightweight machine learning models detect
> abnormal operating behaviour in a robotic system from multivariate force and
> torque data, and what survives once the alarm threshold has to be chosen
> without labels.
>
> Each execution of an industrial manipulator is recorded as fifteen time steps
> of six wrist channels and reduced to 48 interpretable descriptors. Four
> one-class detectors, Isolation Forest, One-Class SVM, PCA reconstruction error
> and Mahalanobis distance, are fitted on healthy executions alone and evaluated
> over five repeats of grouped five-fold cross-validation.
>
> Three results carry the work. One threshold on one sensor reaches 0.931 F1
> against 0.838 for answering "failure" every time, so the difficulty was never
> the model. The detectors then rank failures at ROC-AUC 0.981 to 0.989 while a
> threshold asking for a 5% false alarm rate delivers 54% to 66%, because healthy
> operation is a mixture of task regimes rather than one distribution; under
> realistic line economics that miscalibration costs an order of magnitude more
> than the modelling gains are worth. And the benchmark itself holds 463 rows
> over 251 distinct recordings, so the random split used in published work leaks
> 69% of the test set, inflates cross-validated F1 by up to 0.036, and triples
> the apparent false alarm rate.
>
> Everything reproduces from one command, and continuous integration fails if a
> published number moves.

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

1. **Can a threshold be set without labels when healthy is not one behaviour?**
   The distribution-free tolerance bound implemented here caps the false alarm
   rate under an i.i.d. assumption, halves the damage on one detector and does
   nothing for another. The assumption is what breaks, not the mathematics.
   Per-regime calibration, conformal prediction under relaxed exchangeability and
   online recalibration are the candidates, and the harness to compare them
   already exists.

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
