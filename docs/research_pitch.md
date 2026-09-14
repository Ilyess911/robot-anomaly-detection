# Research pitch

*For researchers in smart manufacturing, predictive maintenance, industrial AI,
edge AI and cyber-physical systems.*

**Ilyess Assadi** — Engineering student, Industry & Robotics, ESILV Paris.
Avionics maintenance engineering apprentice, Air France Industries.

---

## Motivation

Condition monitoring on a production line rarely starts with a labelled fault
catalogue. It starts with a machine that has run acceptably for months and an
engineer who needs to know when it stops. That asymmetry, abundant healthy data
and almost no failure data, is what makes one-class anomaly detection the
practical entry point to predictive maintenance, and it is the setting this
project studies.

The question:

> **How well can lightweight machine learning models detect abnormal operating
> behaviour in a robotic system from multivariate force and torque data, and how
> much of a reported score survives an honest evaluation protocol?**

The second half turned out to matter more than the first.

## Data and method

The UCI Robot Execution Failures dataset: 463 executions of an industrial
manipulator, each recorded as 15 time steps of 6 wrist force and torque
channels, labelled healthy or failed. Eight statistics per channel reduce each
execution to 48 interpretable descriptors (mean, dispersion, extrema, range,
skewness, kurtosis, linear trend). Raw samples were rejected because 90 values
on 463 executions overfits and resists physical interpretation.

Two families are compared on one shared held-out set:

- **Supervised**, with labelled failures available: logistic regression, random
  forest, RBF SVM and gradient boosting, each grid-searched under
  cross-validation with the scaler inside the pipeline.
- **One-class**, fitted on healthy executions only, which is the deployable
  setting: Isolation Forest, One-Class SVM, PCA reconstruction error (the linear
  autoencoder) and Mahalanobis distance under Ledoit-Wolf shrinkage.

Every score is published beside three baselines: the constant answer, the best
single sensor statistic at a single threshold, and a depth-2 decision tree.

## Results

**The problem is easier than the models suggest.** One threshold on the range of
the Fx force channel reaches 0.931 F1 on the failure class, against 0.838 for
answering "failure" every time. A tuned random forest reaches 1.000 on the
held-out fold. Publishing the last number without the first two is the ordinary
way to be technically truthful and practically misleading.

**Detection without labelled failures is viable here.** All four one-class
detectors reach ROC-AUC 1.000 on the grouped held-out fold: fitted on healthy
runs alone, they rank every failure above every healthy execution.

**Ranking is not the problem, the threshold is.** At a label-free threshold set
at the 95th percentile of the healthy training scores, the same four detectors
score between 0.660 and 0.918 F1. Isolation Forest falls from 1.000 at the 90th
percentile to 0.660 at the 95th; One-Class SVM is flat at 0.918 across the whole
sweep; PCA and Mahalanobis improve monotonically to 0.971 at the 99th. An oracle
threshold would give all four 1.000. **The entire operational gap is calibration,
not discrimination**, which is a result about deployment rather than about
models.

**The benchmark contains hidden duplicates, and they produced the perfect
scores.** 463 rows hold 251 distinct sensor traces. LP2 and LP3 annotate the same
47 recordings under two fault taxonomies; LP4 and LP5 share 116 more. A random
80/20 split hands the model 64 of its 93 test executions in advance. Under
grouped cross-validation, where no trace sits on both sides, cross-validated F1
falls by 0.021 to 0.036 depending on the model, and the random forest's
1.000 ± 0.000 becomes 0.979 ± 0.019. A standard deviation of exactly zero across
five folds of 463 samples was the signature, and a shuffled-label control had
passed throughout, because a permutation breaks the association duplicates
carry.

**Transfer across task phases degrades honestly.** Holding out one phase of the
assembly task and removing every copy of its traces, mean ROC-AUC over the four
detectors falls to 0.923 on the hardest phase and 0.952 on the next, against
0.993 and 0.990 when the duplicates are left in. The naive protocol reports
near-perfect transfer. The trace-disjoint one does not.

## Limitations

Stated because they bound what the results support.

- 463 executions, one manipulator, one task, one recording session. Nothing here
  demonstrates transfer to another arm or another cell.
- The class balance is inverted relative to a real line: 72% failures here, the
  opposite in production. Precision on the rare class, the metric that would
  degrade first, is therefore not measured under realistic conditions.
- No temporal validation. Executions are shuffled, not ordered, so sensor drift,
  recalibration and wear are absent from the evaluation.
- The 15 time steps are collapsed into statistics. No sequential model is used
  and no claim about sequence modelling is made.
- Nothing is deployed. No inference latency, no memory footprint, no monitoring.

## A 9 to 12 week research project

Ordered by what the current results make most pressing.

1. **Threshold calibration without labels** (weeks 1-4). The clearest finding
   here is that discrimination is solved and calibration is not. Extreme value
   theory on the healthy score tail, conformal prediction with a
   distribution-free false alarm guarantee, and drift-adaptive thresholds are
   three approaches whose false alarm rates are directly comparable.

2. **Sequential models on the raw traces** (weeks 3-6). LSTM or temporal
   convolutional autoencoders on the 15 by 6 tensor, tested against the 48
   statistics rather than assumed better. With 129 healthy executions, the honest
   question is whether such a model can be fitted at all at this sample size, and
   the answer belongs in the record either way.

3. **Edge deployment and its cost** (weeks 5-8). Inference latency, memory and
   energy for each detector on a microcontroller or a single-board computer.
   Mahalanobis distance is one matrix multiply; Isolation Forest is 200 trees;
   on this data their accuracy is indistinguishable. That is exactly the kind of
   trade-off an edge AI study should quantify.

4. **A dataset where the question is hard** (weeks 6-10). Extension to a
   benchmark with realistic class imbalance and genuine temporal ordering: CWRU
   bearing data, NASA C-MAPSS, or a manipulator log with chronology. The protocol
   built here, grouped splitting, published baselines, threshold sensitivity,
   transfers directly. The conclusions do not.

5. **From detection to diagnosis and remaining useful life** (weeks 9-12). The 16
   original fault classes are collapsed to binary here because the rare ones have
   three and five members. On a larger dataset, multi-class fault diagnosis and
   RUL estimation are the natural continuations, and the industrial value sits
   there rather than in a binary alarm.

## Why this work, from this candidate

The contribution is not a model. It is a protocol that refuses to flatter itself:
three baselines beside every score, leaks measured rather than asserted, a
control that failed to catch the second defect and is documented as insufficient,
and a continuous integration job that fails if any published figure moves. That
habit transfers to any applied ML project and it is the part of research work
that a course grade does not capture.

The industrial framing is not borrowed. I am an avionics maintenance engineering
apprentice at Air France Industries, where condition monitoring and scheduled
maintenance are the daily subject, and I read these results under that
constraint: a detector that is right on average but alarms on 15% of healthy runs
gets switched off within a week.

---

**Repository** — https://github.com/Ilyess911/robot-anomaly-detection

Every figure quoted above regenerates with `make reproduce`, and continuous
integration fails if any of them moves.
