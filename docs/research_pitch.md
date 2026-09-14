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
answering "failure" every time. Publishing a tuned classifier's score without
those two is the ordinary way to be technically truthful and practically
misleading. The modelling contribution on this dataset is close to nil, and what
follows is about the protocol.

**Detection without labelled failures is a solved ranking problem here.** Over
five repeats of grouped five-fold cross-validation, 25 fits per detector, the
four reach ROC-AUC between 0.981 and 0.989. Three of them, One-Class SVM,
Mahalanobis and PCA reconstruction, have overlapping confidence intervals on
every metric and are reported as indistinguishable rather than ranked. Isolation
Forest is distinguishably worse while being 46 times slower and 145 times larger.
On this data the classical control chart is not beaten by anything.

**It is an unsolved calibration problem.** A threshold asking for a 5% false
alarm rate delivers 54% to 66% on healthy executions the detector has not seen.
The mechanism is identifiable: the five subsets are phases of the same task with
different force regimes, so healthy operation is a mixture, and a threshold
calibrated on a sample of that mixture does not hold on an unseen part of it. A
distribution-free tolerance bound, which caps the false alarm rate with stated
confidence under an i.i.d. assumption, halves the damage on Mahalanobis, from
60.4% to 24.7%, and does nothing for One-Class SVM, whose decision function is
nearly discrete. Neither rule fixes the mixture, which is the honest end of the
analysis and the natural start of the next one.

**Under realistic line economics, the miscalibration costs an order of
magnitude.** At one failure per hundred executions with a missed failure worth a
hundred unnecessary stops, expected cost per execution is 1.00 for ignoring every
alarm, 0.99 for stopping on everything, 0.058 at the best reachable operating
point, and 0.28 to 0.68 at the threshold a label-free rule actually sets. Between
80% and 90% of the achievable benefit is lost to the choice of threshold rather
than to the choice of model. This is also why F1 stops being reported at this
point: One-Class SVM scores 0.904 F1 while flagging 54% of healthy runs, an
artefact of a test set that is 72% failures.

**The benchmark itself contains duplicates, and they hid both problems.** 463
rows hold 251 distinct sensor traces. LP2 and LP3 annotate the same 47
recordings; LP4 and LP5 share 116 more. A random 80/20 split hands the model 64
of its 93 test executions in advance. Grouping the folds costs 0.021 to 0.036 of
cross-validated F1, turns a random forest's 1.000 ± 0.000 into 0.979 ± 0.019, and
triples the measured false alarm rate. A shuffled-label control had passed
throughout, because a permutation breaks the association duplicates carry: a leak
control tests one leak.

**Transfer across task phases degrades honestly.** Holding out one phase and
removing every copy of its traces, mean ROC-AUC falls to 0.923 on the hardest
phase against 0.993 when the duplicates are left in.

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
- The confidence intervals assume the 25 folds are exchangeable. They share
  training data across repeats, so they are mildly optimistic.
- Inference timings are a laptop CPU. Only the ratios between detectors transfer
  to an embedded target.
- Nothing is deployed. No inference service, no monitoring, no field trial.

## A 9 to 12 week research project

Ordered by what the current results make most pressing.

1. **Threshold calibration under a mixture of healthy regimes** (weeks 1-4). The
   clearest finding here is that discrimination is saturated and calibration
   fails, and that it fails for an identified reason: healthy operation is
   several behaviours, not one. Four candidates with directly comparable false
   alarm rates: extreme value theory on the healthy score tail, conformal
   prediction under relaxed exchangeability, per-regime calibration after
   clustering the healthy data, and online recalibration. The evaluation harness,
   the baselines and the cost model already exist in the repository, so the first
   comparable number is days away rather than weeks.

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
three baselines beside every score, confidence intervals over 25 fits rather than
a point from one lucky fold, three leaks measured rather than asserted, a control
that failed to catch the second defect and is documented as insufficient, and a
continuous integration job that fails if any published figure moves. That
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
