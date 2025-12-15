Anomaly Detection in Industrial Robot Executions

Project overview

This project focuses on detecting anomalies in industrial robot executions using machine learning techniques.
The goal is to identify abnormal robot behaviors (collisions, obstructions, tool issues, etc.) based on force and torque sensor measurements.

The project is based on the UCI Robot Execution Failures dataset and follows a complete machine learning pipeline, from data exploration to supervised and unsupervised anomaly detection.

This work was carried out in the context of the Machine Learning course at ESILV (A4) and is strongly connected to Industry 4.0 applications such as predictive maintenance and operational safety.


Authors
	•	Ilyess Assadi
	•	Adel Bousri

Context and motivation

Industrial robots operate under strict mechanical constraints.
Unexpected failures can lead to:
	•	production downtime,
	•	increased maintenance costs,
	•	safety risks for operators,
	•	quality degradation.

Force and torque sensors react instantly to abnormal physical interactions.
Using machine learning to analyze these signals makes it possible to detect anomalies early, before failures become critical.


Problem definition
	•	Type: Anomaly detection (binary classification)
	•	Input: Force and torque sensor data
	•	Output: Normal execution or anomalous execution

Each robot execution is represented by:
	•	6 sensors: Fx, Fy, Fz (forces), Tx, Ty, Tz (torques)
	•	15 time samples per sensor
	•	90 numerical values per execution

The original dataset contains multiple failure types.
For simplicity and robustness, the problem is formulated as Normal vs Anomaly.


Dataset

Source: UCI Machine Learning Repository – Robot Execution Failures Dataset

The dataset is composed of 5 subsets corresponding to different phases of a robotic assembly task:
	•	LP1: approach to grasp position
	•	LP2: part transfer
	•	LP3: positioning after transfer
	•	LP4: approach to ungrasp position
	•	LP5: motion with the part

Key characteristics:
	•	463 executions in total
	•	16 original labels (converted to binary)
	•	Strong class imbalance (more anomalies than normal runs)
	•	Time-series sensor data


Project methodology

The project follows the standard machine learning workflow taught in the course:
	1.	Data exploration
	•	Class distribution
	•	Sensor behavior visualization
	•	Correlation analysis
	2.	Preprocessing
	•	Data cleaning
	•	Feature scaling (StandardScaler)
	•	Handling class imbalance in evaluation
	3.	Feature engineering
	•	Statistical features extracted from time-series:
	•	mean, standard deviation
	•	min, max, range
	•	skewness, kurtosis
	•	linear trend
	•	Reduction from raw time samples to compact, interpretable features
	4.	Dimensionality reduction
	•	Principal Component Analysis (PCA)
	•	Variance analysis
	•	2D projections for visualization
	5.	Supervised learning
	•	Logistic Regression
	•	Support Vector Machine (RBF)
	•	Random Forest
	•	Gradient Boosting
	•	Hyperparameter tuning with cross-validation
	6.	Unsupervised anomaly detection
	•	Isolation Forest
	•	One-Class SVM
	7.	Evaluation and interpretation
	•	Accuracy, precision, recall, F1-score
	•	ROC curves and AUC
	•	Feature importance analysis
	•	Physical interpretation of results


Models used

Supervised models
	•	Logistic Regression (baseline)
	•	Support Vector Machine (RBF kernel)
	•	Random Forest
	•	Gradient Boosting

Random Forest achieved the best overall balance between performance, stability, and interpretability.

Unsupervised models
	•	Isolation Forest
	•	One-Class SVM

These models are useful in scenarios where anomaly labels are incomplete or unavailable.


Key results (summary)
	•	Supervised models achieved high recall, which is critical for safety applications.
	•	Random Forest provided the most stable and interpretable results.
	•	Torque-related features (Tx, Ty, Tz) were the most informative for anomaly detection.
	•	Unsupervised methods showed lower performance but remain relevant for detecting unknown failure patterns.