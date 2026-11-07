# BEED-EEG-Classification
Comparative evaluation of five machine learning classifiers for EEG-based epilepsy classification using the BEED dataset.

# EEG-Based Epilepsy Classification

## a. Problem Statement
This project investigates the classification of EEG
recordings using five supervised machine learning
algorithms.
The objective is to compare the classification
performance of Logistic Regression, Decision Tree,
KNN, Gaussian Naive Bayes and Support Vector Machine.
The models are evaluated using six classification
metrics on the same held-out test dataset.
This project is for educational purposes only,
not clinical diagnosis.

## b. Dataset Description
Dataset Name:
BEED - Bangalore EEG Epilepsy Dataset
Source:
UCI Machine Learning Repository
Dataset URL:
https://archive.ics.uci.edu/dataset/1134
DOI:
https://doi.org/10.24432/C5K33B
Donation Year: 2025
Original dataset rows: 8000
Total dataset columns: 17
Number of model input features: 16
Feature names: X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, X16
Target: y
Number of classes: 4
Target label frequencies:
y
0    2000
1    2000
2    2000
3    2000
Missing values (entire dataset): 0
Duplicate rows (whole-row matches): 0
Missing predictor values: 0
Subject/group identifier used: None available/configured
NOTE: Duplicate observations were counted, not automatically discarded.

TRAIN/TEST DETAILS
Split method: Stratified row-level split (NOT patient-independent)
Train records: 6400
Test records: 1600
Train + test = original: True
Test fraction requested: 0.2
Random state: 42
Training target counts:
y
2    1600
1    1600
3    1600
0    1600
Test target counts:
y
3    400
2    400
1    400
0    400
Preprocessing: StandardScaler fitted on training data for LR, KNN and SVM.
No scaling for Decision Tree or Gaussian Naive Bayes.
LIMITATION: Subject-independent performance cannot be claimed without subject IDs.
Saved: test_data.csv (features plus target)

## c. GitHub Repository Link

## d. Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Model Comparison
### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.4544 | 0.6576 | 0.4833 | 0.4544 | 0.4609 | 0.2744 |
| Decision Tree | 0.8031 | 0.9253 | 0.8129 | 0.8031 | 0.8018 | 0.7420 |
| KNN | 0.9538 | 0.9953 | 0.9554 | 0.9538 | 0.9539 | 0.9388 |
| Naive Bayes | 0.6975 | 0.9062 | 0.6948 | 0.6975 | 0.6925 | 0.5990 |
| SVM | 0.7444 | 0.9272 | 0.7731 | 0.7444 | 0.7310 | 0.6791 |


### Performance Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Achieved the lowest performance among all models (Accuracy = 45.44%, F1 = 0.4609, MCC = 0.2744). The model showed significant confusion between classes 2 and 3, indicating that linear decision boundaries are insufficient for this dataset. |
| Decision Tree | Performed well with 80.31% accuracy and 0.8018 F1-score. The model classified Class 0 almost perfectly but showed reduced recall for Class 3. Overall, it provided good predictive performance and interpretability. |
| KNN | Produced the best overall results with 95.38% accuracy, 0.9539 F1-score, 0.9388 MCC, and 0.9953 AUC. The confusion matrix showed very few misclassifications across all classes, indicating excellent class separation. |
| Naive Bayes | Achieved moderate performance with 69.75% accuracy and 0.6925 F1-score. It classified Class 0 effectively but struggled with Classes 1 and 3, likely due to the independence assumptions of the model. |
| SVM | Achieved 74.44% accuracy and 0.7310 F1-score with a strong AUC of 0.9272. However, recall for Class 3 was relatively low, and training time was significantly higher than the other models. |
| Overall Winner | KNN is the overall winner. It achieved the highest Accuracy (95.38%), AUC (0.9953), Precision (0.9554), Recall (0.9538), F1-score (0.9539), and MCC (0.9388), making it the most effective classifier for the BEED EEG dataset. |

### References
BEED: Bangalore EEG Epilepsy Dataset.
UCI Machine Learning Repository.
DOI: 10.24432/C5K33B
