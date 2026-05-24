# German Credit Risk Optimization Pipeline

A machine learning pipeline developed to classify credit applicants as "Good" or "Bad" risk using an optimized $k$-Nearest Neighbors (KNN) framework. This project focuses heavily on handling class imbalances, analyzing the Bias-Variance tradeoff, and tuning hyperparameters via systematic cross-validation.

## 📌 Project Overview
Credit scoring is inherently imbalanced; the cost of misclassifying a "Bad" risk applicant (default) is significantly higher than missing out on a "Good" risk applicant. This project implements a fully reproducible end-to-end workflow—from categorical encoding and feature scaling to model optimization.

## 🛠️ Tech Stack & Workflow
- **Languages & Libraries:** Python, Scikit-Learn, NumPy, Pandas, Matplotlib
- **Preprocessing:** Categorical Label Encoding, Feature Standardization ($\mu=0, \sigma=1$)
- **Algorithm:** $k$-Nearest Neighbors (KNN)
- **Optimization Strategy:** Grid-search style hyperparameter selection using the maximum mean Cross-Validation (CV) score.

## 📊 The Bias-Variance Tradeoff & Optimization
During hyperparameter tuning, a clear Bias-Variance Tradeoff was observed as the number of neighbors ($k$) varied:
- **Low $k$ (High Variance):** The model overfits the training data, capturing local noise and generating complex decision boundaries (high training accuracy, poor generalization).
- **High $k$ (High Bias):** Training accuracy steadily decreases as $k$ increases because the model averages over a larger neighborhood, smoothing out crucial minority class patterns.

To optimize this, the final hyperparameter selection was mathematically defined by solving for the index of the highest cross-validation mean score:

$$\hat{k} = \arg\max_{k} \text{MeanCVScore}(k)$$

## 📈 Model Performance & Metrics
Due to the dataset's class imbalance, performance was evaluated via a detailed classification report rather than relying blindly on overall accuracy.

### Encoding Reference:
- **Class 0:** Bad Credit Risk (Minority Class)
- **Class 1:** Good Credit Risk (Majority Class - Positive Class)

### Final Evaluation Summary:
- **Bad Risk (0) Recall:** `0.32` — Caught 32% of actual default risks.
- **Good Risk (1) Recall:** `0.89` — Correctly identified 89% of good borrowers.
- **F1-Scores:** `0.40` (Bad) and `0.82` (Good), reflecting a highly risk-averse, conservative model boundary.

### Confusion Matrix Breakdown:
- **19** samples were correctly predicted as "Good" (True Positives for the positive class).
- **40** "Good" credit risks were incorrectly predicted as "Bad" (False Negatives for the positive class), highlighting the model's safe, conservative lending posture.

## 🚀 How to Run Locally
1. Clone the repository:
```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)