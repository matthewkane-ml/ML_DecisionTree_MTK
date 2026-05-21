# Decision Tree Classifier — Diabetes Prediction

> End-to-end binary classification pipeline on the Pima Indians Diabetes dataset: full EDA with outlier handling and feature selection, then a Decision Tree trained and tuned with GridSearchCV — lifting accuracy from 68.2% to 71.4%.

---

## Problem

Predict whether a patient has diabetes based on diagnostic measurements. A hospital wants an interpretable model — one that can be visualised as a tree of clinical decision rules, not a black box.

## Dataset

- **Source:** Pima Indians Diabetes dataset (768 rows × 9 features)
- **Target:** `Outcome` — 1 = diabetes, 0 = no diabetes (65% / 35% class split)
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

## EDA & Preprocessing Pipeline

| Step | Action |
|---|---|
| Impossible zeros | Insulin (48.7% zeros) and SkinThickness (29.6% zeros) dropped entirely |
| Zero imputation | Remaining impossible zeros in Glucose, BloodPressure, BMI replaced with group-stratified median |
| Outlier capping | IQR method on 3 right-skewed features: Pregnancies, DiabetesPedigreeFunction, Age |
| Scaling | StandardScaler on all 6 remaining features |
| Feature selection | SelectKBest (f_classif) → top 4: **Glucose, BMI, Age, Pregnancies** |
| Split | 80/20 stratified train/test (614 train / 154 test) |

**Key EDA finding:** Glucose has the strongest single correlation with Outcome (≈ 0.47). Diabetic patients cluster toward the high-Glucose, high-BMI corner in scatter plots — a clear separation that the tree exploits.

## Model Results

| Model | Accuracy | Notes |
|---|---|---|
| Baseline tree (default params) | **68.2%** | Unpruned — overfits training data |
| Tuned tree (GridSearchCV) | **71.4%** | criterion=entropy, max_depth=5, min_samples_leaf=4 |

**GridSearchCV:** searched criterion × max_depth × min_samples_split × min_samples_leaf with 10-fold cross-validation.

**Tuned classification report:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Diabetes | 0.78 | 0.78 | 0.78 |
| Diabetes | 0.59 | 0.59 | 0.59 |

## Key Takeaways

- **Data cleaning is the real work:** Insulin and SkinThickness had so many biologically impossible zeros (~49% and ~30%) that imputation would have manufactured noise — dropping them was the right call.
- **Pruning prevents overfitting:** The unpruned tree memorises training samples. Constraining `max_depth=5` forces the model to learn general patterns and improves test accuracy by 3 points.
- **Class imbalance matters:** The model performs better on the majority class (No Diabetes). With only 54 positive test cases, recall on the diabetic class is still the harder problem.

## Tech Stack

`Python` · `scikit-learn` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn`

## Run It Locally

```bash
git clone https://github.com/matthewkane-ml/ML_DecisionTree_MTK.git
cd ML_DecisionTree_MTK
pip install -r requirements.txt
jupyter notebook src/DecisionTreeProject_revised.ipynb
```

The trained model is saved to `models/` via `pickle`.

## What I'd Do Next

- Try **Random Forest** or **Gradient Boosting** (XGBoost) on the same dataset to quantify the accuracy gain from ensembling over a single tree
- Address class imbalance with **SMOTE** oversampling or `class_weight="balanced"` and track F1 rather than accuracy as the primary metric
- Add **SHAP values** to explain individual predictions — important for any medical use case where the reasoning behind a decision matters as much as the decision itself

---

**Author:** Matthew Kane — [LinkedIn](https://www.linkedin.com/in/thomas-k-392094410/) · [GitHub portfolio](https://github.com/matthewkane-ml)
