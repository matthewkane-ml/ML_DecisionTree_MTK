# %% [markdown]
# ## Decision Tree Project

# 

# **Problem Statement:** Predict whether a patient has diabetes based on diagnostic measures.


# %%
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn import tree

from pickle import dump

import warnings

warnings.filterwarnings('ignore')


# %% [markdown]
# ## 1. Load the Dataset


# %%
URL = "https://breathecode.herokuapp.com/asset/internal-link?id=421&path=diabetes.csv"

df = pd.read_csv(URL)



df.head()


# %% [markdown]
# ## 2. High-Level View of the Data


# %%
print("Shape:")

print(df.shape)



print("\ndtypes & non-null counts:")

df.info()


# %%
# Identify numerical variables and separate target

target = "Outcome"

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

feature_cols = [c for c in numerical_cols if c != target]


# %% [markdown]
# ## 3. Data Cleaning


# %%
# Null values

print("Missing Values:")

print(df.isnull().sum())


# %%
# Finding the Duplicates

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")


# %%
# Biologically impossible zeros

# Glucose, BloodPressure, SkinThickness, Insulin, BMI cannot be 0 (because the patient would be dead if they were)

imp_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

print("Zero counts:")



for col in imp_zero_cols:

    n = (df[col] == 0).sum()

    pct = n / len(df) * 100

    print(f"  {col:}: {n}  ({pct:.1f} %)")


# %% [markdown]
# Here we can notice something: The Insulin and SkinThickness columns have 48.7% and 29.6% impossible zeroes, respectively. Meaning, if we replaced those values with NaN and then did the median imputation process, we would basically have one column that was about half "garbage" and another column that was about 30% "garbage". It would be better to just get rid of the columns. 


# %%
df = df.drop(["Insulin", "SkinThickness"], axis=1)



imp_zero_cols = imp_zero_cols[:2] + imp_zero_cols[4:]



feature_cols.remove("SkinThickness")

feature_cols.remove("Insulin")



feature_cols


# %%
# Replace zeros with NaN, then impute with median 

df[imp_zero_cols] = df[imp_zero_cols].replace(0, np.nan)



for col in imp_zero_cols:

    df[col] = df.groupby(target)[col].transform(

        lambda x: x.fillna(x.median())

    )



print("Missing values after imputation:")

print(df.isnull().sum())


# %% [markdown]
# ## 4. Descriptive Statistics


# %%
df.describe()


# %% [markdown]
# ## 5. Univariate Analysis


# %%
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

axes = axes.flatten()



for i, col in enumerate(feature_cols):

    ax = axes[i]

    sns.histplot(df[col], kde=True, ax=ax, color="blue", edgecolor="white")

    ax.set_title(col, fontsize=12, fontweight="bold")

    ax.set_xlabel("")



# Last subplot: target counts

ax = axes[len(feature_cols)]

sns.countplot(x=target, data=df, ax=ax)

ax.set_title("Outcome (0=No Diabetes, 1=Diabetes)", fontsize=12, fontweight="bold")

ax.bar_label(ax.containers[0], fontsize=9)



for j in range(len(feature_cols) + 1, len(axes)):

    axes[j].set_visible(False)



plt.suptitle("Univariate Distributions", fontsize=15, fontweight="bold", y=1.01)

plt.tight_layout()

plt.show()


# %% [markdown]
# ## 6. Verbal Analysis â€” Univariate Graphs

# 

# **Pregnancies** Right-skewed; most patients have 0â€“3 pregnancies.

# 

# **Glucose** Higher values are associated with diabetes.

# 

# **BloodPressure** Near-normal distribution centred around 70 mm Hg. Some extreme low/high values remain.

# 

# **BMI** Roughly normal, centred ~32. Minimal skew after imputation.

# 

# **DiabetesPedigreeFunction** Strongly right-skewed with a long tail; most values are < 1.0.

# 

# **Age** Right-skewed; most patients are in their 20sâ€“30s. The tail extends to 81.

# 

# **Outcome** Imbalanced: ~65 % negative (0) vs ~35 % positive (1). This should be considered during modelling.

# 

# **Overall takeaways:**

# - Several features (DiabetesPedigreeFunction, Pregnancies, Age) are skewed and contain outliers that need treatment.

# - The target is moderately imbalanced; techniques like stratified splitting or class-weight adjustment may be helpful later.


# %% [markdown]
# ## 7. Multivariate Analysis


# %%
plt.figure(figsize=(10, 8))

corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",

            cmap="coolwarm", center=0, linewidths=0.5,

            square=True, cbar_kws={"shrink": 0.8})

plt.title("Correlation Heatmap", fontsize=14, fontweight="bold")

plt.tight_layout()

plt.show()


# %%
# --- Boxplots by Outcome ---

fig, axes = plt.subplots(2, 3, figsize=(18, 9))

axes = axes.flatten()

palette = {0: "#4C9BE8", 1: "#E8694C"}



for i, col in enumerate(feature_cols):

    sns.boxplot(x=target, y=col, data=df, ax=axes[i], linewidth=1.2)

    axes[i].set_title(col, fontsize=11, fontweight="bold")

    axes[i].set_xlabel("Outcome (0=No, 1=Yes)")



plt.suptitle("Feature Distributions by Diabetes Outcome", fontsize=14,

             fontweight="bold", y=1.01)

plt.tight_layout()

plt.show()


# %%
# Scatter plots: key feature pairs

pairs = [("Glucose", "BMI"), ("Glucose", "Age"), ("BMI", "Age")]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))



for ax, (x, y) in zip(axes, pairs):

    sns.scatterplot(data=df, x=x, y=y, hue=target,

                     alpha=0.6, ax=ax)

    ax.set_title(f"{x} vs {y}", fontweight="bold")



plt.suptitle("Scatter Plots â€” Key Feature Pairs", fontsize=13,

             fontweight="bold", y=1.02)

plt.tight_layout()

plt.show()


# %% [markdown]
# ## 8. Verbal Analysis â€” Multivariate Graphs

# 

# ### Correlation Heatmap

# - Glucose has the strongest correlation with Outcome (â‰ˆ 0.47), confirming it as the most predictive single feature.

# - BMI and Age show moderate positive correlations with the outcome (â‰ˆ 0.29 and 0.24 respectively).

# 

# ### Boxplots by Outcome

# - Glucose: Diabetic patients show a markedly higher median glucose level â€” the clearest separator.

# - BMI: Diabetic patients are consistently heavier; distributions overlap but medians differ by ~5 units.

# - Age: Diabetic patients tend to be older; the diabetic group also shows more variability.

# 

# ### Scatter Plots

# - Glucose vs BMI: Diabetic cases cluster toward the high-Glucose, high-BMI corner â€” a clear separation.

# - Glucose vs Age: Older + high-glucose patients are predominantly diabetic.

# - BMI vs Age: Overlap is larger; BMI alone without glucose is a weaker separator.


# %% [markdown]
# ## 9. Feature Engineering


# %%
df_eng = df.copy()



# Outlier handling with IQR capping 

skewed_cols = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]



for col in skewed_cols:

    Q1, Q3 = df_eng[col].quantile([0.25, 0.75])

    IQR = Q3 - Q1

    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    n_clipped = ((df_eng[col] < lower) | (df_eng[col] > upper)).sum()

    df_eng[col] = df_eng[col].clip(lower, upper)

    print(f"  {col:30s}: {n_clipped} outliers capped  [{lower:.2f} â€“ {upper:.2f}]")


# %%
# No encoding needed â€” all features are numeric

print(f"Engineered dataframe shape: {df_eng.shape}")


# %%
# Scaling 

X = df_eng[feature_cols].copy()

y = df_eng[target].copy()



scaler = StandardScaler()

X_scaled = pd.DataFrame(scaler.fit_transform(X),

                         columns=feature_cols,

                         index=X.index)



print("Features after scaling:")

X_scaled.describe().loc[["mean", "std"]].round(4)


# %% [markdown]
# ## 10. Feature Selection â€” SelectKBest (f_classif)


# %%
selector = SelectKBest(score_func=f_classif, k="all")

selector.fit(X_scaled, y)



feat_scores = pd.DataFrame({

    "Feature": feature_cols,

    "F-Score": selector.scores_,

    "p-value": selector.pvalues_

}).sort_values("F-Score", ascending=False).reset_index(drop=True)



print(feat_scores.to_string(index=False))


# %%
# Visualise

plt.figure(figsize=(9, 5))

sns.barplot(data=feat_scores, x="F-Score", y="Feature",

            palette="Blues_r", edgecolor="white")

plt.axvline(feat_scores["F-Score"].iloc[3], color="red",

            linestyle="--", label="Top-4 cut-off")

plt.title("SelectKBest â€” F-Scores (f_classif)", fontsize=13, fontweight="bold")

plt.legend()

plt.tight_layout()

plt.show()


# %%
# Select Top-4 features

K = 4

top_k = feat_scores["Feature"].head(K).tolist()

print(f"Selected top-{K} features: {top_k}")



X_selected = X_scaled[top_k]


# %% [markdown]
# ## 11. Train / Test Split and Save


# %%
X_train, X_test, y_train, y_test = train_test_split(

    X_selected, y,

    test_size=0.2,

    random_state=42,

    stratify=y           # preserve class ratio

)



print(f"Train size : {X_train.shape}  |  Test size : {X_test.shape}")

print(f"Train outcome balance:\n{y_train.value_counts(normalize=True).round(3)}")

print(f"\nTest  outcome balance:\n{y_test.value_counts(normalize=True).round(3)}")


# %%
# Save the final datasets to CSV files

X_train.assign(Outcome=y_train.values).to_csv("train.csv", index=False)

X_test.assign(Outcome=y_test.values).to_csv("test.csv",  index=False)


# %% [markdown]
# ## DECISION TREE MODEL:


# %%
train_data = pd.concat([X_train, y_train], axis=1)

test_data = pd.concat([X_test, y_test], axis=1)

total_data = pd.concat([train_data, test_data]).reset_index(drop=True)



plt.figure(figsize=(12, 6))

pd.plotting.parallel_coordinates(total_data, "Outcome", color=("green", "red"))

plt.show()


# %%
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)


# %%
fig = plt.figure(figsize=(15, 15))

tree.plot_tree(model, feature_names=list(X_train.columns),

              class_names=["No Diabetes", "Diabetes"], filled=True)

plt.show()


# %%
y_pred = model.predict(X_test)



print(f"Baseline Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")

print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))


# %%
hyperparams = {

    "criterion": ["gini", "entropy"],

    "max_depth": [None, 5, 10, 20],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4]

}



grid = GridSearchCV(model, hyperparams, scoring="accuracy", cv=10)


# %%
grid.fit(X_train, y_train)

print(f"Best hyperparameters: {grid.best_params_}")


# %%
model = DecisionTreeClassifier(criterion="entropy", max_depth=5,

                               min_samples_leaf=4, min_samples_split=2,

                               random_state=42)

model.fit(X_train, y_train)


# %%
y_pred = model.predict(X_test)



print(f"Tuned Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")

print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))


# %%
with open("../models/tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav", "wb") as f:

    dump(model, f)


