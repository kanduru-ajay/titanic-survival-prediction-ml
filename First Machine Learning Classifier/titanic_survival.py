# ==========================================================
# TITANIC SURVIVAL PREDICTION
# Machine Learning - Decision Tree Classifier
# ==========================================================

# 1. IMPORT LIBRARIES

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    mean_squared_error
)

# 2. CREATE PROJECT DIRECTORIES

os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

# 3. LOAD TITANIC DATASET

df = sns.load_dataset("titanic")

print("\nORIGINAL DATASET")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

# 4. SELECT REQUIRED FEATURES

df = df[
    ["survived", "pclass", "sex", "age", "fare"]
].copy()

# Save original selected data

df.to_csv("data/titanic_original.csv", index=False)

# 5. DATA CLEANING

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Remove rows with missing values

df = df.dropna().copy()

# Encode gender

df["sex"] = (df["sex"] == "female").astype(int)

# Save cleaned data

df.to_csv("data/titanic_cleaned.csv", index=False)

print("\nCLEANED DATASET")
print(df.head())

print("\nCLEANED DATASET SHAPE")
print(df.shape)

# 6. DEFINE FEATURES AND TARGET

X = df.drop(columns=["survived"])
y = df["survived"]

print("\nFEATURES")
print(X.head())

print("\nTARGET")
print(y.head())

# 7. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA:", X_train.shape)
print("TEST DATA:", X_test.shape)

# 8. BASELINE MODEL

# Always predict the most common class in training data

majority_class = y_train.mode()[0]

baseline_predictions = np.full(
    len(y_test),
    majority_class
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("\nBASELINE ACCURACY:",
      round(baseline_accuracy, 3))

# 9. TRAIN DECISION TREE CLASSIFIER

model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

# 10. MAKE PREDICTIONS

y_pred = model.predict(X_test)

# 11. MODEL EVALUATION

model_accuracy = accuracy_score(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

print("\nMODEL ACCURACY:",
      round(model_accuracy, 3))

print("\nMEAN SQUARED ERROR:",
      round(mse, 3))

print("\nCLASSIFICATION REPORT")

report = classification_report(
    y_test,
    y_pred,
    target_names=["Did Not Survive", "Survived"],
    output_dict=True
)

print(classification_report(
    y_test,
    y_pred,
    target_names=["Did Not Survive", "Survived"]
))

# 12. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Did Not Survive", "Survived"]
)

disp.plot(cmap="Blues")

plt.title("Titanic Survival Prediction - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "charts/confusion_matrix.png",
    dpi=300
)

plt.close()

# 13. FEATURE IMPORTANCE

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance)

importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)

plt.title("Feature Importance - Decision Tree")
plt.tight_layout()

plt.savefig(
    "charts/feature_importance.png",
    dpi=300
)

plt.close()

# 14. TEST DIFFERENT TREE DEPTHS

depth_results = []

for depth in range(1, 13):

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    test_prediction = tree.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_prediction
    )

    test_accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    depth_results.append({
        "max_depth": depth,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    })

depth_df = pd.DataFrame(depth_results)

print("\nTREE DEPTH COMPARISON")
print(depth_df.round(3))

depth_df.to_csv(
    "outputs/depth_comparison.csv",
    index=False
)

# Plot overfitting analysis

plt.figure(figsize=(10, 6))

plt.plot(
    depth_df["max_depth"],
    depth_df["train_accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depth_df["max_depth"],
    depth_df["test_accuracy"],
    marker="o",
    label="Testing Accuracy"
)

plt.xlabel("Maximum Tree Depth")
plt.ylabel("Accuracy")

plt.title("Decision Tree Overfitting Analysis")

plt.legend()
plt.tight_layout()

plt.savefig(
    "charts/overfitting_analysis.png",
    dpi=300
)

plt.close()

# 15. COMPARE BASELINE AND MODEL

comparison = pd.DataFrame({
    "Model": [
        "Baseline",
        "Decision Tree"
    ],
    "Accuracy": [
        baseline_accuracy,
        model_accuracy
    ]
})

print("\nMODEL COMPARISON")
print(comparison.round(3))

comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# 16. GENERATE RESULTS SUMMARY

best_feature = importance.iloc[0]["Feature"]

with open("RESULTS.md", "w", encoding="utf-8") as file:

    file.write("# Titanic Survival Prediction - Results\n\n")

    file.write("## Model Performance\n\n")

    file.write(
        f"- Baseline accuracy: {baseline_accuracy:.3f}\n"
    )

    file.write(
        f"- Decision Tree accuracy: {model_accuracy:.3f}\n"
    )

    file.write(
        f"- Mean Squared Error: {mse:.3f}\n"
    )

    file.write(
        f"- Training records: {len(X_train)}\n"
    )

    file.write(
        f"- Testing records: {len(X_test)}\n\n"
    )

    file.write("## Confusion Matrix\n\n")
    file.write(
        "The matrix rows represent actual classes and "
        "columns represent predicted classes.\n\n"
    )

    file.write(f"- True negatives: {cm[0, 0]}\n")
    file.write(f"- False positives: {cm[0, 1]}\n")
    file.write(f"- False negatives: {cm[1, 0]}\n")
    file.write(f"- True positives: {cm[1, 1]}\n\n")

    file.write("## Most Important Feature\n\n")

    file.write(
        f"The feature with the highest Decision Tree "
        f"importance is **{best_feature}**.\n\n"
    )

    file.write("## Limitations\n\n")

    file.write(
        "- The dataset is relatively small.\n"
        "- Rows with missing values were removed.\n"
        "- Only four passenger features were used.\n"
        "- The model may overfit the training dataset.\n"
        "- Historical Titanic survival patterns do not "
        "represent modern passenger survival probabilities.\n"
    )

    file.write(
        "\n## Conclusion\n\n"
        "The Decision Tree model was evaluated on a "
        "held-out test dataset and compared with a "
        "majority-class baseline. Its usefulness should "
        "be judged by test performance, the confusion "
        "matrix, and its generalization gap.\n"
    )

# 17. SAVE MODEL PREDICTIONS

predictions = X_test.copy()

predictions["actual_survival"] = y_test
predictions["predicted_survival"] = y_pred

predictions.to_csv(
    "outputs/test_predictions.csv",
    index=False
)

# 18. FINAL OUTPUT

print("\n====================================")
print("TITANIC ML PROJECT COMPLETED")
print("====================================")

print("Generated files:")
print("- data/titanic_original.csv")
print("- data/titanic_cleaned.csv")
print("- charts/confusion_matrix.png")
print("- charts/feature_importance.png")
print("- charts/overfitting_analysis.png")
print("- outputs/feature_importance.csv")
print("- outputs/depth_comparison.csv")
print("- outputs/model_comparison.csv")
print("- outputs/test_predictions.csv")
print("- RESULTS.md")