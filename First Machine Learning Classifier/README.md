Titanic Survival Prediction Using Machine Learning

Project Overview

This project builds a Machine Learning classification model to predict whether a passenger survived the Titanic disaster using Python and Scikit-Learn.

A Decision Tree Classifier is trained using passenger information such as passenger class, gender, age, and ticket fare.

The model is evaluated using unseen test data and compared against a simple majority-class baseline to understand its predictive performance.

Business / Problem Statement

The objective is to predict passenger survival based on available passenger information.

Prediction Target: Survived (0 = No, 1 = Yes)

Input Features:

Passenger Class (Pclass)
Gender (Sex)
Age
Ticket Fare (Fare)
Dataset Information
Dataset: Titanic Passenger Dataset
Source: Seaborn built-in Titanic dataset
Target Variable: survived
Features: pclass, sex, age, fare
Data Type: Public historical dataset
Missing values: Removed before model training

The dataset contains historical passenger information and survival outcomes from the Titanic disaster.

Technologies Used
Python 3.10+
Pandas
NumPy
Seaborn
Matplotlib
Scikit-Learn
Jupyter Notebook
Project Workflow
1. Data Loading

Loaded the Titanic dataset using Seaborn.

2. Data Cleaning
Selected the required columns.
Checked missing values.
Checked duplicate records.
Removed rows containing missing values.
3. Feature Engineering

Converted the gender column into numerical values:

Female = 1
Male = 0
4. Train-Test Split

Split the dataset into:

80% Training Data
20% Testing Data

Used random_state=42 and stratify=y for reproducibility and class balance.

5. Baseline Model

Created a majority-class baseline that always predicts the most common survival outcome in the training dataset.

6. Decision Tree Classification

Trained a Decision Tree Classifier using:

python
DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

7. Model Evaluation

Evaluated the model using:

Accuracy Score
Confusion Matrix
Classification Report
Mean Squared Error
8. Feature Importance

Analysed the importance of passenger features in predicting survival.

9. Overfitting Analysis

Tested tree depths from 1 to 12 and compared training and testing accuracy to understand overfitting.

Evaluation Metrics

Accuracy: Measures the percentage of correct predictions.

Confusion Matrix: Shows correct and incorrect predictions for both survival classes.

Mean Squared Error: Measures the average squared difference between actual binary labels and predicted binary labels. It is included as a supplementary metric; accuracy, precision, recall and F1-score are more informative for classification.

Project Outputs

The project automatically generates:

Original Titanic dataset
Cleaned Titanic dataset
Confusion matrix visualization
Feature importance chart
Overfitting analysis chart
Model comparison report
Test predictions
Results summary
Key Learning Outcomes
Understanding supervised Machine Learning
Data preprocessing and feature selection
Encoding categorical variables
Train-test splitting
Decision Tree Classification
Baseline model comparison
Model evaluation
Confusion matrix interpretation
Feature importance analysis
Understanding overfitting and generalization
Limitations
The dataset is relatively small.
Missing-value rows were removed, which may introduce selection bias.
Only four input features were selected.
A single train-test split may produce variable results.
Decision Trees can overfit training data.
The model demonstrates historical patterns, not real-world modern survival probabilities.
How to Run This Project
Step 1: Clone the repository
bash
git clone https://github.com/YOUR-USERNAME/titanic-survival-prediction-ml.git

Step 2: Open the project folder
bash
cd titanic-survival-prediction-ml

Step 3: Install dependencies
bash
pip install -r requirements.txt

Step 4: Run the Python file
bash
python titanic_survival.py


The script generates the model evaluation results, charts, datasets, and reports.

Author

Your Name

Aspiring Data Analyst | Python | Machine Learning | Data Analysis

Data Source

Seaborn Titanic dataset: https://github.com/mwaskom/seaborn-data

This project was developed for educational purposes as an introduction to Machine Learning classification using Scikit-Learn.