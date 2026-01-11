# Data Science with Python
# Final Project
# Project Title: Student Performance Prediction
### Task description (copied from assignment pdf by Nika Gagua)
**Objective:** Analyze factors affecting student performance and predict academic outcomes.

**Recommended Datasets (beginner-friendly):**
- Student Performance dataset (UCI ML Repository)
- Students Exam Scores dataset (Kaggle)
- Academic Success dataset (Kaggle)
- 
**What to Do:**
- Clean and explore student data
- Visualize grade distributions, study time effects, parental education impact
- Create correlation heatmap for numeric features
- Build Linear Regression to predict final grades
- Build Decision Tree to classify pass/fail
- Discuss which factors most influence student success
- 
**ML Approach:** Regression (predict scores) or Classification (predict pass/fail)


I got my data from https://archive.ics.uci.edu/dataset/320/student+performance.
I use "student-mat.csv" specifically. It can be found as "data/raw.csv" in this project.

You need to run all the .ipynb files to see results. Some of the results can be found in reports directory.

Part 2.1 Data Processing & Cleaning is done in "data_processing.py".

**Initial State:**

Number of record: 396

Number of features (columns): 33

Missing values: 9

**Data cleaning steps:**

1. The function preprocess_data has parameter fill_strategy for numerical values, so first we check if the parameter is valid.
2. Cast numeric dtypes to numeric since they could be considered as object if NaN values occur
3. Turn yes/no columns and 'sex' column into binary columns with 1/0 values
4. Handle missing numeric values based of fill_strategy and categorial columns by flagging as 'Missing'
5. Cast float to int for binary columns
6. Handle outliers using IQR capping for numerical columns
7. Create derived features: avg_grade, pass_fail, age_group, total_alcohol, absence_level, support_score

Justification for handling decisions are documented as comments in the code

Documentation for visualization and ML models can be found with the code.
