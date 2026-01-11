"""
2.1 Data Processing & Cleaning
"""

import numpy as np
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(path, sep=';')

def preprocess_data(df: pd.DataFrame, fill_strategy: str = 'mean') -> pd.DataFrame:
    """
    Clean and preprocess the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data to be processed
    fill_strategy : str, optional
        Strategy for filling missing values in numerical columns
        ('mean', 'median', 'mode'). Default is 'mean'.

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed DataFrame

    Raises
    ------
    ValueError
        If fill_strategy is not recognized
    """

    # Validate fill_strategy
    if fill_strategy not in ['mean', 'median', 'mode']:
        raise ValueError(
            f"fill_strategy must be one of ['mean', 'median', 'mode'], got {fill_strategy}"
        )

    # Numeric columns might be considered as object and not numeric when there are missing values
    # So to make sure all numeric columns get detected when handling missing values
    # I first make sure that those numeric columns are actually considered numeric instead of object
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # non-numeric stay the same

    # 1. Perform data type conversions as
    # I decided to turn yes/no columns and sex column into binary columns because it's easier
    # to interpret like that
    binary_cols = [
        "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic"
    ]

    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    # For this column I included -1 for missing since I can't fill in sex based on any logic
    df["sex"] = df["sex"].map({"F": 1, "M": 0, 'Missing': -1})

    # 2. Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                if fill_strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean()).round(0)
                elif fill_strategy == 'median':
                    df[col] = df[col].fillna(df[col].median()).round(0)
                else:  # mode
                    df[col] = df[col].fillna(df[col].mode()[0]).round(0)
            elif col=="sex":
                df[col] = df[col].fillna(-1).astype('Int64')
            elif col in binary_cols:
                df[col] = df[col].fillna(0).astype('Int64')
            else:
                # Categorical column
                # I use flagging as 'Missing' here because categorial columns are difficult to fill in
                # based on logic. For example, address or Mjob or Fjob or so on can't be filled in
                df[col] = df[col].fillna('Missing')

    # More data type conversions (from float to int)
    for col in binary_cols:
        df[col] = df[col].astype(int)

    df["sex"] = df["sex"].astype(int)

    # 3. Handle outliers using IQR capping for numerical columns
    # The idea to use IQR came from https://www.geeksforgeeks.org/pandas/handling-outliers-with-pandas/
    # I don't think this data needs to handle this at all, however, task asked for it,
    # so I tried to do this only for those columns that could handle it best
    for col in ["absences", "G1", "G2", "G3"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower, upper).round().astype(int)


    # 4. Create derived features from existing data
    # Academic performance
    df["avg_grade"] = df[["G1", "G2", "G3"]].mean(axis=1).round(2) # average grade
    # I guessed that G3 is final score
    df["pass_fail"] = (df["G3"] >= 10).astype(int) # less than half of G3 as fail (0) and otherwise pass(1)

    # Age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[df["age"].min()-1, 17, 19, df["age"].max()+1],
        labels=["15-17", "18-19", "20+"]
    )

    # Behavioral indicator
    df["total_alcohol"] = df["Dalc"] + df["Walc"] # total consuption of alcohol on both weekdays and workdays
    # df["study_efficiency"] = df["studytime"] / (df["absences"] + 1) # +1 is to avoid division by 0

    # Absence risk category
    df["absence_level"] = pd.cut(
        df["absences"],
        bins=[-1, 5, 15, 100],
        labels=["Low", "Medium", "High"]
    )

    # Support
    df["support_score"] = df["schoolsup"] + df["famsup"] + df["paid"]

    return df
