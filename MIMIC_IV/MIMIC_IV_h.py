import pandas as pd
import numpy as np


"""
    Calculate the age of the patient at the time of a lab event.
    
    Parameters:
    - row (pandas.Series): A row of the DataFrame.
    - time_col (str): The column name for the lab time.
    - anchor_year_col (str): The column name for the anchor year.
    - anchor_age_col (str): The column name for the anchor age.

    Returns:
    - int: The age of the patient at the time of the lab event.
"""

def calculate_age_at_time(row, time_col, anchor_year_col, anchor_age_col):
    # Calculate age at the time of the lab
    if pd.isnull(row[time_col]):
        return np.nan
    time_year = row[time_col].year
    anchor_year = row[anchor_year_col]
    return row[anchor_age_col] + (time_year - anchor_year)


"""
    Vectorized calculation approach to estimated Glomerular Filtration Rate (eGFR) using the CKD-EPI formula.
    
    Parameters:
    - df (pandas.DataFrame): DataFrame containing patient data, including creatinine, sex, and age columns.
    - crt_col (str): The column name for the creatinine levels.
    - sex_col (str): The column name for the patient's sex.
    - sex_flg (str): The value in the sex column that indicates the female gender.
    - age_col (str): The column name for the patient's age.

    Returns:
    - pandas.Series: A Series representing the eGFR for each patient.
"""

def calc_eGFR(df, crt_col, sex_col, sex_flg, age_col):
    # Calculate numpy series for constants
    k = np.where(df[sex_col] == sex_flg, 0.7, 0.9)
    alpha = np.where(df[sex_col] == sex_flg, -0.241, -0.302)
    gender_scalar = np.where(df[sex_col] == sex_flg, 1.012, 1.0)
    
    # Calculate pandas series for Serum Creatine / k and min and max components
    scr_k_ratio = df[crt_col] / k
    min_scr_k = np.minimum(scr_k_ratio, 1)
    max_scr_k = np.maximum(scr_k_ratio, 1)
    
    # Calculate eGFR via vectorized operation
    eGFR = 142 * (min_scr_k ** alpha) * (max_scr_k ** -1.2) * (0.9938 ** age_col) * gender_scalar
    
    return eGFR


"""
    Safely calculates eGFR for a given row of the DataFrame.

    Parameters:
    - row (pandas.Series): A row of the DataFrame.
    - crt_col (str): The name of the creatinine column.
    - lab_time_col (str): The name of the time column for the lab.

    Returns:
    - float: The calculated eGFR or NaN if data is missing.
    """

def safe_calc_eGFR(row, crt_col, time_col):
    # Check for NaN values
    if pd.isnull(row[crt_col]) or pd.isnull(row[time_col]):
        return np.nan

    # Calculate age at the time of the lab
    age_at_lab_time = calculate_age_at_time(row, time_col, 'anchor_year', 'anchor_age')
    
    # Use the previously defined calc_eGFR function
    return calc_eGFR(row, 'Creatinine_first', 'gender', 'F', age_at_lab_time)


"""
    Convert ICD-9 codes to corresponding CKD stage labels.
    
    Parameters:
    - df (pandas.DataFrame): DataFrame containing patient data with ICD-9 codes for CKD stages.
    - icd_col (str): The column name for the ICD-9 codes indicating CKD stages.

    Returns:
    - pandas.Series: A Series representing the CKD stage (1-6) or -1 for unclassified stages.
"""

def convert_CKD_stage(df, icd_col):
    def convert_label(icd_col):
        if icd_col == 5851:
            return 1
        elif icd_col == 5852:
            return 2
        elif icd_col == 5853:
            return 3
        elif icd_col == 5854:
            return 4
        elif icd_col == 5855:
            return 5
        elif icd_col == 5856:
            return 6
        elif icd_col == 5859:
            return -1

    return df[icd_col].apply(convert_label)


"""
    Labels the stage of chronic kidney disease based on patient eGFR levels.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing patient data with eGFR levels.
    - egfr_col (str): The column name for the eGFR levels.

    Returns:
    - pandas.Series: A Series representing the CKD stage (1-5) for each patient.
"""

def label_CKD_stage(df, egfr_col):
    def label_patient(egfr):
        if pd.isna(egfr):
            return np.nan
        elif egfr >= 90:
            return 1
        elif egfr >= 60:
            return 2
        elif egfr >= 30:
            return 3
        elif egfr >= 15:
            return 4
        else:
            return 5

    return df[egfr_col].apply(label_patient)