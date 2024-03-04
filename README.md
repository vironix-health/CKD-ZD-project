# Chronic Kidney Disease ML Project

This repository contains data and analysis for the study "Predicting hospital admission at emergency department triage using machine learning" by Hong et al. The study utilizes extensive patient data and ML models to predict hospital admissions, focusing on the relevance of various patient features, including demographic information, emergency department triage variables, and medical history.

# Repository Contents

5v_cleandf.RData:

    This is the raw dataset as provided by Hong et al. It includes a total of 560,486 patient entries, each with 972 features including demographics, ED triage variables, and detailed patient medical histories, including medication. This dataset is provided in RData format for compatibility with R statistical software.

df.csv:

    This is a CSV file conversion of the 5v_cleandf.RData file. It maintains the integrity of the original data, including the entirety of the original set.

eGFR.ipynb:

    A Jupyter Notebook that details the process of calculating the Estimated Glomerular Filtration Rate (eGFR) for all patients in the dataset. The eGFR calculation is based on the CKD-EPI Creatinine Equation (2021). Following the calculation, this notebook also labels each patient with a Chronic Kidney Disease (CKD) stage ranging from 1 to 5 based on their eGFR values. These calculated values and CKD stages are then added as new columns to the dataset. Additionally, the notebook provides a summary of statistical results derived from the newly added information.

# How to Use

    To work with the RData file, it is necessary to have R statistical software installed on your machine. The dataset can then be loaded into your R environment using the load() function.

    The CSV file can be viewed and manipulated using standard data processing tools such as Microsoft Excel, Python (pandas library), or R.

    To execute the analysis in the Jupyter Notebook, it is necessary to have Jupyter installed, along with Python and the necessary libraries (pandas and numpy). With the prerequisites installed, Jupyter Notebook can be launched from the project directory and the eGFR.ipynb script executed.