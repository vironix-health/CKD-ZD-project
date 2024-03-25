# Chronic Kidney Disease ML Project

This is a repository containing data and analysis for the freely available MIMIC-IV database, and the study "Predicting hospital admission at emergency department triage using machine learning" by Hong et al.

MIMIC-IV [4] (Medical Information Mart for Intensive Care, Version IV) is an extensive database comprising de-identified health-related data associated with over 180,000 thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center in Boston, Massachusetts. It is an update and expansion of the earlier MIMIC-III database and includes data such as vital signs, medications, laboratory measurements, observations and notes taken by care providers, fluid balance, procedure codes, diagnostic codes, imaging reports, hospital length of stay, survival data, and more.

The study published by Hong et al. [1] utilizes extensive patient data and ML models to predict hospital admissions, focusing on the relevance of various patient features including demographic information, emergency department triage variables, and medical history.

# Repository Contents

The functionality of the repository contents can be described in three distinct subdivisions: preprocessing the MIMIC-IV dataset, preprocessing the Hong et al. dataset, and feature analysis of a merged set combining data from both sources.

## MIMIC-IV

    /mimic_iv_extract

        mimic_iv_ovrvw.xlsx :

            An excel file providing an overview of the disease and hospitalisation frequencies recorded in the MIMIC-IV database. It specifically focuses on CKD comorbidities including hypertension, anemia, diabetes, cardiac disarrhythmias, ischemic heart disease, thyroid disease, heart failure, cerebrovascular disease, and PVD.

        ckd_stage_frequency.xlsx :

            An excel file providing an overview of the patient and hospitalisation frequencies for each CKD ICD-9 code recorded in the MIMIC-IV database. There are a total of seven ICD-9 codes extracted corresponding to the five progressive stages of CKD, end stage renal disease, and unspecified stage CKD.

        df_all_patients.pkl :

            A pickle file containing a Pandas dataframe of all patients in the MIMIC-IV database, and basic identifiers including gender anchor information.

        df_admissions.pkl :

            A pickle file containing a Pandas dataframe of all admissions to the Beth Israel Deaconess Medical Center recorded in the MIMIC-IV database. Admission timeframes, associated locations, insurance details, and some further demographics such as race and marital status are included.

        df_past_medical_history.pkl :

            A pickle file containing a Pandas dataframe of patient past medical history including previous diagnoses and notes by healthcare providers, as well as associated chart times for the records.

        df_icd_codes_with_description.pkl :

            A pickle file containing a Pandas dataframe of the ICD-9 codes and long-title diagnoses associated with all admissions to the Beth Israel Deaconess Medical Center recorded in the MIMIC-IV database.

        df_lab_items.pkl :

            A pickle file containing a Pandas dataframe of all lab items recorded in the MIMIC-IV database, including the lab item ID's, labels, categories, and associated lab fluid (e.g. Cerebrospinal Fluid).

        df_ckd_lab_items.pkl :

            A pickle file containing a Pandas dataframe of lab items recorded in the MIMIC-IV database which are specifically related to CKD, including the lab item ID's, labels, categories, and associated lab fluid (e.g. Cerebrospinal Fluid).

        df_height_weight.pkl :

            A pickle file containing a Pandas dataframe with first, minimum, maximum, and mean measurements for patient height and weight associated with unique hospital admissions in the MIMIC-IV database.

The above subdirectory /mimic_iv_extract contains .pkl and .xlsx data extracts from the MIMIC-IV database.

    mimic_iv_extract.ipynb

A Jupyter Notebook that first establishes a connection to the MIMIC-IV database via BigQuery, and subsequently extracts a series of relevant dataframes. The datasets are saved as pickle files in the /mimic_ic_extracts subdirectory. Each extracted dataframe is comprehensively detailed within the notebook, and in the README.md under the /mimic_iv_extract subdirectory section.

    mimic_iv_ovrvw.ipynb

A Jupyter Notebook providing an overview of the disease and hospitalisation frequencies recorded in the MIMIC-IV database, leveraging data from the df_all_patients.pkl and df_icd_codes_with_description.pkl pickle files. The disease and hospitalisation frequency information is summarized in a Pandas dataframe, and saved as the Excel file, mimic_iv_ovrvw.xlsx, in the /mimic_ic_extracts subdirectory.

    mimic_iv_ckd_ovrvw.ipynb

A Jupyter Notebook providing an overview of the patient and hospitalisation frequencies for each CKD ICD-9 code recorded in the MIMIC-IV database, using the df_icd_codes_with_description.pkl pickle file. The patient and hospitalisation frequency information is summarized in a Pandas dataframe, and saved as the Excel file,  mimic_iv_ckd_ovrvw.xlsx, in the /mimic_ic_extracts subdirectory.

## "Predicting hospital admission at emergency department triage using machine learning" Hong et al.

    /hong_et_al

        5v_cleandf.RData : 

            This is the raw dataset as provided by Hong et al. It includes a total of 560,486 patient entries, each with 972 features including demographics, ED triage variables, and detailed patient medical histories, including medication. This dataset is provided in RData format for compatibility with R statistical software.

        df.csv : 

            This is a CSV file conversion of the 5v_cleandf.RData file. It maintains the integrity of the original data, including the entirety of the original set.

        df_updt.csv : 

            This is an updated version of df.csv after processing by eGFR.ipynb. It includes the newly calculated eGFR and CKD stage columns.

The above subdirectory /hong_et_al contains .Rdata and .csv data extracts from the dataset used in the study "Predicting hospital admission at emergency department triage using machine learning" by Hong et al.

    eGFR.ipynb

A Jupyter Notebook that details the process of calculating the Estimated Glomerular Filtration Rate (eGFR) for all patients in the dataset. The eGFR calculation is based on the CKD-EPI Creatinine Equation [2]. Following the calculation, this notebook also labels each patient with a Chronic Kidney Disease (CKD) stage ranging from 1 to 5 based on their eGFR values [3]. These calculated values and CKD stages are then added as new columns to the dataset. Additionally, the notebook provides a summary of statistical results derived from the newly added information.

    featuresID.ipynb

A Jupyter Notebook that details the process of identifying the features most relevant to eGFR via XGBoost regression. The data is preprocessed and partitioned into training and test data, then an XGBRegressor is trained and used to evaluate the feature importance. The model's feature importance rankings are displayed and visualized using Matplotlib and Seaborn.

# How to Use

To work with the RData file, it is necessary to have R statistical software installed on your machine. The dataset can then be loaded into your R environment using the load() function.

The CSV and Excel files can be viewed and manipulated using standard data processing tools such as Microsoft Excel, Python (pandas library), or R.

To execute analyses in the Jupyter Notebooks, it is necessary to have Jupyter installed, along with Python and the necessary libraries (pandas, numpy, os, google-cloud-bigquery, xgboost, seaborn, matplotlib, sklearn). With the prerequisites installed Jupyter Notebook can be launched from the project directory, and the eGFR.ipynb, featuresID.ipynb, mimic_iv_extract.ipynb, mimic_iv_ovrvw.ipynb, and mimic_iv_ckd_ovrvw.ipynb scripts can be executed.

# Contributors

Vaibhav Mahajan, (St Catherine's College, University of Oxford)

# References

[1] https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201016

[2] https://www.kidney.org/content/ckd-epi-creatinine-equation-2021

[3] https://www.nhs.uk/conditions/kidney-disease/diagnosis/

[4] https://physionet.org/content/mimiciv/2.2/
 
# Supplementary Resources

[5] https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0271619#abstract0

[6] https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0295234#sec003