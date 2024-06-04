# Chronic Kidney Disease ML Project

This is a repository containing analysis for the freely available MIMIC-IV database, as well as two supplementary datasets from the studies "Predicting hospital admission at emergency department triage using machine learning" by Hong et al and "Data from: Prognosis of chronic kidney disease with normal-range proteinuria: The CKD-ROUTE study" by Iimori et al.

MIMIC-IV [1] (Medical Information Mart for Intensive Care, Version IV) is an extensive database comprising de-identified health-related data associated with over 180,000 thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center in Boston, Massachusetts. It is an update and expansion of the earlier MIMIC-III database and includes data such as vital signs, medications, laboratory measurements, observations and notes taken by care providers, fluid balance, procedure codes, diagnostic codes, imaging reports, hospital length of stay, survival data, and more.

The study published by Hong et al. [2] utilizes extensive patient data and ML models to predict hospital admissions, focusing on the relevance of various patient features including demographic information, emergency department triage variables, and medical history. In contrast to MIMIC-IV, this dataset is weighted more towards emergency department triage features prior to hospital admission.

The study "Prognosis of chronic kidney disease with normal-range proteinuria: The CKD-ROUTE study" pusblished by Iimori et al. [3] explores the outcomes of chronic kidney disease (CKD) in patients who have normal-range proteinuria. It involved a prospective cohort study of 1138 CKD patients, analyzing their risk of CKD progression, cardiovascular events, and mortality.

## Data Sources

### Subdirectory /mimic_iv_extract

    df_diseases_frequency.xlsx

An excel file providing an overview of the disease and hospitalisation frequencies recorded in the MIMIC-IV database. It specifically focuses on CKD comorbidities including hypertension, anemia, diabetes, cardiac disarrhythmias, ischemic heart disease, thyroid disease, heart failure, cerebrovascular disease, and PVD.

    df_labs_frequency.xlsx

An excel file containing a summary of the both the total number of unique patients and hospitalisations recorded in the MIMIC-IV database for lab events of interest with respect to CKD.

    ckd_stage_frequency.xlsx

An excel file containing a summary of the both the total number of unique patients and hospitalisations recorded in the MIMIC-IV database for all CKD stage diagnoses (i.e. Chronic kidney disease, Stage I; Chronic kidney disease, Stage II (mild); Chronic kidney disease, Stage III (moderate); Chronic kidney disease, Stage IV (severe); Chronic kidney disease, Stage V; End stage renal disease; Chronic kidney disease, unspecified).
    
    df_ckd_lab_summary.xlsx

An excel file providing a summary of CKD stage progression a measured by lab events in the recorded in the MIMIC-IV database. The frame contain the first and last recorded eGFR values and CKD stage diagnoses for each unique patient with applicable values recorded in MIMIC-IV.

    df_all_patients.pkl

A pickle file containing a Pandas dataframe of all patients in the MIMIC-IV database, and basic identifiers including gender anchor information.

    df_admissions.pkl

A pickle file containing a Pandas dataframe of all admissions to the Beth Israel Deaconess Medical Center recorded in the MIMIC-IV database. Admission timeframes, associated locations, insurance details, and some further demographics such as race and marital status are included.

    df_past_medical_history.pkl

A pickle file containing a Pandas dataframe of patient past medical history including previous diagnoses and notes by healthcare providers, as well as associated chart times for the records.

    df_icd_codes_with_description.pkl

A pickle file containing a Pandas dataframe of the ICD-9 codes and long-title diagnoses associated with all admissions to the Beth Israel Deaconess Medical Center recorded in the MIMIC-IV database.

    df_lab_items.pkl

A pickle file containing a Pandas dataframe of all lab items recorded in the MIMIC-IV database, including the lab item ID's, labels, categories, and associated lab fluid (e.g. Cerebrospinal Fluid).

    df_lab_events.pkl

A pickle file containing a Pandas dataframe of all charted lab events that can be of interest to CKD patients recorded in the MIMIC-IV database, including the patient subject ID, HADM ID, lab item ID, charttime, and recorded lab result values.

    df_ckd_lab_items.pkl

A pickle file containing a Pandas dataframe of lab items recorded in the MIMIC-IV database which are specifically related to CKD, including the lab item ID's, labels, categories, and associated lab fluid (e.g. Cerebrospinal Fluid).

    df_height_weight.pkl

A pickle file containing a Pandas dataframe with first, minimum, maximum, and mean measurements for patient height and weight associated with unique hospital admissions in the MIMIC-IV database.

### Subdirectory /hong_et_al

    5v_cleandf.RData

This is the raw dataset as provided by Hong et al. It includes a total of 560,486 patient entries, each with 972 features including demographics, ED triage variables, and detailed patient medical histories, including medication. This dataset is provided in RData format for compatibility with R statistical software.

    df.csv

This is a CSV file conversion of the 5v_cleandf.RData file. It maintains the integrity of the original data, including the entirety of the original set.

    df_updt.csv

This is an updated version of df.csv after processing by eGFR.ipynb. It includes the newly calculated eGFR and CKD stage columns.

### Subdirectory /iimori_et_al

    ROUTE_proteinuria_dataset.xlsx

This is the raw dataset as provided by Iimori et al. It includes a total of 1138 CKD patients, each with 50 features analyzing their risk of CKD progression, cardiovascular events, and mortality.

## Data Preprocessing

    mimic_iv_extract.ipynb

A Jupyter Notebook that first establishes a connection to the MIMIC-IV database via BigQuery, and subsequently extracts a series of relevant dataframes. The datasets are saved as pickle files in the /mimic_ic_extracts subdirectory. Each extracted dataframe is comprehensively detailed within the notebook, and in the README.md under the /mimic_iv_extract subdirectory section.

    mimic_iv_ovrvw.ipynb

A Jupyter Notebook providing an overview of the disease and hospitalisation frequencies recorded in the MIMIC-IV database, leveraging data from the df_all_patients.pkl and df_icd_codes_with_description.pkl pickle files. The disease, labe event, and hospitalisation frequency information is summarized in Pandas dataframes, and saved as external Excel files in the /mimic_ic_extracts subdirectory.

    mimic_iv_ckd_ovrvw.ipynb

A Jupyter Notebook providing an overview of the patient and hospitalisation frequencies for each CKD ICD-9 code recorded in the MIMIC-IV database, using the df_icd_codes_with_description.pkl pickle file. The patient and hospitalisation frequency information is summarized in a Pandas dataframe, and saved as the Excel file,  mimic_iv_ckd_ovrvw.xlsx, in the /mimic_ic_extracts subdirectory.

    eGFR.ipynb

A Jupyter Notebook that details the process of calculating the Estimated Glomerular Filtration Rate (eGFR) for all patients in a designated dataset. The eGFR calculation is based on the CKD-EPI Creatinine Equation [4]. Following the calculation, this notebook also labels each patient with a Chronic Kidney Disease (CKD) stage ranging from 1 to 5 based on their eGFR values [5]. Additionally, the notebook provides a summary of statistical results derived from the eGFR and CKD stage calculations. The eGFR.ipynb notebook is used to calculate the eGFR and assign CKD stage labels for all three data sources.

## Data Analysis & Predictive Modeling-0

    ANOVA.ipynb

A Jupyter Notebook used to perform ANOVA analysis on various data frames to determine the statistical significance of the relationships between features and a specified target column. The data is imported from respective sources, preprocessed, and passed into a helper function which facilitates ANOVA analysis via the statsmodels library. The evaluated features' calculated p-values are then displayed in increasing order.

    XG_boost.ipynb

A Jupyter Notebook detailing the process of developing XGBoost classification and regression models from multiple datasets, identifying features most relevant to CKD progression. The data is preprocessed and partitioned into training and test data, then an XGBRegressor and XGBClassifier are tuned with Bayesian Hyperparameter Optimization, trained, and used to evaluate the feature importance. The model's feature importance rankings are displayed and visualized using Matplotlib and Seaborn.

    Simple_NN.ipynb

A Jupyter Notebook detailing the process of developing a simple neural network, identifying features most relevant to eGFR prediction. The data is preprocessed and partitioned into training and test data, dataloaders are prepared, then a simple neural network structure is defined, trained, and used to evaluate the feature importance. The model's feature importance is demonstrated via a SHAP explainer, and the summary plot is displayed.

    ResNet_NN.ipynb

A Jupyter Notebook detailing the process of developing a residual neural network, identifying features most relevant to eGFR prediction. The data is preprocessed and partitioned into training and test data, dataloaders are prepared, then a residual neural network structure is defined, trained, and used to evaluate the feature importance. The model's feature importance is demonstrated via a SHAP explainer, and the summary plot is displayed.

    CPH.ipynb

A Jupyter Notebook detailing the process of training a Cox Proportional Hazards (CPH) model to predict the survival probability of CKD stage progression. The workflow follows importing data from respective sources, preprocessing and partitioning into training and test data, and then training the CPH model to estimate patient survival probabilities. The CPH feature importance summary is are displayed, and survival curves are visualized using Matplotlib.



# How to Use

To work with the RData file, it is necessary to have R statistical software installed on your machine. The dataset can then be loaded into your R environment using the load() function.

The CSV and Excel files can be viewed and manipulated using standard data processing tools such as Microsoft Excel, Python (pandas library), or R.

To execute analyses in the Jupyter Notebooks, it is necessary to have Jupyter installed, along with Python and the necessary libraries (pandas, numpy, os, google-cloud-bigquery, xgboost, statsmodels, seaborn, matplotlib, sklearn). In order to execute the Bayesian hyperparameter optimization and model training within XG_boost.ipynb, it is necessary to have access to a GPU device. With the prerequisites installed, the Jupyter Notebooks can be launched from the project directory, and all other scripts can be executed.

# Contributors

Vaibhav Mahajan, (St Catherine's College, University of Oxford)

# References

[1] https://physionet.org/content/mimiciv/2.2/

[2] https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201016

[3] https://datadryad.org/stash/dataset/doi:10.5061/dryad.kq23s

[4] https://www.kidney.org/content/ckd-epi-creatinine-equation-2021

[5] https://www.nhs.uk/conditions/kidney-disease/diagnosis/
 
# Supplementary Resources

[6] https://www.nature.com/articles/s41598-023-36214-0

[7] https://bmcnephrol.biomedcentral.com/articles/10.1186/s12882-021-02496-7#Abs1