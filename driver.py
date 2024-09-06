import pandas as pd
import numpy as np
import pickle
import shap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from ModelWrappers.XGboostWrapper import XGBoostWrapper
from ModelWrappers.CoxPHWrapper import CoxPHWrapper

class Driver:
    def __init__(self, cfg, master, base, seed=42, preprocess=True):
        self.cfg = cfg
        self.master = self.preprocess(master) if preprocess else master
        self.base = base
        self.seed = seed

    # ------------------------------------------------      DATA HANDLING      ------------------------------------------------ #
    
    def preprocess(self, data):
        # Drop unnecessary columns
        drop_cols = [
            "anchor_year_group",
            "admittime_first",
            "dischtime_last",
            "hadm_id_last_CKD",
            "admittime_CKD",
            "dischtime_CKD", 
            'deathtime',
            "first_stage_icd",
            "Creatinine_first_time",
            "Creatinine_last_time",
            "Creatinine_min_time",
            "Creatinine_max_time",
            "Creatinine_mean_time",
            "Creatinine_median_time"
        ]

        data.drop(columns=drop_cols, axis=1, inplace=True)
        data = data.drop(columns=data.filter(like="CKD_stage_last").columns)
        data = data.drop(columns=data.filter(like="last_stage_icd").columns)
        data = data.drop(columns=data.filter(like="last_long_title").columns)
        data = data.drop(columns=data.filter(like='Chronic kidney disease').columns)
        data = data.drop(columns=data.filter(like='chronic kidney disease').columns)
        data = data.drop(columns=data.filter(like='End stage renal').columns)

        # Convert Int64 columns to int64 for tensor compatibility
        int64_columns = data.select_dtypes(include=['Int64']).columns
        data[int64_columns] = data[int64_columns].astype('int64')

        return data
    
    def baseAugmentation(self, novel_features):
        # Select these columns from df_source
        novel_features = ['subject_id'] + novel_features.tolist()
        df_novel = self.master[novel_features]

        df_augment = pd.merge(self.base, df_novel, on='subject_id', how='outer')

        return df_augment
    
    # def split_Vanilla(self, file_path, target_column):
        
    
    def split_ValSets(self):
        # Exclude response variable from features frame
        X = self.master.drop(self.cfg['response'], axis=1)

        # Set response variable 
        y = self.master[self.cfg['response']]
        
        np.random.seed(self.seed)  # Ensure reproducibility
            
        # Split data into test and the remaining data
        X_traindev, X_test, y_traindev, y_test = train_test_split(X, y, test_size=self.cfg['test_size'], random_state=self.seed)
        
        # Further split the remaining data into multiple train and validation sets
        val_sets = []
        for _ in range(self.cfg['n_valsets']):
            # Randomly select validation set from the remaining data
            X_train, X_val, y_train, y_val = train_test_split(X_traindev, y_traindev, test_size=self.cfg['val_size'], random_state=np.random.randint(10000))
            
            val_sets.append({
                'X_train': X_train, 
                'y_train': y_train, 
                'X_val': X_val, 
                'y_val': y_val
            })

        return X_traindev, y_traindev, X_test, y_test, val_sets
    
    # --------------------------------------------      SHAP FEATURE SELECTION      -------------------------------------------- #  
    
    def SHAP_TreeExplainer(self, model, X_test):
        # Create a SHAP TreeExplainer
        explainer = shap.TreeExplainer(model)

        # Compute SHAP values for a set of data
        shap_values = explainer.shap_values(X_test)
        
        return shap_values

    # def SHAP_DeepExplainer(self, cfg):

    def SHAP_FeatureSelect(self, SHAP_vals, X_traindev):
        SHAP_feature_names = X_traindev.columns.tolist()  # Full set of feature names
        KFRE_feature_names = set(self.base.columns.tolist())  # KFRE feature names as a set for faster lookup

        # Create a mask to keep only specified features
        mask = [feat not in KFRE_feature_names for feat in SHAP_feature_names]

        # Apply the mask to shap_values and SHAP_feature_names
        filtered_shap_values = SHAP_vals[:, mask]
        filtered_feature_names = [feat for feat, keep in zip(SHAP_feature_names, mask) if keep]

        # Calculate the mean absolute SHAP value for each feature
        mean_abs_shap_values = np.mean(np.abs(filtered_shap_values), axis=0)

        # Sort SHAP values and feature names by mean absolute SHAP value
        sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
        sorted_shap_values = mean_abs_shap_values[sorted_indices]
        sorted_feature_names = np.array(filtered_feature_names)[sorted_indices]

        # Extract top features
        NovelFeatures = sorted_feature_names[:self.cfg['n_novel']]
        vals = sorted_shap_values[:self.cfg['n_novel']]

        # Save the list to a pickle file
        with open(f"features/{self.cfg['tag']}_Features.pkl", 'wb') as f:
            pickle.dump(list(NovelFeatures), f)

        return NovelFeatures, vals
    
    # --------------------------------------------------      SHAP PLOTS      -------------------------------------------------- #

    def plot_SHAPma(self, features, vals):
        # Create a horizontal bar plot
        plt.figure(figsize=(6, 8))
        plt.barh(features[::-1], vals[::-1], color='skyblue')
        plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')

        # Save the plot as a PNG file
        plt.savefig(f"figs/{self.cfg['tag']}_SHAPma.png", format='png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_SHAPbeeswarm(self, features, vals, X_test):
        # Create a mask to select only the features in novel features
        feature_mask = np.isin(X_test.columns, features)

        # Filter the SHAP values and feature names to keep only the selected features
        filtered_shap_values = vals[:, feature_mask]
        filtered_feature_names = np.array(X_test.columns)[feature_mask]

        # Filter the X_test DataFrame to only include the selected features
        X_test_filtered = X_test.loc[:, filtered_feature_names]

        plt.figure(figsize=(9, 8))

        # Create the summary plot with the selected features
        shap.summary_plot(
            filtered_shap_values, 
            X_test_filtered,
            feature_names=filtered_feature_names, 
            plot_size=(10, 10), 
            max_display=40,
            show=False
        )

        # Customize plot appearance
        plt.xlabel('SHAP value (impact on model output)')

        # Save the plot as a PNG file
        plt.savefig(f"figs/{self.cfg['tag']}_Beeswarm.png", format='png', dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()

    # --------------------------------------------------      PIPELINES      -------------------------------------------------- #   

    def XGBoostPipe(self):
        # Split data into training, testing, and validation sets
        X_traindev, y_traindev, X_test, y_test, val_sets = self.split_ValSets()
        
        # Initialize XGBoostWrapper and perform Bayesian hyperparameter optimization
        XGboostWrapper = XGBoostWrapper(self.cfg)
        XGboostWrapper.BayesianHyperparameterOptimizer(val_sets, self.seed)
        XGboostWrapper.Trainer(X_traindev, y_traindev)
        XGboostWrapper.Evaluator(X_test, y_test)

        # Compute SHAP values and select features
        SHAP_vals = self.SHAP_TreeExplainer(XGboostWrapper.model, X_test)
        XGBoostFeatures, vals = self.SHAP_FeatureSelect(SHAP_vals, X_traindev)
        self.plot_SHAPma(XGBoostFeatures, vals)
        self.plot_SHAPbeeswarm(XGBoostFeatures, SHAP_vals, X_test)
        
        base_augment = self.baseAugmentation(XGBoostFeatures)
        
        CoxPH = CoxPHWrapper(self.cfg, base_augment, self.seed)
        CoxPH.Summary()
        CoxPH.FeatureRank()
        CoxPH.SchoenfeldTest()
        CoxPH.plot_BrierScore()
        CoxPH.plot_DynamicAUC()

        # Iterate through CKD stages 1 to 5 and plot the survival curves for each stage
        for stage in range(2, 6):
            CoxPH.plot_SurvivalCurves(stage)
