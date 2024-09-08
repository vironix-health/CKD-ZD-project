import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SHAPSelector:
    def __init__(self, cfg, master_features, base, model, train, test):
        self.cfg = cfg
        self.master_features = master_features
        self.base = base
        self.model = model
        self.train = train
        self.test = test
        self.SHAP_vals = None
        self.NovelFeatures = None

    # ------------------------------------------------      SHAP EXPLAINERS      ------------------------------------------------ #  
    
    def SHAP_TreeExplainer(self):
        # Create a SHAP TreeExplainer
        explainer = shap.TreeExplainer(self.model)

        # Compute SHAP values for a set of data
        shap_values = explainer.shap_values(self.test)

        # Store SHAP values
        self.SHAP_vals = shap_values

    def SHAP_LinearExplainer(self):
        # Create a SHAP LinearExplainer
        explainer = shap.LinearExplainer(self.model, self.train)

        # Compute SHAP values for a set of data
        shap_values = explainer.shap_values(self.test)

        # Store SHAP values
        self.SHAP_vals = np.array(shap_values)

    # def SHAP_DeepExplainer(self, cfg):

    # --------------------------------------------      SHAP FEATURE SELECTION      -------------------------------------------- #  

    def SHAP_FeatureSelect(self):
        # Build appropriate SHAP explainer based on the model type
        if self.cfg['tag'] in ['xgb', 'dct', 'rdmf']:
            self.SHAP_TreeExplainer()
        elif self.cfg['tag'] in ['lr']:
            self.SHAP_LinearExplainer()

        SHAP_feature_names = self.master_features  # Full set of feature names
        KFRE_feature_names = set(self.base.columns.tolist())  # KFRE feature names as a set for faster lookup

        # Handle SHAP output for tree based models
        if len(self.SHAP_vals.shape) == 3 and self.SHAP_vals.shape[2] == 2:
            self.SHAP_vals = self.SHAP_vals[:, :, 1]  # Select the SHAP values for the positive class

        # Create a mask to keep only specified features
        mask = [feat not in KFRE_feature_names for feat in SHAP_feature_names]

        # Apply the mask to shap_values and SHAP_feature_names
        filtered_shap_values = self.SHAP_vals[:, mask]
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

        self.NovelFeatures = (NovelFeatures, vals)

    def get_NovelFeatures(self):
        return self.NovelFeatures[0]
        
    # --------------------------------------------------      SHAP PLOTS      -------------------------------------------------- #

    def plot_SHAPma(self):
        # Create a horizontal bar plot
        plt.figure(figsize=(6, 8))
        plt.barh(self.NovelFeatures[0][::-1], self.NovelFeatures[1][::-1], color='skyblue')
        plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')

        # Save the plot as a PNG file
        plt.savefig(f"figs/{self.cfg['tag']}_SHAPma.png", format='png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_SHAPbeeswarm(self):
        # Create a mask to select only the features in novel features
        feature_mask = np.isin(self.master_features, self.NovelFeatures[0])

        # Filter the SHAP values and feature names to keep only the selected features
        filtered_shap_values = self.SHAP_vals[:, feature_mask]
        filtered_feature_names = np.array(self.master_features)[feature_mask]

        # Check if self.test is a DataFrame or numpy array
        if isinstance(self.test, pd.DataFrame):
            X_test_filtered = self.test.loc[:, filtered_feature_names]
        else:
            feature_indices = [self.master_features.index(feat) for feat in filtered_feature_names]
            X_test_filtered = self.test[:, feature_indices]

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