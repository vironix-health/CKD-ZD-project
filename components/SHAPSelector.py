import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm


class SHAPSelector:
    def __init__(self, cfg, master_features, base, model, train, test, train_tensor=None, test_tensor=None):
        self.cfg = cfg
        self.master_features = master_features
        self.base = base
        self.model = model
        self.train = train
        self.test = test
        self.train_tensor = train_tensor
        self.test_tensor = test_tensor
        self.SHAP_vals = None
        self.NovelFeatures = None
        self._filtered_feature_names = None
        self._filtered_shap_values = None

        # Define device to enable SHAP DeepExplainer GPU acceleration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def SHAP_KernelExplainer(self):
        # Create a SHAP KernelExplainer
        explainer = shap.KernelExplainer(self.model.predict_proba, self.train)

        # Compute SHAP values for a set of data
        shap_values = explainer.shap_values(self.test)

        # Store SHAP values
        self.SHAP_vals = np.array(shap_values)

    def SHAP_DeepExplainer(self):
        # Select a subset of the training data to use as the background dataset for SHAP
        background = self.train_tensor[:100].to(self.device)

        # Initialize the SHAP DeepExplainer with the model and background dataset
        explainer = shap.DeepExplainer(self.model, background)

        # Define the list to store SHAP values
        shap_list = []

        # Convert the tensor to numpy for slicing
        X_test_np = self.test_tensor.cpu().numpy()

        # Create indices for batching, ensuring all samples are covered
        num_batches = int(np.ceil(X_test_np.shape[0] / self.cfg['batch_size']))
        idxs = np.array_split(np.arange(X_test_np.shape[0]), num_batches)

        # Use tqdm to display a progress bar for SHAP value computation
        for idx in tqdm(idxs, total=len(idxs)):
            # Compute SHAP values for the current batch
            shaps = explainer.shap_values(torch.from_numpy(X_test_np[idx]).to(self.device), check_additivity=False)
            
            if isinstance(shaps, list):
                shap_list.append(shaps[0])
            else:
                shap_list.append(shaps)

        # Concatenate the list of SHAP values into a single array
        shap_values = np.concatenate(shap_list, axis=0)

        # Store SHAP values
        self.SHAP_vals = shap_values

    # --------------------------------------------      SHAP FEATURE SELECTION      -------------------------------------------- #  

    def SHAP_FeatureFilter(self):
        # Build appropriate SHAP explainer based on the model type
        if self.cfg['tag'] in ['xgb', 'dct', 'rdmf']:
            self.SHAP_TreeExplainer()
        elif self.cfg['tag'] in ['lr']:
            self.SHAP_LinearExplainer()
        elif self.cfg['tag'] in ['knn', 'lsvm']:
            self.SHAP_KernelExplainer()
        elif self.cfg['tag'] in ['fcnn', 'rsnt']:
            self.SHAP_DeepExplainer()

        SHAP_feature_names = self.master_features  # Full set of feature names
        KFRE_feature_names = set(self.base.columns.tolist())  # KFRE feature names as a set for faster lookup

        # Handle SHAP output for tree based models
        if len(self.SHAP_vals.shape) == 3 and self.SHAP_vals.shape[2] == 2:
            self.SHAP_vals = self.SHAP_vals[:, :, 1]  # Select the SHAP values for the positive class

        if self.cfg['tag'] in ['fcnn', 'rsnt']:
            # Get the indices of the features to remove
            KFRE_feat_indices = [SHAP_feature_names.index(feat) for feat in KFRE_feature_names if feat in SHAP_feature_names]

            # Create a mask to keep only specified features
            mask = np.ones(len(SHAP_feature_names), dtype=bool)
            mask[KFRE_feat_indices] = False

            # Apply the mask to shap_values and SHAP_feature_names
            filtered_shap_values = self.SHAP_vals[:, mask].reshape(self.SHAP_vals.shape[0], -1)  # Ensure 2D shape
            filtered_feature_names = [SHAP_feature_names[i] for i in range(len(SHAP_feature_names)) if mask[i]]
        else:
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
        with open(f"novel_predictors/{self.cfg['tag']}_Features.pkl", 'wb') as f:
            pickle.dump(list(NovelFeatures), f)

        self.NovelFeatures = (NovelFeatures, vals)
        self._filtered_feature_names = filtered_feature_names
        self._filtered_shap_values = filtered_shap_values

    def get_NovelFeatures(self):
        return self.NovelFeatures[0]
        
    # --------------------------------------------------      SHAP PLOTS      -------------------------------------------------- #

    def plot_SHAPma(self):
        # Create a horizontal bar plot
        plt.figure(figsize=(6, 8))
        plt.barh(self.NovelFeatures[0][::-1], self.NovelFeatures[1][::-1], color='skyblue')
        plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')

        # Save the plot as a PNG file
        plt.savefig(f"figs/{self.cfg['name']}/{self.cfg['tag']}_SHAPma.png", format='png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_ClassicalSHAPbeeswarm(self):
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
        plt.savefig(f"figs/{self.cfg['name']}/{self.cfg['tag']}_Beeswarm.png", format='png', dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()

    def plot_DeepSHAPbeeswarm(self):
        # Ensure X_test is a DataFrame if it was converted to numpy array
        if isinstance(self.test, np.ndarray):
            self.test = pd.DataFrame(self.test, columns=self.master_features)

        # Create a mask to select only the top 40 features from filtered_shap_values
        top_40_mask = [name in self.NovelFeatures[0] for name in self._filtered_feature_names]

        # Filter the SHAP values to keep only the top 40 features
        top_40_shap_values = self._filtered_shap_values[:, top_40_mask]

        # Order features by mean absolute SHAP value in descending order
        mean_abs_shap_values = np.mean(np.abs(top_40_shap_values), axis=0)
        sorted_indices = np.argsort(mean_abs_shap_values)[::-1]

        # Reorder SHAP values and feature names
        top_40_shap_values_sorted = top_40_shap_values[:, sorted_indices]
        top_40_feature_names_sorted = np.array(self._filtered_feature_names)[top_40_mask][sorted_indices]

        # Filter the X_test DataFrame to only include the top 40 features
        X_test_top_40 = self.test[top_40_feature_names_sorted]

        plt.figure(figsize=(10, 12))

        # Create the summary plot with the top 40 features
        shap.summary_plot(
            top_40_shap_values_sorted, 
            X_test_top_40,  # Use only the top 40 features from X_test
            feature_names=top_40_feature_names_sorted, 
            plot_size=(10, 12), 
            max_display=40,  # Limit to top 40 features
            show=False  # Disable immediate display to customize further
        )

        # Access the current figure and axis
        fig = plt.gcf()
        ax = plt.gca()

        # Set x-axis limits to display SHAP values between -7.5 and 7.5
        ax.set_xlim(-7.5, 7.5)

        # Customize plot appearance
        plt.xlabel('SHAP value (impact on model output)')
        # plt.ylabel('Features')

        # Save the plot as a PNG file
        plt.savefig(f"figs/{self.cfg['name']}/{self.cfg['tag']}_Beeswarm.png", format='png', dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()