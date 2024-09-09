import shap
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

class SHAPSelector:
    def __init__(self, cfg, master_features, base, model, train, test, train_tensor=None, test_tensor=None):
        """
        Initialize the SHAPSelector class with configuration, features, model, and data.

        Parameters:
        - cfg: Configuration dictionary
        - master_features: List of all feature names
        - base: Base dataset
        - model: Trained model
        - train: Training dataset
        - test: Testing dataset
        - train_tensor: Training dataset as tensor (optional)
        - test_tensor: Testing dataset as tensor (optional)
        """
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
        """
        Compute SHAP values using TreeExplainer for tree-based models.
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.test)
        self.SHAP_vals = shap_values

    def SHAP_LinearExplainer(self):
        """
        Compute SHAP values using LinearExplainer for linear models.
        """
        explainer = shap.LinearExplainer(self.model, self.train)
        shap_values = explainer.shap_values(self.test)
        self.SHAP_vals = np.array(shap_values)

    def SHAP_KernelExplainer(self):
        """
        Compute SHAP values using KernelExplainer for kernel-based models.
        """
        explainer = shap.KernelExplainer(self.model.predict_proba, self.train)
        shap_values = explainer.shap_values(self.test)
        self.SHAP_vals = np.array(shap_values)

    def SHAP_DeepExplainer(self):
        """
        Compute SHAP values using DeepExplainer for deep learning models.
        """
        background = self.train_tensor[:100].to(self.device)
        explainer = shap.DeepExplainer(self.model, background)
        shap_list = []

        X_test_np = self.test_tensor.cpu().numpy()
        num_batches = int(np.ceil(X_test_np.shape[0] / self.cfg['batch_size']))
        idxs = np.array_split(np.arange(X_test_np.shape[0]), num_batches)

        for idx in tqdm(idxs, total=len(idxs)):
            shaps = explainer.shap_values(torch.from_numpy(X_test_np[idx]).to(self.device), check_additivity=False)
            if isinstance(shaps, list):
                shap_list.append(shaps[0])
            else:
                shap_list.append(shaps)

        shap_values = np.concatenate(shap_list, axis=0)
        self.SHAP_vals = shap_values

    # --------------------------------------------      SHAP FEATURE SELECTION      -------------------------------------------- #  

    def SHAP_FeatureFilter(self):
        """
        Select novel features based on SHAP values and save them to a pickle file.
        """
        # Build appropriate SHAP explainer based on the model type
        if self.cfg['tag'] in ['xgb', 'dct', 'rdmf']:
            self.SHAP_TreeExplainer()
        elif self.cfg['tag'] in ['lr']:
            self.SHAP_LinearExplainer()
        elif self.cfg['tag'] in ['knn', 'lsvm', 'ksvm']:
            self.SHAP_KernelExplainer()
        elif self.cfg['tag'] in ['fcnn', 'rsnt']:
            self.SHAP_DeepExplainer()

        SHAP_feature_names = self.master_features
        KFRE_feature_names = set(self.base.columns.tolist())

        # Handle SHAP output for tree-based models
        if len(self.SHAP_vals.shape) == 3 and self.SHAP_vals.shape[2] == 2:
            self.SHAP_vals = self.SHAP_vals[:, :, 1]

        if self.cfg['tag'] in ['fcnn', 'rsnt']:
            KFRE_feat_indices = [SHAP_feature_names.index(feat) for feat in KFRE_feature_names if feat in SHAP_feature_names]
            mask = np.ones(len(SHAP_feature_names), dtype=bool)
            mask[KFRE_feat_indices] = False
            filtered_shap_values = self.SHAP_vals[:, mask].reshape(self.SHAP_vals.shape[0], -1)
            filtered_feature_names = [SHAP_feature_names[i] for i in range(len(SHAP_feature_names)) if mask[i]]
        else:
            mask = [feat not in KFRE_feature_names for feat in SHAP_feature_names]
            filtered_shap_values = self.SHAP_vals[:, mask]
            filtered_feature_names = [feat for feat, keep in zip(SHAP_feature_names, mask) if keep]

        mean_abs_shap_values = np.mean(np.abs(filtered_shap_values), axis=0)
        sorted_indices = np.argsort(mean_abs_shap_values)[::-1]
        sorted_shap_values = mean_abs_shap_values[sorted_indices]
        sorted_feature_names = np.array(filtered_feature_names)[sorted_indices]

        NovelFeatures = sorted_feature_names[:self.cfg['n_novel']]
        vals = sorted_shap_values[:self.cfg['n_novel']]
        with open(f"novel_predictors/{self.cfg['tag']}_Features.pkl", 'wb') as f:
            pickle.dump(list(NovelFeatures), f)
        self.NovelFeatures = (NovelFeatures, vals)
        self._filtered_feature_names = filtered_feature_names
        self._filtered_shap_values = filtered_shap_values

    def get_NovelFeatures(self):
        """
        Get the list of novel features.
        """
        return self.NovelFeatures[0]
        
    # --------------------------------------------------      SHAP PLOTS      -------------------------------------------------- #

    def plot_SHAPma(self):
        """
        Plot the mean absolute SHAP values for the novel features.
        """
        plt.figure(figsize=(6, 8))
        plt.barh(list(map(str, self.NovelFeatures[0][::-1])), self.NovelFeatures[1][::-1], color='skyblue')
        plt.xlabel('mean(|SHAP value|) (average impact on model output magnitude)')
        plt.savefig(f"figs/{self.cfg['name']}/{self.cfg['tag']}_SHAPma.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_ClassicalSHAPbeeswarm(self):
        """
        Plot a SHAP beeswarm plot for classical models.
        """
        feature_mask = np.isin(self.master_features, self.NovelFeatures[0])
        filtered_shap_values = self.SHAP_vals[:, feature_mask]
        filtered_feature_names = np.array(self.master_features)[feature_mask]

        if isinstance(self.test, pd.DataFrame):
            X_test_filtered = self.test.loc[:, filtered_feature_names]
        else:
            feature_indices = [self.master_features.index(feat) for feat in filtered_feature_names]
            X_test_filtered = self.test[:, feature_indices]

        plt.figure(figsize=(9, 8))
        shap.summary_plot(
            filtered_shap_values, 
            X_test_filtered,
            feature_names=filtered_feature_names, 
            plot_size=(10, 10), 
            max_display=40,
            show=False
        )
        plt.xlabel('SHAP value (impact on model output)')
        plt.savefig(f"figs/{self.cfg['name']}/{self.cfg['tag']}_Beeswarm.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_DeepSHAPbeeswarm(self):
        """
        Plot a SHAP beeswarm plot for deep learning models.
        """
        if isinstance(self.test, np.ndarray):
            self.test = pd.DataFrame(self.test, columns=self.master_features)

        top_40_mask = [name in self.NovelFeatures[0] for name in self._filtered_feature_names]
        top_40_shap_values = self._filtered_shap_values[:, top_40_mask]

        mean_abs_shap_values = np.mean(np.abs(top_40_shap_values), axis=0)
        sorted_indices = np.argsort(mean_abs_shap_values)[::-1]

        top_40_shap_values_sorted = top_40_shap_values[:, sorted_indices]
        top_40_feature_names_sorted = np.array(self._filtered_feature_names)[top_40_mask][sorted_indices]

        X_test_top_40 = self.test[top_40_feature_names_sorted]

        plt.figure(figsize=(10, 12))
        shap.summary_plot(
            top_40_shap_values_sorted, 
            X_test_top_40,  
            feature_names=top_40_feature_names_sorted, 
            plot_size=(10, 12), 
            max_display=40,  
            show=False  
        )

        fig = plt.gcf()
        ax = plt.gca()
        ax.set_xlim(-7.5, 7.5)
        plt.xlabel('SHAP value (impact on model output)')
        plt.savefig(f"figs/{self.cfg['name']}/{self.cfg['tag']}_Beeswarm.png", format='png', dpi=300, bbox_inches='tight')
        plt.show()