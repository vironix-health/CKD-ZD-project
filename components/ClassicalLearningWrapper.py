from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from skopt.utils import use_named_args
from skopt import gp_minimize
from copy import deepcopy
import warnings
import statistics
import numpy as np
import pickle


# --------------------------------------------------------------      WRAPPER CLASS      -------------------------------------------------------------- #

class ClassicalLearningWrapper:
    def __init__(self, cfg):
        """
        Initialize the ClassicalLearningWrapper class with configuration settings.

        Parameters:
        - cfg: Configuration dictionary containing various settings.
        """
        # Initialize classification models
        self.model_tags = {
            'xgb': XGBClassifier(objective='binary:logistic', verbosity=0, tree_method='gpu_hist'),
            'dct': DecisionTreeClassifier(),
            'rdmf': RandomForestClassifier(),
            'lr': LogisticRegression(max_iter=1000),
        }
        self.model = self.model_tags[cfg['tag']]
        self.best_auc_params = None
        self.best_auc_score = None
        self.X_traindev = None
        self.X_test = None
        self.cfg = cfg

    # -----------------------------------------------------      BAYESIAN OPTIMIZATION      ----------------------------------------------------- #

    def AUC_validate(self, params, val_sets):
        """
        Validate the model using AUC score across multiple validation sets.

        Parameters:
        - params: Dictionary of hyperparameters to set for the model.
        - val_sets: List of validation sets, each containing 'X_train', 'y_train', 'X_val', and 'y_val'.

        Returns:
        - Mean AUC score across all validation sets.
        """
        auc_scores = []
        for val in val_sets:
            self.model.set_params(**params)
            self.model.fit(val['X_train'], val['y_train'])
            try:
                preds_proba = self.model.predict_proba(val['X_val'])[:, 1]
                auc = roc_auc_score(val['y_val'], preds_proba)
            except Exception as e:
                print(f"Error during model prediction: {str(e)}")
                auc = 0.0
            auc_scores.append(auc)
        return np.mean(auc_scores)

    def BayesianHyperparameterOptimizer(self, data):
        """
        Perform Bayesian hyperparameter optimization to find the best parameters for the model.

        Parameters:
        - val_sets: List of validation sets, each containing 'X_train', 'y_train', 'X_val', and 'y_val'.
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignore FutureWarning

        best_auc_params = []
        best_auc_scores = []

        for i, d in enumerate(data):
            # Define objective function for Bayesian optimization
            @use_named_args(self.cfg['auc_space'])
            def AUC_objective(**params):
                print("Testing params:", params)  # Print current test parameters to console
                auc = self.AUC_validate(params, d['val_sets'])
                print("AUC for params:", auc)  # Print AUC for parameters console
                return -auc  # Invert to optimize for minimum AUC

        
            # Perform Bayesian Optimization
            result = gp_minimize(AUC_objective, self.cfg['auc_space'], n_calls=self.cfg['n_bayesian'], random_state=self.cfg['seeds'][i])

            # Extract the best parameters and the corresponding score
            best_auc_params.append({dimension.name: result.x[i] for i, dimension in enumerate(self.cfg['auc_space'])})
            best_auc_scores.append(-result.fun)

        best_index = best_auc_scores.index(max(best_auc_scores))
        self.best_auc_score = best_auc_scores[best_index]
        self.best_auc_params = best_auc_params[best_index]

        print("Best parameters found: ", self.best_auc_params)
        print("Best mean AUC across validation sets: ", self.best_auc_score)

    # -------------------------------------------------------------      EVALUATOR      ------------------------------------------------------------- #

    def Evaluator(self, data):
        """
        Train the model using the best hyperparameters found during optimization.

        Parameters:
        - X_traindev: Training data features.
        - y_traindev: Training data labels.
        """
        best_model = deepcopy(self.model)
        best_model.set_params(**self.best_auc_params)

        auc_scores = []
        max_auc = 0
        for d in data:
            best_model.fit(d['X_traindev'], d['y_traindev'])
            y_pred = best_model.predict(d['X_test'])
            auc = roc_auc_score(d['y_test'], y_pred)
            auc_scores.append(auc)

            new_max = max(auc_scores)
            if new_max > max_auc:
                max_auc = new_max
                self.model = deepcopy(best_model)
                self.X_traindev = d['X_traindev']
                self.X_test = d['X_test']

        mean_auc = np.mean(auc_scores)
        dev_auc = statistics.stdev(auc_scores)
                
        print(f"Mean AUC: {mean_auc}")
        print(f"Standard Deviation: {dev_auc}")
        print(f"Max AUC: {max_auc}")

        # Save model to pickle file
        with open(f"models/{self.cfg['tag']}_MIMIC.pkl", 'wb') as file:
            pickle.dump(self.model, file)