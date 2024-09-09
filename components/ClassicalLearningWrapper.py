from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from skopt.utils import use_named_args
from skopt import gp_minimize
import warnings
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
            # 'lsvm': SVC(kernel="linear", probability=True),
            # 'ksvm': SVC(kernel="rbf", probability=True),
            # 'knn': KNeighborsClassifier(),
        }
        self.model = self.model_tags[cfg['tag']]
        self.best_auc_params = None
        self.best_auc_score = None
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

    def BayesianHyperparameterOptimizer(self, val_sets):
        """
        Perform Bayesian hyperparameter optimization to find the best parameters for the model.

        Parameters:
        - val_sets: List of validation sets, each containing 'X_train', 'y_train', 'X_val', and 'y_val'.
        """
        warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignore FutureWarning

        # Define objective function for Bayesian optimization
        @use_named_args(self.cfg['auc_space'])
        def AUC_objective(**params):
            print("Testing params:", params)  # Print current test parameters to console
            auc = self.AUC_validate(params, val_sets)
            print("AUC for params:", auc)  # Print AUC for parameters console
            return -auc  # Invert to optimize for minimum AUC

        # Perform Bayesian Optimization
        result = gp_minimize(AUC_objective, self.cfg['auc_space'], n_calls=self.cfg['n_bayesian'], random_state=self.cfg['random_state'])

        # Extract the best parameters and the corresponding score
        self.best_auc_params = {dimension.name: result.x[i] for i, dimension in enumerate(self.cfg['auc_space'])}
        self.best_auc_score = -result.fun
        print("Best parameters found: ", self.best_auc_params)
        print("Best average AUC across validation sets: ", self.best_auc_score)

    # --------------------------------------------------------------      TRAINER      -------------------------------------------------------------- #  

    def Trainer(self, X_traindev, y_traindev):
        """
        Train the model using the best hyperparameters found during optimization.

        Parameters:
        - X_traindev: Training data features.
        - y_traindev: Training data labels.
        """
        self.model.set_params(**self.best_auc_params)
        self.model.fit(X_traindev, y_traindev)

        # Save model to pickle file
        with open(f"models/{self.cfg['tag']}_MIMIC.pkl", 'wb') as file:
            pickle.dump(self.model, file)

    # -------------------------------------------------------------      EVALUATOR      ------------------------------------------------------------- #

    def Evaluator(self, X_test, y_test):
        """
        Evaluate the model on the test set and print the AUC score.

        Parameters:
        - X_test: Test data features.
        - y_test: Test data labels.

        Returns:
        - AUC score on the test set.
        """
        y_pred = self.model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        print(f"Test AUC: {auc}")
        return auc