from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from skopt.utils import use_named_args
from skopt import gp_minimize
import warnings
import numpy as np
import pickle


class ClassicalLearningWrapper:
    def __init__(self, cfg):
        # Initialize classification models
        self.model_tags = {
            'xgb': XGBClassifier(objective='binary:logistic', verbosity=0, tree_method='gpu_hist'),
            'dct': DecisionTreeClassifier(),
            'rdmf': RandomForestClassifier(),
            'lr': LogisticRegression(max_iter=1000),
            # 'lsvm': SVC(kernel="linear", probability=True),
            # "Kernel SVM": SVC(kernel="rbf", probability=True),
            # "KNN": KNeighborsClassifier(),
        }
        self.model = self.model_tags[cfg['tag']]
        self.best_auc_params = None
        self.best_auc_score = None
        self.cfg = cfg

    def AUC_validate(self, params, val_sets):
        auc_scores = []
        for val in val_sets:
            # params['tree_method'] = 'gpu_hist'
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
        warnings.simplefilter(action='ignore', category=FutureWarning) # ignore FutureWarning

        # Define objective function
        @use_named_args(self.cfg['auc_space'])
        def AUC_objective(**params):
            print("Testing params:", params)  # Print current test parameters to console
            auc = self.AUC_validate(params, val_sets)
            print("AUC for params:", auc) # Print AUC for parameters console
            return -auc # Invert to optimize for minimum AUC 
        
        # Perform Bayesian Optimization
        result = gp_minimize(AUC_objective, self.cfg['auc_space'], n_calls=self.cfg['n_bayesian'], random_state=self.cfg['random_state'])

        # Extract the best parameters and the corresponding score
        self.best_auc_params = {dimension.name: result.x[i] for i, dimension in enumerate(self.cfg['auc_space'])}
        self.best_auc_score = -result.fun
        print("Best parameters found: ", self.best_auc_params)
        print("Best average AUC across validation sets: ", self.best_auc_score)

    def Trainer(self, X_traindev, y_traindev):
        self.model.set_params(**self.best_auc_params)
        self.model.fit(X_traindev, y_traindev)

        # Save model to pickle file
        with open(f"models/{self.cfg['tag']}_MIMIC.pkl", 'wb') as file:
            pickle.dump(self.model, file)

    def Evaluator(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        print(f"Test AUC: {auc}")
        return auc