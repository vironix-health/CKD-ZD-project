import numpy as np
from sksurv.util import Surv
from sksurv.metrics import brier_score, cumulative_dynamic_auc
import matplotlib.pyplot as plt


class ChartHandler:
    def __init__(self, cfg):
        """
        Initialize the ChartHandler class with configuration settings.

        Parameters:
        - cfg: Configuration dictionary containing various settings.
        """
        self.cfg = cfg

    def plot_MultipleBrier(self):
        """
        Plot and save the Brier scores for multiple datasets over time.

        Parameters:
        - datasets: List of tuples, where each tuple contains (train, test) DataFrames.
        - cfgs: List of configuration dictionaries corresponding to each dataset.
        - labels: List of labels for each dataset to be used in the plot legend.
        """
        plt.figure(figsize=(17.5, 10))

        for model in self.cfg['models']:
            # Prepare the survival data in the format required by scikit-survival
            survival_train = Surv.from_dataframe(self.cfg['response'], self.cfg['duration'], model.train)
            survival_test = Surv.from_dataframe(self.cfg['response'], self.cfg['duration'], model.test)

            # Prepare the feature data
            X_train = model.train.drop(columns=[self.cfg['duration'], 'stage_delta'])
            X_test = model.test.drop(columns=[self.cfg['duration'], 'stage_delta'])

            # Predict the survival function for the test data using the fitted model
            survival_functions = model.cph.predict_survival_function(X_test)

            # Extract available times directly from the survival functions
            available_times = survival_functions.index

            # Ensure valid_times are strictly within the follow-up time range of the test data
            test_duration_min, test_duration_max = model.test[self.cfg['duration']].min(), model.test[self.cfg['duration']].max()
            valid_times = available_times[(available_times > test_duration_min) & (available_times < test_duration_max)]

            # Extract survival probabilities at the specified time points
            estimate = survival_functions.loc[valid_times].T.values

            # Calculate Brier scores using scikit-survival
            valid_times, brier_scores = brier_score(
                survival_train=survival_train,
                survival_test=survival_test,
                estimate=estimate,
                times=valid_times
            )

            # Plot the Brier scores
            plt.plot(valid_times, brier_scores, marker='o', markersize=2, label=model.cfg['name'])
            # plt.plot(valid_times, brier_scores, label=model.cfg['name'])

        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Brier Score', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(np.arange(0, 0.225, 0.025), fontsize=12)
        plt.grid(True)
        plt.xlim(-50, 2900)
        plt.ylim(-0.01, 0.225)
        plt.legend(fontsize=14)
        
        # Save the plot as a PNG file
        plt.savefig("figs/multiple_Brier.png", format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_MultipleDynamicAUC(self):
        plt.figure(figsize=(17.5, 10))

        for model in self.cfg['models']:
            # Prepare the survival data in the format required by scikit-survival
            survival_train = Surv.from_dataframe(self.cfg['response'], self.cfg['duration'], model.train)
            survival_test = Surv.from_dataframe(self.cfg['response'], self.cfg['duration'], model.test)

            # Prepare the feature data
            X_train = model.train.drop(columns=[self.cfg['duration'], 'stage_delta'])
            X_test = model.test.drop(columns=[self.cfg['duration'], 'stage_delta'])

            # Predict risk scores for the test data using the fitted model
            risk_scores = model.cph.predict_partial_hazard(X_test)

            # Define time points to evaluate ROC curves
            times = np.arange(1, model.test[self.cfg['duration']].max() + 1, 1)

            # Ensure times are within the follow-up time range of the test data
            valid_times = times[(times >= model.test[self.cfg['duration']].min()) & (times < model.test[self.cfg['duration']].max())]

            # Compute time-dependent ROC curves using valid times
            auc_values, mean_auc = cumulative_dynamic_auc(survival_train, survival_test, risk_scores, valid_times)

            # Plot mean AUC over time
            plt.plot(valid_times, auc_values, marker='o', markersize=2, label=model.cfg['name'])

        plt.xlabel('Time', fontsize=14)
        plt.ylabel('AUC', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.xlim(-100, 3000)
        plt.ylim(0.70, 1.0)
        plt.legend(fontsize=14)
        
        # Save the plot as a PNG file
        plt.savefig("figs/multiple_DynamicAUC.png", format="png", dpi=300, bbox_inches="tight")

        plt.show()