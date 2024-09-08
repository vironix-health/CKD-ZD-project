import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from lifelines.statistics import proportional_hazard_test
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from IPython.display import display
import matplotlib.pyplot as plt
import statistics
import copy


class CoxPHWrapper:
    def __init__(self, cfg, base_augment):
        self.cfg = cfg
        self.base_augment = base_augment
        self.best_c_index, self.avg_c_index, self.std_c_index, self.cph, self.train, self.test = self.cph_KFold()

    def cph_KFold(self):
        # Initialize variables to store the best split and best score
        scores = []
        cph = None
        best_c_index = -1
        best_train_split = None
        best_test_split = None
        
        cph_CrossVal = CoxPHFitter(penalizer=self.cfg['penalizer'])
        
        # Define the number of folds
        kf = KFold(n_splits=self.cfg['n_folds'], shuffle=True, random_state=self.cfg['random_state'])
        
        # Perform manual cross-validation
        for train_index, test_index in kf.split(self.base_augment):
            # Split the data into train and test sets
            train = self.base_augment.iloc[train_index]
            test = self.base_augment.iloc[test_index]
        
            # Fit the model on the training set
            cph_CrossVal.fit(train, duration_col=self.cfg['duration'], event_col=self.cfg['response'], show_progress=True)
        
            # Calculate the C-index on the test set
            c_index = cph_CrossVal.score(test, scoring_method="concordance_index")
        
            # Add C-index to list of scores
            scores.append(c_index)
        
            # Update the best score and corresponding train/test splits if this fold is better
            if c_index > best_c_index:
                cph = copy.deepcopy(cph_CrossVal)
                best_c_index = c_index
                best_train_split = train
                best_test_split = test

        avg_c_index = statistics.mean(scores)
        std_c_index = statistics.stdev(scores)

        print(f'Average C-Index from cross-validation: {avg_c_index:.4f}') # Output the average C-index from cross validation
        print(f'Best C-Index from cross-validation: {best_c_index:.4f}') # Output the best C-index and the corresponding train/test split indices
        print(f'Standard Deviation of C-Index from cross-validation: {std_c_index:.4f}') # Output the standard deviation of C-index from cross validation

        return best_c_index, avg_c_index, std_c_index, cph, best_train_split, best_test_split

    def Summary(self):
        cph_summary = self.cph.summary

        # Save the summary to a CSV file
        cph_summary.to_csv(f"models/{self.cfg['tag']}_CoxPH_Summary.csv", index=True)

        # Print the model summary
        display(cph_summary)       

    def FeatureRank(self):
        # Extract the coefficients and their corresponding feature names
        coefficients = self.cph.params_
        feature_importances = pd.DataFrame({
            'Feature': coefficients.index,
            'Coefficient': coefficients.values
        })

        # Sort the features by the absolute value of the coefficients
        feature_importances['Abs_Coefficient'] = feature_importances['Coefficient'].abs()
        feature_importances = feature_importances.sort_values(by='Abs_Coefficient', ascending=False)

        # Plot the feature importances
        plt.figure(figsize=(10, 14))
        plt.barh(feature_importances['Feature'], feature_importances['Coefficient'])
        plt.xlabel('Coefficient')
        plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top

        # Save the plot as a PNG file
        plt.savefig(f"figs/{self.cfg['tag']}_CoxPH_FeatureRank.png", format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def SchoenfeldTest(self):
        # Perform Schoenfeld test to assess the proportional hazards assumption
        results = proportional_hazard_test(self.cph, self.train, time_transform='rank')
        results_summary = results.summary

        # Save the results summary to a CSV file
        results_summary.to_csv(f"models/{self.cfg['tag']}_CoxPH_Schoenfeld.csv", index=True)

        display(results_summary)

    def plot_BrierScore(self):
        # Prepare the survival data in the format required by scikit-survival
        survival_train = Surv.from_dataframe(self.cfg['response'], self.cfg['duration'], self.train)
        survival_test = Surv.from_dataframe(self.cfg['response'], self.cfg['duration'], self.test)

        # Prepare the feature data
        X_train = self.train.drop(columns=[self.cfg['duration'], 'stage_delta'])
        X_test = self.test.drop(columns=[self.cfg['duration'], 'stage_delta'])

        # Predict the survival function for the test data using the fitted model
        survival_functions = self.cph.predict_survival_function(X_test)

        # Extract available times directly from the survival functions
        available_times = survival_functions.index

        # Ensure valid_times are strictly within the follow-up time range of the test data
        test_duration_min, test_duration_max = self.test[self.cfg['duration']].min(), self.test[self.cfg['duration']].max()
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

        # Create a plot of Brier scores over time
        plt.figure(figsize=(14, 8))
        plt.plot(valid_times, brier_scores, marker='o', markersize=3)
        plt.xlabel('Time')
        plt.ylabel('Average Brier Score')
        plt.yticks(np.arange(0, 0.225, 0.025))
        plt.grid(True)
        plt.xlim(-50, 2900)
        plt.ylim(-0.01, 0.225)

        # Save the plot as a PNG file
        plt.savefig(f"figs/{self.cfg['tag']}_Brier.png", format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_DynamicAUC(self):
        # Prepare the survival data in the format required by scikit-survival
        survival_train = Surv.from_dataframe(self.cfg['response'], self.cfg['duration'], self.train)
        survival_test = Surv.from_dataframe(self.cfg['response'], self.cfg['duration'], self.test)

        # Prepare the feature data
        X_train = self.train.drop(columns=[self.cfg['duration'], 'stage_delta'])
        X_test = self.test.drop(columns=[self.cfg['duration'], 'stage_delta'])

        # Predict risk scores for the test data using the fitted model
        risk_scores = self.cph.predict_partial_hazard(X_test)

        # Define time points to evaluate ROC curves
        times = np.arange(1, self.test[self.cfg['duration']].max() + 1, 1)

        # Ensure times are within the follow-up time range of the test data
        valid_times = times[(times >= self.test[self.cfg['duration']].min()) & (times < self.test[self.cfg['duration']].max())]

        # Compute time-dependent ROC curves using valid times
        auc_values, mean_auc = cumulative_dynamic_auc(survival_train, survival_test, risk_scores, valid_times)

        # Plot mean AUC over time
        plt.figure(figsize=(14, 8))
        plt.plot(valid_times, auc_values, marker='o', markersize=3)
        plt.xlabel('Time')
        plt.ylabel('AUC')
        plt.grid(True)
        plt.xlim(-100, 3000)
        plt.ylim(0.70, 1.0)

        # Save the plot as a PNG file
        plt.savefig(f"figs/{self.cfg['tag']}_DynamicAUC.png", format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_SurvivalCurve(self, stage):
        # Filter the test set for the given CKD stage
        test_stage = self.test[self.test['CKD_stage_first'] == stage]

        # Predict the survival function for the filtered test set
        survival_functions = self.cph.predict_survival_function(test_stage)

        # Calculate the time when survival probability first drops below 0.1
        drop_below_0_1_times = survival_functions.apply(
            lambda x: x.index[x <= 0.1].min() if any(x <= 0.1) else x.index[-1], axis=0
        )

        # Sort by the time of first drop below 0.1
        sorted_indices = drop_below_0_1_times.sort_values().index

        # Select first, the last, and 8 evenly spaced in between
        selected_indices = [
            sorted_indices[0],  # Fastest drop below 0.1
            sorted_indices[-1]  # Slowest drop below 0.1
        ] + list(sorted_indices[np.linspace(1, len(sorted_indices) - 2, 8, dtype=int)])

        # Plot the survival curves for the selected indices
        plt.figure(figsize=(12, 8))
        for idx in selected_indices:
            plt.plot(survival_functions.index, survival_functions[idx], label=f'ID {idx}')

        plt.title(f'Representative Survival Curves for CKD Stage {stage}')
        plt.xlabel('Days')
        plt.ylabel('Survival Probability')
        plt.legend()

        # Save the plot as a PNG file
        plt.savefig(f"figs/{self.cfg['tag']}_SurvCurv{stage}.png", format="png", dpi=300, bbox_inches="tight")

        plt.show()

    def plot_SurvivalCurves(self, start, stop):
        for stage in range(start, stop + 1):
            self.plot_SurvivalCurve(stage) # Plot survival curves for CKD stages in range [start, stop]