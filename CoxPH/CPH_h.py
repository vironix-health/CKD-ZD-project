import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import statistics
import copy


"""
    Perform K-Fold cross-validation for a Cox Proportional Hazards model.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - n_splits (int): The number of folds for cross-validation.
    - penalizer (float): The penalizer value to be used in the CoxPHFitter.

    Returns:
    - float: The best C-index score across the folds.
    - float: The average C-index score across all folds.
    - CoxPHFitter: The CoxPHFitter model fitted on the best split.
    - pandas.DataFrame: The training set corresponding to the best split.
    - pandas.DataFrame: The test set corresponding to the best split.
"""

def cph_KFold(df, n_splits, penalizer):
    # Initialize variables to store the best split and best score
    scores = []
    cph = None
    best_c_index = -1
    best_train_split = None
    best_test_split = None
    
    cph_CrossVal = CoxPHFitter(penalizer=penalizer)
    
    # Define the number of folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Perform manual cross-validation
    for train_index, test_index in kf.split(df):
        # Split the data into train and test sets
        train = df.iloc[train_index]
        test = df.iloc[test_index]
    
        # Fit the model on the training set
        cph_CrossVal.fit(train, duration_col='duration', event_col='stage_delta', show_progress=True)
    
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

    return best_c_index, avg_c_index, cph, best_train_split, best_test_split


"""
    Create a calibration plot comparing predicted survival probabilities 
    to observed survival probabilities at a specific time horizon.
    
    Parameters:
    - test (pandas.DataFrame): The test DataFrame.
    - cph (CoxPHFitter): The fitted CoxPHFitter model.
    - time_horizon (float): The time horizon in days for which to plot the calibration.

    Returns:
    - None: Displays the calibration plot.
"""

def create_calibration_plot(test, cph, time_horizon):
    # Predict survival probabilities
    survival_functions = cph.predict_survival_function(test)
    predicted_survival_probs = survival_functions.loc[time_horizon]

    # Create a copy of the test DataFrame to avoid modifying the original DataFrame
    test_copy = test.copy()

    # Group data into deciles based on predicted survival probabilities
    test_copy.loc[:, 'predicted_survival_prob'] = predicted_survival_probs.values
    test_copy.loc[:, 'decile'] = pd.qcut(test_copy['predicted_survival_prob'], 10, labels=False)

    # Calculate observed survival probabilities for each decile
    observed_survival_probs = []
    for i in range(10):
        decile_data = test_copy[test_copy['decile'] == i]
        # Calculate the observed survival probability
        num_events = decile_data[decile_data['stage_delta'] == 1].shape[0]
        num_at_risk = decile_data.shape[0]
        observed_survival_prob = (num_at_risk - num_events) / num_at_risk
        observed_survival_probs.append(observed_survival_prob)

    # Calculate the mean predicted survival probability for each decile
    mean_predicted_probs = test_copy.groupby('decile')['predicted_survival_prob'].mean()

    # Plot the calibration curve
    plt.figure(figsize=(10, 6))
    plt.plot(mean_predicted_probs, observed_survival_probs, 'o-', label='Observed vs Predicted')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel('Predicted Survival Probability')
    plt.ylabel('Observed Survival Probability')
    plt.title(f'Calibration Plot at {time_horizon} Days')
    plt.legend()
    plt.show()


"""
    Plot survival curves for a specific CKD stage, showing the probability 
    of stage progression over time for selected patient IDs.

    Parameters:
    - stage (int): The CKD stage for which to plot survival curves.
    - model (str): The name of the model being used, to include in the plot filename.
    - test (pandas.DataFrame): The test DataFrame.
    - cph (CoxPHFitter): The fitted CoxPHFitter model.

    Returns:
    - None: Displays the survival curves plot and saves it as a PNG file.
"""

# Function to plot survival curves for a specific CKD stage
def plot_survival_curves_for_stage(stage, model, test, cph):
    # Filter the test set for the given CKD stage
    test_stage = test[test['CKD_stage_first'] == stage]

    # Predict the survival function for the filtered test set
    survival_functions = cph.predict_survival_function(test_stage)

    # Calculate the time when survival probability first drops below 0.1
    drop_below_0_1_times = survival_functions.apply(
        lambda x: x.index[x <= 0.1].min() if any(x <= 0.1) else x.index[-1], axis=0
    )

    # Sort by the time of first drop below 0.1
    sorted_indices = drop_below_0_1_times.sort_values().index

    # Select indices: the first, the last, and 8 evenly spaced in between
    selected_indices = [
        sorted_indices[0],  # Fastest drop below 0.1
        sorted_indices[-1]  # Slowest drop below 0.1
    ] + list(sorted_indices[np.linspace(1, len(sorted_indices) - 2, 8, dtype=int)])  # Evenly spaced

    # Plot the survival curves for the selected indices
    plt.figure(figsize=(12, 8))
    for idx in selected_indices:
        plt.plot(survival_functions.index, survival_functions[idx], label=f'ID {idx}')

    plt.title(f'Representative Survival Curves for CKD Stage {stage}')
    plt.xlabel('Days')
    plt.ylabel('Survival Probability')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f'figs/{model}SurvCurv{stage}.png', format="png", dpi=300, bbox_inches="tight")

    plt.show()