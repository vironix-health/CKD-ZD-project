import pandas as pd
import numpy as np
import pickle
import shap
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # modified
from sklearn.svm import SVC  # modified
from sklearn.tree import DecisionTreeClassifier  # modified
from sklearn.ensemble import RandomForestClassifier  # modified
from sklearn.neighbors import KNeighborsClassifier  # modified
import matplotlib.pyplot as plt
import warnings

# Cleaned Master MIMIC Data Set
df_master = pd.read_pickle(
    "/home/ammar/work/CKD-ZD-project/MIMIC_IV/df_ckd_master_clean.pkl"
)
df_base = pd.read_pickle("/home/ammar/work/CKD-ZD-project/MIMIC_IV/df_ckd_base.pkl")

# Drop unnecessary columns
drop_cols = [
    "anchor_year_group",
    "admittime_first",
    "dischtime_last",
    "hadm_id_last_CKD",
    "admittime_CKD",
    "dischtime_CKD",
    "deathtime",
    "Creatinine_first_time",
    "Creatinine_last_time",
    "Creatinine_min_time",
    "Creatinine_max_time",
    "Creatinine_mean_time",
    "Creatinine_median_time",
]

df_master.drop(columns=drop_cols, axis=1, inplace=True)
df_master = df_master.drop(columns=df_master.filter(like="CKD_stage_last").columns)
df_master = df_master.drop(columns=df_master.filter(like="last_stage_icd").columns)
df_master = df_master.drop(columns=df_master.filter(like="last_long_title").columns)
df_master = df_master.drop(
    columns=df_master.filter(like="Chronic kidney disease").columns
)
df_master = df_master.drop(
    columns=df_master.filter(like="chronic kidney disease").columns
)
df_master = df_master.drop(columns=df_master.filter(like="End stage renal").columns)

# Convert Int64 columns to int64 for tensor compatibility
int64_columns = df_master.select_dtypes(include=["Int64"]).columns
df_master[int64_columns] = df_master[int64_columns].astype("int64")

# Exclude response variable from features frame
X = df_master.drop("stage_delta", axis=1)

# Set response variable to stage_delta; binary CKD stage progression indicator
y = df_master["stage_delta"]


def create_splits(X, y, test_size=0.1, val_size=0.2, n_splits=5, seed=42):
    np.random.seed(seed)  # Ensure reproducibility

    # Split data into test and the remaining data
    X_traindev, X_test, y_traindev, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Further split the remaining data into multiple train and validation sets
    val_sets = []
    for _ in range(n_splits):
        # Randomly select validation set from the remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_traindev,
            y_traindev,
            test_size=val_size,
            random_state=np.random.randint(10000),
        )

        val_sets.append(
            {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val}
        )

    return X_traindev, y_traindev, X_test, y_test, val_sets


X_traindev, y_traindev, X_test, y_test, val_sets = create_splits(X, y, test_size=0.2)

# Initialize classification models #modified
models = {
    # "Decision Tree": DecisionTreeClassifier(),
    # "Random Forest": RandomForestClassifier(),
    # "Logistic Regression": LogisticRegression(max_iter=1000),
    # "Linear SVM": SVC(kernel="linear", probability=True),
    "Kernel SVM": SVC(kernel="rbf", probability=True),
    # "KNN": KNeighborsClassifier(),
}


# AUC evaluation function for validation sets
def AUC_validate(model, params, val_sets):
    auc_scores = []
    for val in val_sets:
        # Set parameters and reinitialize model to avoid leakage from previous fits
        model.set_params(**params)
        model.fit(val["X_train"], val["y_train"])

        try:
            # Predict probabilities on the validation set and calculate AUC
            preds_proba = model.predict_proba(val["X_val"])[
                :, 1
            ]  # Probability of the positive class
            auc = roc_auc_score(val["y_val"], preds_proba)
        except Exception as e:
            print(f"Error during model prediction: {str(e)}")
            auc = 0.0  # Consider the worst case if prediction fails

        auc_scores.append(auc)

    # Calculate average AUC across all validation sets
    return np.mean(auc_scores)


warnings.simplefilter(action="ignore", category=FutureWarning)

# Define the search space #modified
auc_spaces = {
    "Logistic Regression": [Real(0.01, 1.0, name="C")],
    "Linear SVM": [Real(0.01, 1.0, name="C")],
    "Kernel SVM": [Real(0.01, 1.0, name="C"), Real(0.01, 10.0, name="gamma")],
    "Decision Tree": [
        Integer(1, 50, name="max_depth"),
        Integer(2, 20, name="min_samples_split"),
        Integer(1, 20, name="min_samples_leaf"),
    ],
    "Random Forest": [
        Integer(50, 200, name="n_estimators"),
        Integer(2, 50, name="max_depth"),
        Integer(2, 20, name="min_samples_split"),
        Integer(1, 20, name="min_samples_leaf"),
    ],
    "KNN": [Integer(1, 50, name="n_neighbors")],
}

# Perform Bayesian Optimization #modified
best_auc_results = {}
for model_name, model in models.items():
    auc_space = auc_spaces[model_name]

    @use_named_args(auc_space)
    def AUC_objective(**params):
        print(
            f"Testing params for {model_name}:", params
        )  # Print parameters to console
        auc = AUC_validate(model, params, val_sets)
        print(
            f"AUC for params in {model_name}:", auc
        )  # Print AUC for parameters console
        return -auc  # Invert to optimize for minimum AUC

    result = gp_minimize(AUC_objective, auc_space, n_calls=10, random_state=42)

    # Extract the best parameters and the corresponding score
    best_auc_params = {
        dimension.name: result.x[i] for i, dimension in enumerate(auc_space)
    }
    best_auc_score = -result.fun  # Convert back to positive AUC

    best_auc_results[model_name] = {"params": best_auc_params, "score": best_auc_score}

    print(f"Best parameters found for {model_name}: ", best_auc_params)
    print(f"Best average AUC across validation sets for {model_name}: ", best_auc_score)

# Train each model on the full training set with tuned hyperparameters #modified
for model_name, model in models.items():
    print(f"Training {model_name} with best parameters...")
    model.set_params(**best_auc_results[model_name]["params"])
    model.fit(X_traindev, y_traindev)

    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)

    print(f"Area Under ROC Curve for {model_name}: {auc}")

    # Create a SHAP TreeExplainer #modified
    if model_name in ["Decision Tree", "Random Forest"]:  # Tree-based models
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(
            model.predict_proba, X_traindev
        )  # KernelExplainer for other models

    # Compute SHAP values for a set of data
    shap_values = explainer.shap_values(X_test)

    if len(shap_values.shape) == 3:  # For KernelExplainer output
        # Use the SHAP values for the positive class
        shap_values = shap_values[:, :, 1]

    # List of feature names to remove
    SHAP_feature_names = X.columns.tolist()
    KFRE_feature_names = df_base.columns.tolist()  # replace with actual feature names

    # Get the indices of the features to remove
    KFRE_feat_indices = [
        SHAP_feature_names.index(feat)
        for feat in KFRE_feature_names
        if feat in SHAP_feature_names
    ]

    # Create a mask to keep only specified features
    mask = np.ones(len(SHAP_feature_names), dtype=bool)
    mask[KFRE_feat_indices] = False

    # Apply the mask to shap_values and SHAP_feature_names
    filtered_shap_values = shap_values[:, mask]
    filtered_feature_names = [
        SHAP_feature_names[i] for i in range(len(SHAP_feature_names)) if mask[i]
    ]

    # Calculate the mean absolute SHAP value for each feature
    mean_abs_shap_values = np.mean(np.abs(filtered_shap_values), axis=0)  # ISSUE

    # Sort SHAP indices, values, and feature names by mean absolute SHAP value
    sorted_shap_indices = np.argsort(mean_abs_shap_values)[::-1]
    sorted_shap_values = mean_abs_shap_values[sorted_shap_indices]
    sorted_feature_names = np.array(filtered_feature_names)[sorted_shap_indices]

    # Extract top 40 features
    top_n = 40
    top_shap_values = sorted_shap_values[:top_n]  # (40, 2, 2)
    XGboost40 = sorted_feature_names[:top_n]  # Adjust the name as necessary for clarity

    # Create a horizontal bar plot
    plt.figure(figsize=(6, 8))
    # breakpoint()
    plt.barh(XGboost40[::-1], top_shap_values[::-1], color="skyblue")
    # plt.barh(range(len(XGboost40)), top_shap_values[::-1], color="skyblue")
    # plt.yticks(range(len(XGboost40)), [str(x) for x in XGboost40[::-1]])
    plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
    plt.title(model_name)

    # Save the plot as a PNG file
    plt.savefig(f"{model_name}_SHAP.png", format="png", dpi=300, bbox_inches="tight")

    plt.show()

    # Create a mask to select only the top 40 features
    top_40_mask = np.isin(filtered_feature_names, XGboost40)

    # Filter the SHAP values and feature names to keep only the top 40 features
    top_40_shap_values = filtered_shap_values[:, top_40_mask]
    top_40_feature_names = np.array(filtered_feature_names)[top_40_mask]

    # Filter the X_test DataFrame to only include the top 40 features
    X_test_top_40 = X_test.loc[:, top_40_feature_names]

    plt.figure(figsize=(9, 8))

    # Create the summary plot with the top 40 features
    shap.summary_plot(
        top_40_shap_values,
        X_test_top_40,
        feature_names=top_40_feature_names,
        plot_size=(10, 10),
        max_display=40,
        show=False,
    )

    # Customize plot appearance
    plt.title(model_name)
    plt.xlabel("SHAP value (impact on model output)")

    # Save the plot as a PNG file
    plt.savefig(
        f"{model_name}_Beeswarm.png", format="png", dpi=300, bbox_inches="tight"
    )

    # Show the plot
    plt.show()

    # Save the list to a pickle file
    with open(f"{model_name}_Top40.pkl", "wb") as f:
        pickle.dump(list(XGboost40), f)
