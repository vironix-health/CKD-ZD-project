import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, cfg, master, base, preprocess=True):
        self.cfg = cfg
        self.master = self.preprocess(master) if preprocess else master
        self.base = base
        self.scaler = StandardScaler()  # Initialize the scaler
        self.scale_flag = True if cfg['tag'] in ['lr'] else False
        self.X_traindev = None
        self.y_traindev = None
        self.X_test = None
        self.y_test = None
        self.val_sets = None
        self.master_features = None
        self.augment = None

    # ------------------------------------------------      DATA HANDLING      ------------------------------------------------ #
    
    def preprocess(self, data):
        # Drop unnecessary columns
        drop_cols = [
            'anchor_year',
            'anchor_year_group',
            "hadm_id_first",
            "admittime_first",
            "hadm_id_last",
            "dischtime_last",
            "hadm_id_first_CKD",
            "admittime_CKD",
            "hadm_id_last_CKD",
            "dischtime_CKD", 
            'hospital_expire_flag',
            'deathtime',
            "first_stage_icd",
            "CKD_stage_last",
            "last_stage_icd",
            "Creatinine_first_time",
            "Creatinine_last_time",
            "Creatinine_min_time",
            "Creatinine_max_time",
            "Creatinine_mean_time",
            "Creatinine_median_time"
        ]

        data.drop(columns=drop_cols, axis=1, inplace=True)
        data = data.drop(columns=data.filter(like="CKD_stage_last").columns)
        data = data.drop(columns=data.filter(like="last_stage_icd").columns)
        data = data.drop(columns=data.filter(like="last_long_title").columns)
        data = data.drop(columns=data.filter(like='Chronic kidney disease').columns)
        data = data.drop(columns=data.filter(like='chronic kidney disease').columns)
        data = data.drop(columns=data.filter(like='End stage renal').columns)

        # Convert Int64 columns to int64 for tensor compatibility
        int64_columns = data.select_dtypes(include=['Int64']).columns
        data[int64_columns] = data[int64_columns].astype('int64')

        return data
    
    def split_Vanilla(self):
        # Exclude response variable from features frame
        X = self.master.drop(self.cfg['response'], axis=1)

        # Store the original master feature names
        self.master_features = X.columns.tolist()

        # Set response variable 
        y = self.master[self.cfg['response']]
        
        np.random.seed(self.cfg['random_state'])  # Ensure reproducibility

        if self.scale_flag:
           X = self.scaler.fit_transform(X) # Scale the features
            
        # Split data into test and the remaining data
        self.X_traindev, self.X_test, self.y_traindev, self.y_test = train_test_split(X, y, test_size=self.cfg['test_size'], random_state=self.cfg['random_state'])
        
    def split_ValSets(self):
        # Split data into test and the remaining data
        self.split_Vanilla()

        # Further split the remaining data into multiple train and validation sets
        val_sets = []
        for _ in range(self.cfg['n_valsets']):
            # Randomly select validation set from the remaining data
            X_train, X_val, y_train, y_val = train_test_split(self.X_traindev, self.y_traindev, test_size=self.cfg['val_size'], random_state=np.random.randint(10000))
            
            val_sets.append({
                'X_train': X_train, 
                'y_train': y_train, 
                'X_val': X_val, 
                'y_val': y_val
            })

        self.val_sets = val_sets

    def baseAugmentation(self, novel_features):
        # Select these columns from df_source
        novel_features = ['subject_id'] + novel_features.tolist()
        df_novel = self.master[novel_features]

        # Merge novel predictor features with base features
        self.augment = pd.merge(self.base, df_novel, on='subject_id', how='outer')