import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataHandler:
    def __init__(self, cfg, master, base, preprocess=True):
        """
        Initialize the DataHandler class.

        Parameters:
        - cfg: Configuration dictionary containing various settings.
        - master: The main dataset.
        - base: The base dataset for augmentation.
        - preprocess: Boolean flag to indicate if preprocessing is needed.
        """
        self.cfg = cfg
        self.master = self.preprocess(master) if preprocess else master
        self.master_features = None
        self.base = base
        self.augment = None
        self.seed_splits = []

        # Initialize the scaler
        self.scaler = StandardScaler()
        self.scale_flag = True if cfg['tag'] in ['lr', 'fcnn', 'rsnt'] else False

    # ------------------------------------------------      DATA HANDLING      ------------------------------------------------ #
    
    def preprocess(self, data):
        """
        Preprocess the data by removing specific columns and converting data types.

        Parameters:
        - data: The dataset to preprocess.

        Returns:
        - Preprocessed dataset.
        """
        # Drop columns related to chronic kidney disease
        data = data.drop(columns=data.filter(like='Chronic kidney disease').columns)
        data = data.drop(columns=data.filter(like='chronic kidney disease').columns)
        data = data.drop(columns=data.filter(like='End stage renal').columns)

        # Convert Int64 columns to int64 for tensor compatibility
        int64_columns = data.select_dtypes(include=['Int64']).columns
        data[int64_columns] = data[int64_columns].astype('int64')

        return data
    
    # ------------------------------------------------      PREPROCESSING      ------------------------------------------------ #
    
    def split_Vanilla(self):
        """
        Split the data into training/development and test sets.
        """
        # Exclude response variable from features frame
        X = self.master.drop(self.cfg['response'], axis=1)

        # Store the original master feature names
        self.master_features = X.columns.tolist()

        # Set response variable 
        y = self.master[self.cfg['response']]

        if self.scale_flag:
            X = self.scaler.fit_transform(X)  # Scale the features
            
        for seed in self.cfg['seeds']:
            X_traindev, X_test, y_traindev, y_test = train_test_split(
                X, y, test_size=self.cfg['test_size'], random_state=seed
            )

            self.seed_splits.append({
                'X_traindev': X_traindev,
                'X_test': X_test,
                'y_traindev': y_traindev,
                'y_test': y_test,
                'val_sets': [],
                "X_train_tensor": None,
                "X_test_tensor": None,
                "y_train_tensor": None,
                "y_test_tensor": None,
                "train_dataset": None,
                "test_dataset": None,
                "train_loader": None,
                "test_loader": None
            })
        
    def split_ValSets(self):
        """
        Split the data into multiple training and validation sets.
        """
        # Split data into test and the remaining data
        self.split_Vanilla()

        for seed_split in self.seed_splits:
            # Further split the remaining data into multiple train and validation sets
            for _ in range(self.cfg['n_valsets']):
                # Randomly select validation set from the remaining data
                X_train, X_val, y_train, y_val = train_test_split(
                    seed_split['X_traindev'], seed_split['y_traindev'], test_size=self.cfg['val_size'], random_state=np.random.randint(10000)
                )
                
                seed_split['val_sets'].append({
                    'X_train': X_train, 
                    'y_train': y_train, 
                    'X_val': X_val, 
                    'y_val': y_val
                })

    # ------------------------------------------------      TENSOR WORKFLOW      ------------------------------------------------ #

    def tensorWorkFlow(self):
        """
        Convert data to PyTorch tensors and create DataLoader for batch processing.
        """

        for seed_split in self.seed_splits:
            # Convert data to PyTorch tensors
            seed_split['X_train_tensor'] = torch.tensor(seed_split['X_traindev'], dtype=torch.float32)
            seed_split['X_test_tensor'] = torch.tensor(seed_split['X_test'], dtype=torch.float32)
            seed_split['y_train_tensor'] = torch.tensor(seed_split['y_traindev'].values, dtype=torch.float32).view(-1, 1)
            seed_split['y_test_tensor'] = torch.tensor(seed_split['y_test'].values, dtype=torch.float32).view(-1, 1)

            # Ensure no NaN values in tensors
            assert not seed_split['X_train_tensor'].isnan().any(), "NaN values found in X_train_tensor"
            assert not seed_split['y_train_tensor'].isnan().any(), "NaN values found in y_train_tensor"

            # Create DataLoader for batch processing
            seed_split['train_dataset'] = TensorDataset(seed_split['X_train_tensor'], seed_split['y_train_tensor'])
            seed_split['test_dataset'] = TensorDataset(seed_split['X_test_tensor'], seed_split['y_test_tensor'])

            seed_split['train_loader'] = DataLoader(seed_split['train_dataset'], batch_size=64, shuffle=True)
            seed_split['test_loader'] = DataLoader(seed_split['test_dataset'], batch_size=64, shuffle=False)

    # ------------------------------------------------      DATA AUGMENTATION      ------------------------------------------------ #

    def baseAugmentation(self, novel_features):
        """
        Augment the base dataset with novel predictor features.

        Parameters:
        - novel_features: List of novel features to augment.
        """
        # Select these columns from df_source
        novel_features = ['subject_id'] + novel_features.tolist()
        df_novel = self.master[novel_features]

        # Merge novel predictor features with base features
        self.augment = pd.merge(self.base, df_novel, on='subject_id', how='outer')