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
        
        # Initialize data splits
        self.X_traindev = None
        self.y_traindev = None
        self.X_test = None
        self.y_test = None
        self.val_sets = None

        # Initialize tensors and DataLoader
        self.X_train_tensor = None
        self.X_test_tensor = None
        self.y_train_tensor = None
        self.y_test_tensor = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None

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
        
        np.random.seed(self.cfg['random_state'])  # Ensure reproducibility

        if self.scale_flag:
            X = self.scaler.fit_transform(X)  # Scale the features
            
        # Split data into test and the remaining data
        self.X_traindev, self.X_test, self.y_traindev, self.y_test = train_test_split(
            X, y, test_size=self.cfg['test_size'], random_state=self.cfg['random_state']
        )
        
    def split_ValSets(self):
        """
        Split the data into multiple training and validation sets.
        """
        # Split data into test and the remaining data
        self.split_Vanilla()

        # Further split the remaining data into multiple train and validation sets
        val_sets = []
        for _ in range(self.cfg['n_valsets']):
            # Randomly select validation set from the remaining data
            X_train, X_val, y_train, y_val = train_test_split(
                self.X_traindev, self.y_traindev, test_size=self.cfg['val_size'], random_state=np.random.randint(10000)
            )
            
            val_sets.append({
                'X_train': X_train, 
                'y_train': y_train, 
                'X_val': X_val, 
                'y_val': y_val
            })

        self.val_sets = val_sets

    # ------------------------------------------------      TENSOR WORKFLOW      ------------------------------------------------ #

    def tensorWorkFlow(self):
        """
        Convert data to PyTorch tensors and create DataLoader for batch processing.
        """
        # Convert data to PyTorch tensors
        self.X_train_tensor = torch.tensor(self.X_traindev, dtype=torch.float32)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_train_tensor = torch.tensor(self.y_traindev.values, dtype=torch.float32).view(-1, 1)
        self.y_test_tensor = torch.tensor(self.y_test.values, dtype=torch.float32).view(-1, 1)

        # Ensure no NaN values in tensors
        assert not self.X_train_tensor.isnan().any(), "NaN values found in X_train_tensor"
        assert not self.y_train_tensor.isnan().any(), "NaN values found in y_train_tensor"

        # Create DataLoader for batch processing
        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.test_dataset = TensorDataset(self.X_test_tensor, self.y_test_tensor)

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)

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