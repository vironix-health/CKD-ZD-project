from components.DataHandler import DataHandler
from components.SHAPSelector import SHAPSelector
from components.ClassicalLearningWrapper import ClassicalLearningWrapper
from components.DeepLearningWrapper import DeepLearningWrapper
from components.CoxPHWrapper import CoxPHWrapper
import pickle


class Driver:
    def __init__(self, cfg, master=None, base=None):
        """
        Initialize the Driver class with configuration settings, master dataset, and base dataset.

        Parameters:
        - cfg: Configuration dictionary containing various settings.
        - master: Master dataset.
        - base: Base dataset.
        """
        self.cfg = cfg
        self.master = master
        self.base = base

    # --------------------------------------------------      PIPELINES      -------------------------------------------------- #   

    def VanillaPipe(self):
        """
        Execute the vanilla pipeline which includes only CoxPH evaluation with base dataset.
        """
        CoxPH = CoxPHWrapper(self.cfg, self.base) # Initialize CoxPHWrapper with base dataset
        CoxPH.Summary()
        CoxPH.FeatureRank()
        CoxPH.SchoenfeldTest()
        CoxPH.plot_BrierScore()
        CoxPH.plot_DynamicAUC()
        CoxPH.plot_SurvivalCurves(2, 5) # Plot survival curves for CKD stages 2-5
        
        with open(f"models/{self.cfg['tag']}_CoxPH.pkl", 'wb') as f:
            pickle.dump(CoxPH, f)

    def ClassicalLearningPipe(self):
        """
        Execute the classical learning pipeline which includes data handling, model training, evaluation, 
        SHAP value computation, feature selection, and CoxPH evaluation.
        """
        # Split data into training, testing, and validation sets
        Data = DataHandler(self.cfg, self.master, self.base)
        Data.split_ValSets()
        
        # Initialize ClassicalLearningWrapper and perform Bayesian hyperparameter optimization
        CLWrapper = ClassicalLearningWrapper(self.cfg)
        CLWrapper.BayesianHyperparameterOptimizer(Data.seed_splits)
        CLWrapper.Evaluator(Data.seed_splits)

        # Compute SHAP values, select novel predictor features
        Selector = SHAPSelector(
            self.cfg,
            Data.master_features,
            self.base, 
            CLWrapper.model,
            CLWrapper.X_traindev,
            CLWrapper.X_test, 
        )
        Selector.SHAP_FeatureFilter()
        Selector.plot_SHAPma()
        Selector.plot_ClassicalSHAPbeeswarm()
        
        # Augment base dataset with novel predictor features
        Data.baseAugmentation(Selector.get_NovelFeatures())
        
        # Initialize CoxPHWrapper and evaluate on augmented dataset
        CoxPH = CoxPHWrapper(self.cfg, Data.augment)
        CoxPH.Summary()
        CoxPH.FeatureRank()
        CoxPH.SchoenfeldTest()
        CoxPH.plot_BrierScore()
        CoxPH.plot_DynamicAUC()        
        CoxPH.plot_SurvivalCurves(2, 5) # Plot survival curves for CKD stages 2-5

        with open(f"models/{self.cfg['tag']}_CoxPH.pkl", 'wb') as f:
            pickle.dump(CoxPH, f)

    def DeepLearningPipe(self):
        """
        Execute the deep learning pipeline which includes data handling, model training, evaluation, 
        SHAP value computation, feature selection, and CoxPH evaluation.
        """
        # Split data into train and test; initialize tensors and dataloaders
        Data = DataHandler(self.cfg, self.master, self.base)
        Data.split_Vanilla()
        Data.tensorWorkFlow()

        # Initialize DeepLearningWrapper and perform training and evaluation
        DLWrapper = DeepLearningWrapper(self.cfg, Data.seed_splits[0]['X_traindev'].shape[1])
        DLWrapper.Evaluator(Data.seed_splits)
        
        # Compute SHAP values, select novel predictor features
        Selector = SHAPSelector(
            self.cfg,
            Data.master_features,
            self.base, 
            DLWrapper.model,
            DLWrapper.X_traindev,
            DLWrapper.X_test,
            DLWrapper.X_train_tensor,
            DLWrapper.X_test_tensor,
        )
        Selector.SHAP_FeatureFilter()
        Selector.plot_SHAPma()
        Selector.plot_DeepSHAPbeeswarm()
        
        # Augment base dataset with novel predictor features
        Data.baseAugmentation(Selector.get_NovelFeatures())
        
        # Initialize CoxPHWrapper and evaluate on augmented dataset
        CoxPH = CoxPHWrapper(self.cfg, Data.augment)
        CoxPH.Summary()
        CoxPH.FeatureRank()
        CoxPH.SchoenfeldTest()
        CoxPH.plot_BrierScore()
        CoxPH.plot_DynamicAUC()        
        CoxPH.plot_SurvivalCurves(2, 5) # Plot survival curves for CKD stages 2-5

        with open(f"models/{self.cfg['tag']}_CoxPH.pkl", 'wb') as f:
            pickle.dump(CoxPH, f)


