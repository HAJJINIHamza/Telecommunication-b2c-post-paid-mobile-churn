import pandas as pd 
import numpy as np
from datetime import datetime
import xgboost as xgb
import pickle

from src.logger import logging
from src.eda import utils 

#Paths
train_dev_test_path = "data/train_dev_test"
ressources_path = "src/ressources"
data_path = "data/experiments_data"
x_y_sets_path = "data/x_y_sets"
models_path = "models/ml_models"
#Today date
date_time = datetime.today().strftime("%Y-%m-%d")

class TrainingPipeline():
    def __init__(self):
        pass

    def load_data(self, data_date):
        """
        Loading data
        Returns : x_train_norm, y_train, x_dev_norm, y_dev, x_test_norm, y_test
        """
        print ("Loading training data = x_train_norm, y_train, x_dev_norm, y_dev, x_test_norm, y_test ...............................................")
        x_train_norm = pd.read_csv(f"{x_y_sets_path}/{data_date}_x_train_norm.csv", index_col=0, nrows = 10000) #TODO: Delete nrows = 10000
        x_dev_norm = pd.read_csv(f"{x_y_sets_path}/{data_date}_x_dev_norm.csv", index_col=0, nrows = 10000) #TODO: Delete nrows = 10000
        x_test_norm = pd.read_csv(f"{x_y_sets_path}/{data_date}_x_test_norm.csv", index_col=0)
        y_train = pd.read_csv(f"{x_y_sets_path}/{data_date}_y_train.csv", index_col=0, nrows = 10000)  #TODO: Delete nrows = 10000
        y_dev = pd.read_csv(f"{x_y_sets_path}/{data_date}_y_dev.csv", index_col=0, nrows = 10000)  #TODO: Delete nrows = 10000
        y_test = pd.read_csv(f"{x_y_sets_path}/{data_date}_y_test.csv", index_col=0)
        print ("------------------")
        print (f"x_train shape : {x_train_norm.shape}")
        print (f"y_train shape : {y_train.shape}")
        print ("------------------")
        print (f"x_dev shape : {x_dev_norm.shape}")
        print (f"y_dev shape : {y_dev.shape}")
        print ("------------------")
        print (f"x_test shape : {x_test_norm.shape}")
        print (f"y_test shape : {y_test.shape}")
        logging.info("Loading training data = x_train_norm, y_train, x_dev_norm, y_dev, x_test_norm, y_test")
        return x_train_norm, y_train, x_dev_norm, y_dev, x_test_norm, y_test

    def train_model(self, 
                    x_train_norm, y_train, 
                    x_dev_norm, y_dev, 
                    x_test_norm, y_test, 
                    num_boosting_round:int, 
                    early_stopping_rounds:int, 
                    eval_metric = "logloss"):
        """
        Model training on imported data
        returns : trained model and eval_hist witch is the history of training and dev loss (or the chosen metric) and data
        returns : XGB_MODEL, eval_hist, dtrain, ddev, dtest
        Parameters : 
        eval_metric : string: logloss, auc or aucpr
        num_boosting_round : int: Equivalant to the number of estimators in xgboost or nbr of boosting rounds
        early_stopping_rounds : stop the training if the performance on eval data is not increasing after this number of rounds
        """
        if eval_metric not in ["logloss", "auc", "aucpr"]:
            raise ValueError("""eval_metric parameter must be on of the elements of this list ["logloss", "auc", "aucpr"]""")
        
        print ("Transforming DataFrames to DMatrix ...............................................")
        #Data to DMatrix
        dtrain = xgb.DMatrix(data=x_train_norm, label=y_train)
        ddev = xgb.DMatrix(data=x_dev_norm, label=y_dev)
        dtest = xgb.DMatrix(data=x_test_norm, label=y_test)
        logging.info("Transformed DataFrames to DMatrix")
        evals = [(dtrain, "train"), (ddev, "dev")]
        eval_hist = {}
        #Model parameters 
        params_2 = {
            'objective': 'binary:logistic',  
            'eval_metric': eval_metric,     
            'eta': 0.1,                     
            'max_depth': 5,                  
            'subsample': 0.9,                 
            'colsample_bytree': 1,           
            'min_child_weight': 1,            
            'gamma': 0.1,                     
            'scale_pos_weight': 1,                        
            'learning_rate': 0.01
        }
        print ("Training xgboost model ...............................................")
        #Trianing model
        XGB_MODEL = xgb.train( params = params_2,
                                dtrain=dtrain,
                                num_boost_round=num_boosting_round,
                                evals=evals,
                                evals_result=eval_hist,
                                early_stopping_rounds=early_stopping_rounds,  #Stop early if no improvement
                                verbose_eval=True
                            )
        logging.info("End of model training")
        return XGB_MODEL, eval_hist, dtrain, ddev, dtest
    
    def evaluate_model(self, MODEL, eval_hist,x_test, y_train, y_test, dtrain, dtest, THRESHOLD = 0.5):
        """
        Generate a report about model evaluation and performances
        """
        #Plot log loss
        print ("Ploting train loss and dev loss ...............................................")
        utils.vis_eval_metric(eval_hist, eval_metric="logloss")
        print ("Computing predictions on test data for evaluation...............................................")
        #Predictions
        y_test_predicted_prob = MODEL.predict(dtest)
        y_train_predicted_prob = MODEL.predict(dtrain)
        #Transform probas into predictions
        y_test_pred = [int(y_test_predicted_prob[i]>THRESHOLD) for i in range(len(y_test_predicted_prob))]
        y_train_pred = [int(y_train_predicted_prob[i]>THRESHOLD) for i in range(len(y_train_predicted_prob))]
        logging.info("Computing predictions on test data for evaluation")
        print ("Reporting model performance ...............................................")
        utils.report_model_performances(y_train, y_train_pred, y_test, y_test_pred, model_name = "xgboost")

        print ("Plot roc curve")
        utils.vis_roc_curve(y_test, y_test_predicted_prob)

        print ("Plot calibration curve to evaluate model calibration")
        utils.vis_calibration_curve(n_bins=[10, 20], y_test=y_test, y_test_predicted_prob=y_test_predicted_prob)

        print ("Get feature importance from model")
        importance =  MODEL.get_score(importance_type = "weight")
        #Plot importance
        utils.plot_feature_importance(importance.values(), importance.keys(), model_type="xgboost", max_n_features=100, figsize=(20, 20))

        print ("Plot number of correctly predicted scores and wrong ones")
        utils.vis_count_mistakes_and_correct_scores(y_test.values.flatten(), y_test_pred, y_test_predicted_prob)

        print ("Plot data distribution of acctual and predicted target with tsne")
        utils.vis_data_distribution_of_acctual_and_predicted_target_with_tsne(x_test, y_test, y_test_pred)

        print ("Precision recall curve for choosing the best thereshold")
        utils.vis_precision_recall_thereshold(y_test, y_test_predicted_prob)

        logging.info("Evaluated model and reproted its performances ")
    
    def save_model(self, MODEL, model_name):
        print ("Saving model ...............................................")
        with open(f"{models_path}/{date_time}_{model_name}.pkl", "wb") as f:
            pickle.dump(MODEL, f)
        logging.info("Saved model")
    
    def run_training_pipeline(self, data_date, num_boosting_rounds, early_stopping_rounds, eval_metric="logloss", THRESHOLD = 0.5):
        """
        Training pipeline include these steps :
        - load data : x_train_norm, y_train, x_dev_norm, y_dev, x_test_norm, y_test
        - train xgboost model evaluate model with plots 
        Returns : xgb model
        """
        x_train_norm, y_train, x_dev_norm, y_dev, x_test_norm, y_test = self.load_data(data_date)

        XGB_MODEL, eval_hist, dtrain, ddev, dtest = self.train_model(x_train_norm, y_train, 
                                                                        x_dev_norm, y_dev, 
                                                                        x_test_norm, y_test, 
                                                                        num_boosting_round = num_boosting_rounds, 
                                                                        early_stopping_rounds = early_stopping_rounds, 
                                                                        eval_metric = eval_metric
                                                                        )
        self.evaluate_model(XGB_MODEL , eval_hist, x_test_norm, y_train, y_test, dtrain, dtest, THRESHOLD )
        return XGB_MODEL

        #TODO: HAMZA DEBUG THIS PIPELINE
#END OF CLASS
        

        





