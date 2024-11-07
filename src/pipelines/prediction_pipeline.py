import pandas as pd
import numpy as np
import pickle 
from xgboost import DMatrix

from src.logger import logging


class PredictionPipeline():
    def __init__(self):
        pass

    def load_data(self, data_date):
        """
        Load inference data from data/inference_data path
        Returns x_norm and dns
        Parameters:
        -----------
        data_date: string, inference data date
        """
        print ("Loading data")
        x_norm = pd.read_csv(f"data/inference_data/{data_date}_x_norm.csv", index_col=0)
        dns = pd.read_csv(f"data/inference_data/{data_date}_dns.csv", index_col=0)
        logging.info("Loaded data")
        print ("data shape :", x_norm.shape)
        print ("dns shape :", dns.shape)
        return x_norm, dns
    
    def load_model(self, model_name):
        """
        Load model from model/ml_models path
        Returns model
        """
        print(f"Loading model {model_name}")
        with open(f"models/ml_models/{model_name}", "rb") as f:
            MODEL = pickle.load(f)
        logging.info(f"Loaded model {model_name}")
        return MODEL
    
    def predict_churners(self, data, dns, MODEL, THERESHOLD = 0.5):
        """
        Predict churners non churners from data with THERESHOLD
        Returns data frame of dns and new column "churn_segment"
        Parameters:
        -----------
        data : pd.DataFrame 
        MODEL : pickle model
        THERESHOLD : the probability thereshold based on wich classify as churner or not (between 0 and 1)
        """
        print (f"Predecting churners, probabilitie thereshold is:{THERESHOLD}")
        ddata = DMatrix(data=data)
        y_predicted_probabilities =  MODEL.predict(ddata)
        y_predicted = [int(y_predicted_probabilities[i]>THERESHOLD) for i in range(len(y_predicted_probabilities))]
        y_predicted = ["churner" if i==1 else "non_churner" for i in y_predicted]
        dns["churn_segment"] = y_predicted
        dns["churn_prediction_score"]=y_predicted_probabilities
        logging.info(f"Predicted churners, probabilitie thereshold is:{THERESHOLD}")
        return dns
    
    def run_prediction_pipeline(self,batch_date, model_name = "2024-10-25_xgb_model.pkl"):
        """
        Steps : 
        - load data and dns 
        - load model 
        - predict churners 

        Returns : dns_churn
        """
        logging.info("############################# Runing Prediction Pipeline ############################# ")
        x_norm, dns = self.load_data(data_date = batch_date)
        MODEL = self.load_model(model_name = model_name)
        dns_churn = self.predict_churners(data=x_norm, dns=dns,MODEL=MODEL)
        print ("Saving output data")
        dns_churn.to_csv(f"data/output_data/{batch_date}_dns_churn.csv", index=True)
        logging.info("Saved output data")
        return dns_churn
#END OF CLASS





        
        



