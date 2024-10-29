import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from datetime import datetime

from src.exception import CustomException
from src.logger import logging

#Get todays's date
date_time = datetime.today().strftime("%Y-%m-%d")
train_dev_test_path = "data/train_dev_test"


#This class aims at transforming ingested data, including sampling, creating new variables and train dev test split 
class DataTransformation:
    def __init__(self, df: pd.DataFrame ):
        self.df = df

    #Sample data 
    def sample_data_by_churn_segment(self):
        """
        Split original df by churn_segment variable, and take only nbr_non_churners, nbr_inactif_churners and nbr_churn_operateur
        Returns a dataframe after concatinating all the samples
        """
        try :
            nbr_non_churners = 150 #150000
            nbr_inactif_churners = 150 #140000
            nbr_churn_operateur = 150 #10602

            print ("Sampling data based on churn segement")
            print (f"nbr_non_churners {nbr_non_churners}")
            print (f"nbr_inactif_churners {nbr_inactif_churners}")
            print (f"nbr_churn_operateur {nbr_churn_operateur}")

            df_non_churners = self.df[self.df["churn_segment"].isin(["non_churners"])].sample(n=nbr_non_churners, random_state=42)
            df_inactif_churners = self.df[self.df["churn_segment"].isin(["inactif_unknown_churners"])].sample(n=nbr_inactif_churners, random_state=42)
            df_churn_operateur_actif = self.df[self.df["churn_segment"].isin(["churn_operateur_actif"])].sample(n=nbr_churn_operateur, random_state=42)
            concat_df = pd.concat ([df_non_churners, df_inactif_churners, df_churn_operateur_actif])
            logging.info("Succefully sampled data based on churn segement")

            return concat_df

        except Exception as e:
            CustomException(e, sys)
    
    
    
    #Create churn target from churn segment
    def get_churn_target_from_churn_segment(self, df:pd.DataFrame):
        """
        Returns df with new column "churn", from churn_segment variable, containing 0 if not churner else 1.
        """
        print ("Creating churn target from churn segemnt ")
        target_list = [0 if churn_segment == "non_churners" else 1 for churn_segment in df["churn_segment"].to_list()]
        df['churn'] = target_list
        logging.info("Successfully created churn target from churn segment")
        return self.df
    
    
    #Train test split 
    def get_train_dev_test_sets(self, df:pd.DataFrame):
        print ("Train, dev and test Spliting")
        df_train, df_dev = train_test_split(df, train_size = 0.499, random_state = 42, shuffle = True, stratify = df["churn"] )
        df_dev, df_test  = train_test_split(df_dev, train_size = 0.5, random_state = 42, shuffle = True, stratify = df_dev["churn"])

        n_dev_set = 20 #25000
        n_test_set = 10 #10000
        df_dev = df_dev.sample(n=n_dev_set, random_state=42)
        df_test = df_test.sample(n=n_test_set, random_state=42)

        print (f"df_train shape :{df_train.shape}")
        print (f"df_dev shape: {df_dev.shape}")
        print (f"df_test shape: {df_test.shape}")

        logging.info("Succefully splited data into train dev and test sets")
        logging.info (f"df_train shape :{df_train.shape}")
        logging.info (f"df_dev shape: {df_dev.shape}")
        logging.info (f"df_test shape: {df_test.shape}")

        return df_train, df_dev, df_test 
    
    def run_data_transformation(self):
        """
        Run transformation pipeline
        """
        df = self.sample_data_by_churn_segment()
        df = self.get_churn_target_from_churn_segment(df)
        df_train, df_dev, df_test = self.get_train_dev_test_sets(df)
        save_train_dev_test_sets(df_train, df_dev, df_test)
        return df_train, df_dev, df_test
#END OF CLASS
    
class HandlingMissingValues:
    def __init__(self, data_date: str):
        self.data_date = data_date

    def load_train_dev_test(self):
        """
        Loading train dev test sets of date
        """
        print ("Loading train dev and test sets, ensuring dn is a string type")
        df_train = pd.read_csv(f"{train_dev_test_path}/{self.data_date}_df_train.csv", index_col = 0, dtype= {"dn": "string"}) 
        df_dev = pd.read_csv(f"{train_dev_test_path}/{self.data_date}_df_dev.csv", index_col = 0, dtype= {"dn": "string"})
        df_test = pd.read_csv(f"{train_dev_test_path}/{self.data_date}_df_test.csv", index_col = 0, dtype= {"dn": "string"})
        print (f"df_train shape :{df_train.shape}")
        print (f"df_dev shape: {df_dev.shape}")
        print (f"df_test shape: {df_test.shape}")
        logging.info("Successfully loaded train dev and test sets")
        return df_train, df_dev, df_test
    
    def replace_0_values_with_nan(self, df_train, df_dev, df_test):
        #Get numerical columns from df
        df_numerical_columns = df_train.dtypes[df_train.dtypes != "object" ].index.to_list()
        df_numerical_columns = [ column for column in df_numerical_columns if column not in ["dn", "dn_group_id", "churn" ]]
        
        print ("Replacing 0 values with nan, in df_train, df_dev and df_test")
        df_train[df_numerical_columns] = df_train[df_numerical_columns].replace(0, np.nan)
        df_dev[df_numerical_columns] = df_dev[df_numerical_columns].replace(0, np.nan)
        df_test[df_numerical_columns] = df_test[df_numerical_columns].replace(0, np.nan)
        logging.info("Successfully replaced 0 values with nan, in df_train, df_dev and df_test")
        print (f"Are still there any 0 values in df_train after transforming 0 values into nan : {(df_train[df_numerical_columns] == 0).any().any()} ")

        return df_train, df_dev, df_test 
    
    def drop_columns_and_rows_with_all_values_null(self, df_train, df_dev, df_test, threshold=99):
        """
        Drop columns and rows where nan values percentage is bigger than thereshold
        """
        print ("Deleting all df_train null columns")
        df_train = drop_df_null_columns(df_train, threshold = threshold)
        #Take same columns as df_train
        print ("Taking same columns as df_train in df_dev and df_test")
        df_dev = df_dev[df_train.columns]
        df_test = df_test[df_train.columns]
        logging.info("Droped all columns with all values nulles")

        print ("Deleting all null rows from train dev and test sets")
        print ("Drop all null rows of df_train")
        df_train_T = df_train.T
        df_train_T=drop_df_null_columns(df_train_T, threshold=threshold)
        df_train = df_train_T.T
        print ("Drop all null rows of df_train")
        df_dev_T = df_dev.T
        df_dev_T=drop_df_null_columns(df_dev_T, threshold=threshold)
        df_dev = df_dev_T.T
        print ("Drop all null rows of df_train")
        df_test_T = df_test.T
        df_test_T=drop_df_null_columns(df_test_T, threshold=threshold)
        df_test = df_test_T.T
        logging.info("Droped all rows with all values nulles")

        print(f"df_train shape : {df_train.shape}")
        print(f"df_dev shape : {df_dev.shape}")
        print(f"df_test shape : {df_test.shape}")

        return df_train, df_dev, df_test
    
    def run_handling_missing_values(self):
        df_train, df_dev, df_test = self.load_train_dev_test()
        df_train, df_dev, df_test = self.replace_0_values_with_nan(df_train, df_dev, df_test)
        df_train, df_dev, df_test = self.drop_columns_and_rows_with_all_values_null(df_train, df_dev, df_test, threshold=99)
        print ("filling all NAN with 0, in train dev and test sets")
        df_train = df_train.fillna(0)
        df_dev = df_dev.fillna(0)
        df_test = df_test.fillna(0)
        logging.info("Filled all NAN values with 0, in train dev and test sets")
        save_train_dev_test_sets(df_train, df_dev, df_test, name_sufix="_fillna_0")
#END OF CLASS


def save_train_dev_test_sets( df_train, df_dev, df_test, name_sufix=""):
    """
    Save train, dev and test sets 
    Parameters:
    -----------
    name_sufix: string, the sufix to add to the name (df_train) of data when saved 
            example : _fillna_0 -> df_train_fillna_0
    """
    print ("Saving train dev and test sets")
    df_train.to_csv(f"{train_dev_test_path}/{date_time}_df_train{name_sufix}.csv", index=True)
    df_dev.to_csv(f"{train_dev_test_path}/{date_time}_df_dev{name_sufix}.csv", index=True)
    df_test.to_csv(f"{train_dev_test_path}/{date_time}_df_test{name_sufix}.csv", index=True)
    logging.info("Successfully saved train dev and test sets")

def drop_df_null_columns(df, threshold = 99):
    """
    Drop columns that contains a percentage of null values that surpass thereshold from dataframe
    Paramerters:
    ------------
    feature_names: list of features to look for null values in, and drop only null columns from this list
    thereshold: is a percentage not a number
    """
    
    df_missing_values_per_column = pd.DataFrame (
                                        { "column": ((df.isna().sum()/len(df))*100).index, 
                                        "prc_null_values": ((df.isna().sum()/len(df))*100).to_list() }
                                        )
    #Get list of null columns
    null_columns_to_be_deleted = df_missing_values_per_column [df_missing_values_per_column['prc_null_values'] > threshold]['column'].to_list()
    print (f"number of null columns in dataframe : {len(null_columns_to_be_deleted)}")
    #Drop null columns from df
    df.drop(null_columns_to_be_deleted, axis = "columns", inplace = True)
    print (f"DataFrame new shape {df.shape}")
    return df








    






    




