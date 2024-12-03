import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle
import ast

from src.exception import CustomException
from src.logger import logging

#Get todays's date
date_time = datetime.today().strftime("%Y-%m-%d")
train_dev_test_path = "data/train_dev_test"
ressources_path = "src/ressources"
data_path = "data/experiments_data"
x_y_sets_path = "data/x_y_sets"


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
            max_nbr_of_non_churners = len (self.df[self.df["churn_segment"].isin(["non_churners"])])
            max_nbr_of_inactif_churners = len (self.df[self.df["churn_segment"].isin(["inactif_unknown_churners"])])
            max_nbr_of_churn_operateur = len (self.df[self.df["churn_segment"].isin(["churn_operateur_actif"])])
            nbr_non_churners = min(243000, max_nbr_of_non_churners)
            nbr_inactif_churners = min (233097, max_nbr_of_inactif_churners)
            nbr_churn_operateur = min(10602, max_nbr_of_churn_operateur)

            print ("Sampling data based on churn segement .......................................................")
            print (f"max nbr of non_churners {max_nbr_of_non_churners} select nbr_non_churners {nbr_non_churners} ...........................")
            print (f"max nbr of inactif unknown churners {max_nbr_of_inactif_churners} selected nbr_inactif_churners {nbr_inactif_churners} ......")
            print (f"max nbr of nbr_churn_operateur {max_nbr_of_churn_operateur} selected nbr_churn_operateur {nbr_churn_operateur} ........")

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
        print ("Creating churn target from churn segemnt....................................................... ")
        target_list = [0 if churn_segment == "non_churners" else 1 for churn_segment in df["churn_segment"]]
        df['churn'] = target_list
        logging.info("Successfully created churn target from churn segment")
        return df
    
    
    #Train test split 
    def get_train_dev_test_sets(self, df:pd.DataFrame):
        print ("Train, dev and test Spliting .......................................................")
        df_train, df_dev = train_test_split(df, train_size = 0.89, random_state = 42, shuffle = True, stratify = df["churn"] )
        df_dev, df_test  = train_test_split(df_dev, train_size = 0.7, random_state = 42, shuffle = True, stratify = df_dev["churn"])

        #Take only a sample of test and train data
        n_train_set = 440000
        n_dev_set = 30000
        n_test_set = 10000
        print ("Taking samples from train, dev and test sets .......................................................")
        df_train = df_train.sample(n=min(n_train_set, len(df_train)), random_state=42)
        df_dev = df_dev.sample(n= min(n_dev_set, len(df_dev)), random_state=42)
        df_test = df_test.sample(n=min(n_test_set, len(df_test)), random_state=42)

        print (f"df_train shape :{df_train.shape} .......................................................")
        print (f"df_dev shape: {df_dev.shape} .......................................................")
        print (f"df_test shape: {df_test.shape} .......................................................")

        logging.info("Succefully splited data into train dev and test sets")
        logging.info("Selected samples from train dev and test sets")
        logging.info(f"df_train shape :{df_train.shape}")
        logging.info(f"df_dev shape: {df_dev.shape}")
        logging.info(f"df_test shape: {df_test.shape}")

        return df_train, df_dev, df_test 
    
    def run_data_transformation(self):
        """
        Run transformation pipeline

        Returns : df_train, df_dev, df_test
        """
        df = self.sample_data_by_churn_segment()       # TODO: we need to sample data by churn segment only in training pipeline
        df = self.get_churn_target_from_churn_segment(df)
        df_train, df_dev, df_test = self.get_train_dev_test_sets(df)
        #save_train_dev_test_sets(df_train, df_dev, df_test)    #For now don't save at this point
        #!!!!! Delete some non churners from the training data for better training
        nbr_of_non_churners_to_drop = 80000            # NOTE: This part will drop the given number on non churners from training data
        print(f"Drop {nbr_of_non_churners_to_drop} rows of non_churners from df_train .......................................................")
        indexes_to_be_deleted = df_train[df_train['churn'] == 0].index[:nbr_of_non_churners_to_drop]
        indexes = df_train.index.difference (indexes_to_be_deleted)
        df_train = df_train.loc[indexes]
        logging.info(f"Drop {nbr_of_non_churners_to_drop} rows of non_churners from df_train")
        return df_train, df_dev, df_test
#END OF CLASS

class FeatureSelection:
    def __init__(self):
        pass
    
    def select_train_features(self, df, feature_names_file = "2024-10-16_feature_names_iter1.txt"):
        """
        Select only train features from df_train, df_dev, df_test
        Returns df_train, df_dev, df_test
        """
        with open(f"{ressources_path}/{feature_names_file}") as f:
            feature_names = f.read()
        # Séparer les colonnes en utilisant la virgule comme délimiteur
        feature_names_iter1 = feature_names.split(',')
        feature_names_iter1 = [col.strip() for col in feature_names_iter1]
        df = df[feature_names_iter1]
        return df
    
    def select_inference_features(self, df):
        """
        Select inference features from df
        Returns df
        """
        print ("Selecting inference featrues from df .......................................................")
        with open("models/ressources/2024-10-25_inference_feature_names.txt", "r") as f:
            feature_names = ast.literal_eval(f.read())
        df_new = df[feature_names]
        logging.info("Selected inference features from df")
        return df_new
        
    def run_feature_selection(self, df_train, df_dev, df_test):
        """
        Select only iter1 features from df_train, df_dev, df_test
        Returns df_train, df_dev, df_test
        """
        print ("Selecting iter 1 features .......................................................")
        df_train = self.select_train_features(df_train)
        df_dev = self.select_train_features(df_dev)
        df_test = self.select_train_features(df_test)
        print (f"df_train shape: {df_train.shape} .......................................................")
        print (f"df_dev shape: {df_dev.shape} .......................................................")
        print (f"df_test shape: {df_test.shape} .......................................................")
        logging.info("Successfully selected iter 1 features")
        return df_train, df_dev, df_test

    
class HandlingMissingValues:
    def __init__(self, df_train, df_dev, df_test):
        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test
    """
    #Need this function only when data is not passed as inputs want to load data 
    def load_train_dev_test(self, data_date):
        "
        Loading train dev test sets of date
        "
        print ("Loading train dev and test sets, ensuring dn is a string type for handling missing values")
        df_train = pd.read_csv(f"{train_dev_test_path}/{data_date}_df_train.csv", index_col = 0, dtype= {"dn": "string"}) 
        df_dev = pd.read_csv(f"{train_dev_test_path}/{data_date}_df_dev.csv", index_col = 0, dtype= {"dn": "string"})
        df_test = pd.read_csv(f"{train_dev_test_path}/{data_date}_df_test.csv", index_col = 0, dtype= {"dn": "string"})
        print (f"df_train shape :{df_train.shape}")
        print (f"df_dev shape: {df_dev.shape}")
        print (f"df_test shape: {df_test.shape}")
        logging.info("Successfully loaded train dev and test sets for handling missing values")
        return df_train, df_dev, df_test
    """    
    def drop_columns_and_rows_with_all_values_null(self, threshold=97):
        """
        Drop columns and rows where nan values percentage is bigger than thereshold
        """
        """        #Drop columns with all values null                                  #THIS PART IS ON COMMENT BECAUSE NULL COLUMNS WILL BE TRAITED IN FEATURE SELECTION
        print ("Deleting all df_train null columns")
        df_train = drop_df_null_columns(self.df_train, threshold = threshold)
        #Take same columns as df_train
        print ("Taking same columns as df_train in df_dev and df_test")
        df_dev = self.df_dev[df_train.columns]
        df_test = self.df_test[df_train.columns]
        logging.info("Droped all columns with all values nulles")"""

        print ("Deleting all null rows from train dev and test sets")
        df_train_T = self.df_train.T            
        df_train_T=drop_df_null_columns(df_train_T, threshold=threshold)
        df_train = df_train_T.T
        print ("Drop all null rows of df_dev")
        df_dev_T = self.df_dev.T                 
        df_dev_T=drop_df_null_columns(df_dev_T, threshold=threshold)
        df_dev = df_dev_T.T
        print ("Drop all null rows of df_test")
        df_test_T = self.df_test.T           
        df_test_T=drop_df_null_columns(df_test_T, threshold=threshold)
        df_test = df_test_T.T
        logging.info("Droped all rows with all values nulles")

        print(f"df_train shape : {df_train.shape} .......................................................")
        print(f"df_dev shape : {df_dev.shape} .......................................................")
        print(f"df_test shape : {df_test.shape} .......................................................")

        return df_train, df_dev, df_test
    
    def replace_0_values_with_nan(self, df_train, df_dev, df_test):
        #Get numerical columns from df
        df_numerical_columns = df_train.dtypes[df_train.dtypes != "object" ].index.to_list()
        df_numerical_columns = [ column for column in df_numerical_columns if column not in ["dn", "dn_group_id", "churn" ]]
        
        print ("Replacing 0 values with nan, in df_train, df_dev and df_test .......................................................")
        df_train[df_numerical_columns] = df_train[df_numerical_columns].replace(0, np.nan)
        df_dev[df_numerical_columns] = df_dev[df_numerical_columns].replace(0, np.nan)
        df_test[df_numerical_columns] = df_test[df_numerical_columns].replace(0, np.nan)
        logging.info("Successfully replaced 0 values with nan, in df_train, df_dev and df_test")
        print (f"Are still there any 0 values in df_train after transforming 0 values into nan : {(df_train[df_numerical_columns] == 0).any().any()}...... ")

        return df_train, df_dev, df_test 
    
    def run_handling_missing_values(self, save_final_data = False):
        """
        Returns df_train, df_dev and df_test after applying these steps:
        - Replace all 0 values with nan
        - Drop all columns and rows with all values null 
        - Replace all nan values with 0
        - Save dataframes if save_final_data is True
        Parameters:
        -----------
        save_final_data : Boolean, save or not the final output data
        """
        #df_train, df_dev, df_test = self.load_train_dev_test()
        #df_train, df_dev, df_test = FeatureSelection(df_train, df_dev, df_test).select_features()
        df_train, df_dev, df_test = self.drop_columns_and_rows_with_all_values_null()
        df_train, df_dev, df_test = self.replace_0_values_with_nan(df_train, df_dev, df_test)
        print ("filling all NAN with 0, in train dev and test sets .......................................................")
        df_train = df_train.fillna(0)
        df_dev = df_dev.fillna(0)
        df_test = df_test.fillna(0)
        logging.info("Filled all NAN values with 0, in train dev and test sets")
        print (f"Total number of missing values in df_train after filling all nan with 0 is : {df_train.isna().sum().sum()}...........................")
        if save_final_data == True:
            save_train_dev_test_sets(df_train, df_dev, df_test, name_sufix="_fillna_0")
        return df_train, df_dev, df_test
#END OF CLASS

class HandlingDuplicatedValues:
    def __init__(self):
        pass

    def drop_duplicated_values(self, df):
        """
        Declare number of duplicates in df and drop theme

        returns df 
        """
        principal_feature_names = [col for col in df.columns if col not in ['dn', 'churn_segment','churn_date', 'activation_bscs_date','id_date']]
        print ("""Dropped columns : ['dn','churn_segment','churn_date', 'activation_bscs_date','id_date'] from df""")
        df_new = df[principal_feature_names]
        logging.info("""Dropped columns : ['dn','churn_segment','churn_date', 'activation_bscs_date','id_date'] from df""")
        print (f"Number of duplicates in df {df_new.duplicated().sum()}")
        print ("Dropping duplcated values")
        df_new = df_new.drop_duplicates()
        logging.info("Dropped all duplicated values successfully")
        return df_new
    
    def run_handling_duplicates(self, df_train, df_dev, df_test):
        """
        Drop all duplicated values in train, dev and test sets

        Returns : df_train, df_dev, df_test
        """
        print ("df_train")
        df_train = self.drop_duplicated_values(df_train)
        print ("df_dev")
        df_dev = self.drop_duplicated_values(df_dev)
        print ("df_test")
        df_test = self.drop_duplicated_values(df_test)
        logging.info("Droped duplicated values in train, dev and test sets successfully")
        return df_train, df_dev, df_test
#END OF CLASS

class FeatureEncoding:
    def __init__(self):
        pass

    def gamme_encoding(self, df):
        """
        Returns dataframe with a new column gamme_encoded
        Parameters:
        -----------
        df: should contain "gamme" feature
        """
        print ("Encoding gamme feature .......................................................")
        gamme_mapping = {
            "Forfaits 49 dhs": 1,
            "Forfaits 99 dhs": 2,
            "Forfaits Hors 99 dhs": 3
        }
        df["gamme_encoded"] = df["gamme"].map(gamme_mapping).fillna(0).astype(int)
        logging.info("Encoded gamme feature successfully")
        return df
    
    def run_feature_encoding (self, df_train, df_dev, df_test):
        """
        Encode features
        Returns df_train, df_dev and df_test
        """
        print (f"df_train.......................................................")
        df_train = self.gamme_encoding(df_train)
        print (f"df_dev .......................................................")
        df_dev = self.gamme_encoding(df_dev)
        print (f"df_test.......................................................")
        df_test = self.gamme_encoding(df_test)
        logging.info("Encoded gamme to gamme_encoded successfully")
        return df_train, df_dev, df_test
#END OF CLASS

class DataSplitting:
    def __init__(self):
        pass

    def get_x_y_data(self, df):
        """
        Returns x_data and y_data from df
        """
        #Get features and target variable
        features, target = [col for col in df.columns if col not in ['dn',"gamme", 'churn_segment','churn_date', 'activation_bscs_date','id_date', 'churn']], ["churn"]
        x, y = df[features], df[target]
        return x, y 
    
    def run_data_splitting(self, df_train, df_dev, df_test):
        """
        Split data and returns
        Returns x_train, y_train, x_dev, y_dev, x_test, y_test
        """
        print ("Splitting data to x and y.......................................................")
        x_train, y_train = self.get_x_y_data(df_train)
        x_dev, y_dev = self.get_x_y_data(df_dev)
        x_test, y_test = self.get_x_y_data(df_test)
        logging.info ("Splitted data to x and y")
        print (f"x_train shape is: {x_train.shape}.......................................................")
        print (f"x_dev shape is: {x_dev.shape}.......................................................")
        print (f"x_test shape is: {x_test.shape}.......................................................")
        return x_train, y_train, x_dev, y_dev, x_test, y_test
#END OF CLASS

class DataNormalization:
    def __init__(self):
        pass
    
    def normalize_data(self, df):
        """
        Normalize dataframe using stored standard scaler
        """
        print ("Normalizing data .......................................................")
        #load normalizer
        with open("models/processors/2024-10-22_standard_scaler.pkl", "rb") as f:
            standard_scaler = pickle.load(f)  
        #Normalize
        #Transform data sets
        df_norm = standard_scaler.transform(df)
        df_norm = pd.DataFrame(df_norm, columns = df.columns)
        logging.info("Normalized data")
        return df_norm

    def run_data_normalization(self, x_train, x_dev, x_test, save_final_data = True):
        """
        Run normalization on x_train, x_dev and x_test

        returns : x_train_norm, x_dev_norm, x_test_norm
        Parameters:
        -----------
        x_train, x_dev, x_test : dataframes
        save_final_data : Boolean, save or not the final output data
        """
        print ("x_train.......................................................")
        x_train_norm = self.normalize_data(x_train)
        print ("x_dev.......................................................")
        x_dev_norm = self.normalize_data(x_dev)
        print ("x_test.......................................................")
        x_test_norm = self.normalize_data(x_test)
        logging.info("Successfully normalized x_train, x_dev, x_test")
        if save_final_data == True:
            print ("Saving x_train_norm.......................................................")
            x_train_norm.to_csv(f"{x_y_sets_path}/{date_time}_x_train_norm.csv", index=True)
            print ("Saving x_dev_norm.......................................................")
            x_dev_norm.to_csv(f"{x_y_sets_path}/{date_time}_x_dev_norm.csv", index=True)
            print ("Saving x_test_norm.......................................................")
            x_test_norm.to_csv(f"{x_y_sets_path}/{date_time}_x_test_norm.csv", index=True)
            logging.info("Successfully saved x_train_norm, x_dev_norm and x_test_norm")
        return x_train_norm, x_dev_norm, x_test_norm
#END OF CLASS       


############################################ Helper functions #####################################################
# TODO : THESE FUNCTIONS MUST PASS TO src/utils.py LATER
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
    #Drop null columns from df
    df.drop(null_columns_to_be_deleted, axis = "columns", inplace = True)
    print (f"DataFrame new shape {df.shape}.......................................................")
    return df
#######################################################################################################################

def run_training_data_processing_pipeline(df, save_final_data=True):
    """
    Applies these steps on df:
    - Data Transformation
    - Feature Selection
    - Handling Missing Values
    - Feature Encoding
    - Data Splitting
    - Data Normalization
    - Save data 
    Returns : x_train_norm, y_train, x_dev_norm, y_dev, x_test_norm, y_test 
    """
    logging.info("############################# Running training data processing pipeline #############################")
    df_train, df_dev, df_test = DataTransformation(df).run_data_transformation()
    df_train, df_dev, df_test = FeatureSelection().run_feature_selection(df_train, df_dev, df_test)       # TODO: Undash this part after experiments          
    df_train, df_dev, df_test = HandlingMissingValues(df_train, df_dev, df_test).run_handling_missing_values(save_final_data = False)
    df_train, df_dev, df_test = HandlingDuplicatedValues().run_handling_duplicates(df_train, df_dev, df_test)
    df_train, df_dev, df_test = FeatureEncoding().run_feature_encoding(df_train, df_dev, df_test)
    x_train, y_train, x_dev, y_dev, x_test, y_test = DataSplitting().run_data_splitting(df_train, df_dev, df_test)
    print ("Saving y_train.......................................................")
    y_train.to_csv(f"{x_y_sets_path}/{date_time}_y_train.csv", index=True)
    print ("Saving y_dev.......................................................")
    y_dev.to_csv(f"{x_y_sets_path}/{date_time}_y_dev.csv", index=True)
    print ("Saving y_test.......................................................")
    y_test.to_csv(f"{x_y_sets_path}/{date_time}_y_test.csv", index=True)
    logging.info("Saved y_train, y_dev and y_test")

    nbr_rows_with_only_zeros = (x_train == 0).all(axis=1).sum()
    print (f"Nbr of rows with only zeros is : {nbr_rows_with_only_zeros}")
    logging.info(f"Nbr of rows with only zeros is : {nbr_rows_with_only_zeros}")

    x_train_norm, x_dev_norm, x_test_norm = DataNormalization().run_data_normalization(x_train, x_dev, x_test, save_final_data=save_final_data) # TODO: Undash this part after experiments 
    return x_train_norm, y_train, x_dev_norm, y_dev, x_test_norm, y_test           # TODO: Undash this part after experiments 



def run_inference_data_processing_pipeline(df, batch_date):
    """
    Applies these steps on df:
    - Extract list of dns from df
    - Select inference features 
    - Fill all nan values with 0
    - Encode features
    - Normalize data
    - Saved inference data with dns

    Returns df_norm, dns
    """
    logging.info("############################# Running inference data processing pipeline #############################")
    print ("Extracting dns list from df .......................................................")
    dns = df[['dn']]
    logging.info("Extracted dns from df")

    df = FeatureSelection().select_inference_features(df)

    print ("Filling nan values with 0 .......................................................")
    df = df.fillna(0)
    logging.info("Filled nan values with 0")

    print (f"Total number of missing values in df_train after filling all nan with 0 is : {df.isna().sum().sum()} ............................................")
    df = FeatureEncoding().gamme_encoding(df)
    df_norm = DataNormalization().normalize_data(df)
    print ("Saving inference data with dns .......................................................")

    df_norm.to_csv(f"data/inference_data/{batch_date}_x_norm.csv", index=True)
    dns.to_csv(f"data/inference_data/{batch_date}_dns.csv", index = True)
    logging.info("Saved inference data with dns")
    return df_norm, dns


#TODO : Retest run_training_data_processing_pipeline()

    



    

    













    






    




