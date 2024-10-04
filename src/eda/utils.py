import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from src.exception import CustomException
import sys

def vis_perc_missing_values_per_column(df: DataFrame, figsize=(15,4)):
    """
    Visualize the number of missing values per columns.
    Takes as input a DataFrame or a sample of DataFrame
    """
    try:
        missing_values_per_column = pd.DataFrame (
                                                { "column": ((df.isna().sum()/len(df))*100).index, 
                                                "nbr_null_values": ((df.isna().sum()/len(df))*100).to_list() }
                                                )
        
        df = missing_values_per_column
        plt.figure(figsize=figsize)
        fig = sns.barplot(x= df.column, y=df.nbr_null_values)
        fig.set_title("nbr of missing values per column")
        plt.xticks(rotation= 315, ha="left")
        plt.show()
    except Exception as e:
        raise CustomException(e, sys)
        

        
def get_df_columns_startswith_prefrix(df, prefix: str):
    """
    Retruns df columns that starts with a prefix
    Meant for feature famillies like usagespecification_localdataservice
    """
    return [col for col in df.columns if col.startswith(prefix)]

def get_df_columns_ends_with_sufix(df, sufix: str):
    """
    Retruns df columns that ends with a sufix 
    Meant for features that occured in a certain month like : "nb_3m"
    """
    return [col for col in df.columns if col.endswith(sufix)]

def get_list_of_df_columns_prefix(df):
    """
    Retruns a list of df columns prefixes to choose from 
    in order to build feature famillies
    """
    try:
        sufix = "nb_1m"
        list_of_cols = get_df_columns_ends_with_sufix(df, sufix)
        return [col.replace("_nb_1m", "") for col in list_of_cols]
    except Exception as e:
        raise CustomException(e, sys) 

        