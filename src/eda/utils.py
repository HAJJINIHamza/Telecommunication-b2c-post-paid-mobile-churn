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
                                                "prc_null_values": ((df.isna().sum()/len(df))*100).to_list() }
                                                )
        
        df = missing_values_per_column
        plt.figure(figsize=figsize)
        fig = sns.barplot(x= df.column, y=df["prc_null_values"], color = "#FA5656")
        fig.set_title("percentage of missing values per column")
        plt.xticks(rotation= 315, ha="left")
        plt.show()
    except Exception as e:
        raise CustomException(e, sys)
        

class columnsFamilies:
    
    def __init__(self, df):
        self.df = df

    def get_df_columns_startswith_prefrix(self, prefix: str):
        """
        Retruns df columns that starts with a prefix
        Meant for feature famillies like usagespecification_localdataservice
        """
        return [col for col in self.df.columns if col.startswith(prefix)]

    def get_df_columns_ends_with_sufix(self, sufix: str):
        """
        Retruns df columns that ends with a sufix 
        Meant for features that occured in a certain month like : "nb_3m"
        """
        return [col for col in self.df.columns if col.endswith(sufix)]

    def get_list_of_df_columns_prefix(self):
        """
        Retruns a list of df columns prefixes to choose from 
        in order to build feature famillies
        """
        try:
            sufix = "nb_1m"
            list_of_cols = self.get_df_columns_ends_with_sufix(sufix)
            return [col.replace("_nb_1m", "") for col in list_of_cols]
        except Exception as e:
            raise CustomException(e, sys) 

    def get_columns_families(self):
        """
        Return dictionnarie where values are column familie name (prefix)
        and values are df.columns that belongs to this familiy
        """
        try:
            #Get columns families by prefix
            prefix_list = self.get_list_of_df_columns_prefix()
            #Get columns families
            columns_families = {prefix: self.get_df_columns_startswith_prefrix(prefix) for prefix in prefix_list}
            return columns_families
        
        except Exception as e:
            raise CustomException(e, sys)

        