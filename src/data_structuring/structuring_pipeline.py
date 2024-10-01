import pandas as pd
from src.exeption import CustomException
import sys

"""
This class is meant for structuring data 
1. Join stat and trend features
2. Concat vertically different domain features
3. Pivot tables based on pivot + value columns
"""
class structuringPipeline:
    def __init__(self, features_tables_dict: dict):
        """
        features_tables_dict : dictionnarie of features along with their names. key : feature name, value : feature
                                example : {"data_stat_features": data_stat_features, "data_trend_features": data_trend_features}
        """
        self.features_tables_dict = features_tables_dict

    def merge_same_domain_features (self, domain_stat_features, domain_trend_features):
        """
        Outer Join stat and trend features that belong to same domain based on "dn", "pivot", "value", "dn_group_id" 

        parameters:
        ----------
        domain_stat_features  : example : data_stat_features
        domain_trend_features : example : data_trend_features
        """
        return pd.merge(domain_stat_features, domain_trend_features, on = ["dn", "pivot", "value", "dn_group_id"], how= "outer")
    
    def concat_different_domain_features (self, features_list: list):
        """
        Concat vertically features that belong to different domains, but have the same feature types.

        parameters :
        -----------
        feature_list : list of feature tables. Should have the same columns.
                        example: [data_features, voice_features, complaints_features]
        """
        return pd.concat (features_list)
    
    def merge_and_concat_features(self):
        """
        Get as input a dict of table features. It joins same domain features, then concat everything into one dataframe
        """
        try : 
            domain_features_splited = {}
            for key, value in self.features_tables_dict.items():
                #Detect feature domain
                domain = key.split("_")[0]
                #if domain not already in dict create a new key equal to its value, else append value to its key
                if domain not in domain_features_splited.keys():
                    domain_features_splited[domain] = [value]
                else:
                    domain_features_splited[domain].append(value)
            
            domain_features = []
            for key in domain_features_splited.keys():
                domain_table = self.merge_same_domain_features(domain_features_splited[key][0], domain_features_splited[key][1])
                domain_features.append (domain_table)
        
            return self.concat_different_domain_features(domain_features)
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def pivot_table (self, dataframe: pd.DataFrame):
        """
        - Creates a new column "pivot_value" based on pivot + value
        - Pivotes the table based on created column "pivot_value"
        - Example of generated columns: roamstate_nonroaming_duration_nb_1m
        """
        try:
            #Get df numerical columns and Exclued "dn" and "dn_group_id"
            df_numerical_columns = dataframe.dtypes[dataframe.dtypes != "object" ].index.to_list()
            df_numerical_columns = [ column for column in df_numerical_columns if column not in ["dn", "dn_group_id"] ]

            #Create new colun "pivot_value"
            dataframe["pivot_value"] = dataframe["pivot"] + "_" + dataframe["value"]
            #Pivto column
            pivoted_df = dataframe.pivot(index = "dn", columns= "pivot_value", values=df_numerical_columns )
            pivoted_df.columns = [f'{pivot_value}_{col}' for col, pivot_value in pivoted_df.columns]
            return pivoted_df
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def run_structuring_pipeline(self):
        """
        Run all the pipeline to structure data
        1. Takes as input table features
        2. Apply merge_and_concat_features() 
        3. Apply pivot_table()
        """
        df = self.merge_and_concat_features()
        pivoted_df =  self.pivot_table(df)
        return pivoted_df




        


