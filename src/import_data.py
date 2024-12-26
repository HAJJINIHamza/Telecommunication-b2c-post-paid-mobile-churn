import pandas as pd 
import os
from datetime import datetime 
from src.components.data_ingestion import get_feature_tables_from_impala, get_churn_target 
from src.components.data_structuring import structuringPipeline

os.chdir("../")
print ("Current working directory :", os.getcwd())

#Load data
domains =["data", "voice", "complaints", "payement"]
feature_types = ["stat", "trend"]
dn_group_interval = [0, 58]
#Initiate a spark session and get table features 
features_dict = get_feature_tables_from_impala(domains, feature_types, dn_group_interval) 
#Get target table
churners_non_churners = get_churn_target()
#Structure table featues into a dataframe
df = structuringPipeline(features_dict, churners_non_churners).run_structuring_pipeline()


#Saving a sample of final df
data_path = "data/experiments_data"
date_time = datetime.today().strftime("%Y-%m-%d")
print("")
df.to_csv(f"{data_path}/{date_time}_final_df.csv", index = True)


    