import yaml
import os 
import sys
from pyspark.sql import SparkSession
import subprocess

from src.exception import CustomException
from src.logger import logging

#Load spark config yaml file
config_path = "sparkConfig.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

#Env variables
os.environ["SPARK_HOME"] = config['spark_bcppmchurn']['spark_home']
os.environ["PYSPARK_PYTHON"] = config['spark_bcppmchurn']['python_path']
os.environ["PYSPARK_DRIVER_PYTHON"] = config['spark_bcppmchurn']['python_path']

def get_kinit():
    try:
        os.chdir("..")
        command = ["kinit", "hamza_hajjini", "-kt", "hamza_hajjini.keytab"]
        subprocess.run(command, check=True )
        os.chdir("bcppmchurn")
        print ("Obtained kerberos ticket succeffully")
        logging.info ("Obtained kerberos ticket succeffully")
    except Exception as e:
        print ("Couldn't obtain kerberos ticket")
        raise CustomException(e, sys)

def get_spark_session(app_name = config['spark_bcppmchurn']['app_name']):
    try:
        return SparkSession.builder \
            .master(config['spark_bcppmchurn']['master']) \
            .appName(app_name) \
            .enableHiveSupport() \
            .config("spark.submit.deployMode", config['spark_bcppmchurn']['deploy_mode']) \
            .config("spark.yarn.appMasterEnv.PYSPARK_PYTHON", config['spark_bcppmchurn']['python_path']) \
            .config("spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON", config['spark_bcppmchurn']['python_path']) \
            .config("spark.driver.memory", config['spark_bcppmchurn']['driver_memory']) \
            .config("spark.executor.memory", config['spark_bcppmchurn']['executor_memory']) \
            .config("spark.yarn.queue", config['spark_bcppmchurn']['queue']) \
            .getOrCreate()
    except Exception as e:
        raise CustomException(e, sys)
        
    
        
def get_feature_tables_from_impala(domains:list, feature_types:list, dn_group_interval: list):
    """
    Loads data from impala and Returns a dataframe containing data from domains and feature_types
    parameters:
    ----------
    domains: could be data, voice, complaints, payement.
    features_types : either "stat" or "trend"
    dn_group_intervall : example [0, 10] -> get table where dn_group_id is between 0 and 10
    """
    
    #Initiate a spark session
    print("Getting Kerberos ticket .............................................................................")
    logging.info ("Getting Kerberos ticket")
    get_kinit()
    print ("Initiating spark session ............................................................................")
    logging.info("Initiate spark session")
    spark = get_spark_session()
    
    #Feature tables names 
    data_stat_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_data_stat_features"
    data_trend_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_data_trend_features"
    voice_stat_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_voice_stat_features"
    voice_trend_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_voice_trend_features"
    complaints_stat_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_complaints_stat_features"
    complaints_trend_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_complaints_trend_features"
    payement_stat_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_payment_stat_features"
    payement_trend_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_payment_trend_features"
    
    #Feature names dictinnarie
    feature_names_dict = {"data": {"stat": data_stat_features_name, "trend": data_trend_features_name},
                          "voice": {"stat": voice_stat_features_name, "trend": voice_trend_features_name},
                          "complaints": {"stat": complaints_stat_features_name, "trend": complaints_trend_features_name},
                          "payement": {"stat": payement_stat_features_name, "trend": payement_trend_features_name}
                         }
                          
    #loop over domains and feature_types, get table_name, create a query and load tables
    table_names = []
    feature_dict = {}
    logging.info("Starting ingestion")
    for domain in domains:
        for feature_type in feature_types:
            table_name = feature_names_dict[domain][feature_type]
            QUERY = f"SELECT * FROM {table_name} WHERE dn_group_id BETWEEN {dn_group_interval[0]} AND {dn_group_interval[1]}" 
            print (f"Loading {table_name} ..................................")
            data = spark.sql(QUERY).toPandas()
            logging.info(f"table {table_name} succefully loaded")
            print (f"{table_name} shape is: {data.shape} ................................")
            #insure dn is a string
            data["dn"] = data["dn"].astype("string")
            data["dn_group_id"] = data["dn_group_id"].astype("string")
            #Add data to feature_dict
            if domain not in feature_dict.keys():
                feature_dict[domain] = {feature_type : data}
            else : 
                feature_dict[domain][feature_type]=data
    logging.info("Ingestion completed")
    return feature_dict

def get_churn_target():
    
    """
    Loads target table from impala : dev_bcppmchurn_target_table_pc3_20240607
    """
    
    #Initiate a spark session
    print("Getting Kerberos ticket .............................................................................")
    logging.info ("Getting Kerberos ticket")
    get_kinit()
    print ("Initiating spark session ............................................................................")
    logging.info("Initiate spark session")
    spark = get_spark_session()
    
    try:
        #Loading target table  
        table_name = "tel_test_dtddds.dev_bcppmchurn_target_table_pc3_20240607"    #This table name could change in the future
        QUERY = f"SELECT * FROM {table_name}"    #TODO HAMZA: DELETE LIMIT 100
        print(f"Loading target_table : {table_name}")
        churners_non_churners = spark.sql(QUERY)
        logging.info(f"Target table: {table_name} succefully loaded")
        churners_non_churners = churners_non_churners.toPandas()
        print (f"{table_name} shape is: {churners_non_churners.shape}")
        churners_non_churners.rename(columns = {"mdn": "dn"}, inplace = True)
        churners_non_churners["dn"] = churners_non_churners["dn"].astype("string")
        return churners_non_churners
    
    except Exception as e:
        print ("Couldn't load target table")
        raise CustomException(e, sys)

