import yaml
import os 
from pyspark.sql import SparkSession
from src.exception import CustomException


#Load spark config yaml file
config_path = "sparkConfig.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

#Env variables
os.environ["SPARK_HOME"] = config['spark_bcppmchurn']['spark_home']
os.environ["PYSPARK_PYTHON"] = config['spark_bcppmchurn']['python_path']
os.environ["PYSPARK_DRIVER_PYTHON"] = config['spark_bcppmchurn']['python_path']


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
        raise customException(e, sys)
        
        
def get_tables_from_impala(domains:list, feature_types:list):
    """
    Loads data from impala and Returns a dataframe containing data from domains and feature_types
    parameters:
    ----------
    domains: could be data, voice, complaints, ...
    features_types : either "stat" or "trend"
    """
    #Initiate a spark session
    print ("Initiating spark session ............................................................................")
    spark = get_spark_session()
    
    #Feature tables names 
    data_stat_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_data_stat_features"
    data_trend_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_data_trend_features"
    voice_stat_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_voice_stat_features"
    voice_trend_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_voice_trend_features"
    complaints_stat_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_complaints_stat_features"
    complaints_trend_features_name = "tel_test_dtddds.dev_bcppmchurn_learning_complaints_trend_features"
    
    #Feature names dictinnarie
    feature_names_dict = { "data": {"stat": data_stat_features_name, "trend": data_trend_features_name},
                "voice": {"stat": voice_stat_features_name, "trend": voice_trend_features_name},
                "complaints": {"stat": complaints_stat_features_name, "trend": complaints_trend_features_name}}
                          
    #loop over domains and feature_types, get table_name, create a query and load tables
    table_names = []
    feature_dict = {}
    for domain in domains:
        for feature_type in feature_types:
            table_name = feature_names_dict[domain][feature_type]
            QUERY = f"SELECT * FROM {table_name} LIMIT 100" #Should delete the LIMIT 100
            print (f"Loading {table_name} ..................................")
            data = spark.sql(QUERY).toPandas()
            print (f"{table_name} shape is: {data.shape} ................................")
            feature_dict[domain] = {feature_type : data}
    return feature_dict