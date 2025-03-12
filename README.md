# BCPPMCHURN
B2C Postpaid mobile churn project 

## Summary
- [Introduction](#intoduction)
- [Tools](#tools)
- [Project structure](#project-structure)
- [Project capabilities](#instructions)
    - [Prediction pipeline](#prediction-pipeline-inference)
    - [Training pipeline](#training-pipeline-train-a-new-model)
    - [Backtesting](#backtesting)

## Intoduction 
This project is the second part of the B2C Postpaid mobile churn. It consists of the Machine Learning components including :
- EDA
- Data cleansing
- Data processing
- Training pipeline
- Prediction (Inference) pipline
- Backtesting

## Tools
Python version : 3.12.6

## Project structure
The project has the following structure:
```
BCPPMCHURN/
├── .gitignore
├── README.md
├── requirements.txt
├── data/                                         
│   ├── train_dev_test                          #Train, dev and test sets before processing
│   ├── x_y_sets                                #x, y data (train, dev and test)
│   ├── inference_data                          #Inference data, right after inference data processing and ready to be used by the model
│   ├── output_data                             #Scores obtained after applying prediction pipeline on inference data
│   ├── real_data                               #Real observed data, used for backtesting with model predictions (scores). ingested directly from impala 
├── models/
│   ├── ml_models                               #All machine learning models that were trained till now 
│   │   ├── 2024-10-25_xgb_model.pkl            #Main model used in inference   
│   │   ├── 2024-10-28_xgb_model_aucpr.pkl
│   │   ├── 2024-10-25_xgb_model.json
│   │   ├── 2024-10-28_xgb_model_aucpr.json
│   ├── processors                              #Processors used during training and beeing used in inference
│   │   ├── 2024-10-22_standard_scaler.pkl      #Main scaler used in inference
│   ├── ressources                              #Other ressources such as : feature names, learning curves
│   │   ├── 2024-10-25_inference_feature_importance.txt
│   │   ├── 2024-10-30_eval_hist_xgboost1.json
│   │   ├── 2024-10-31_eval_hist_xgboost_aucpr.json
├── notebooks/                                          #EDA, Experiments and Reports notebooks 
│   ├── experiements_notebooks
│   │   ├── 01_ingest_structure_save.ipynb
│   │   ├── 02_handle_duplcates.ipynb
│   │   ├── 03_handle_missing_values_01.ipynb
│   │   ├── 04_handle_outliers.ipynb
│   │   ├── 05_correlations.ipynb
│   │   ├── 06_feature_distribution.ipynb
│   │   ├── 07_data_spliting_x_y.ipynb
│   │   ├── 08_data_distribution.ipynb
│   │   ├── 09_feature_importance.ipynb
│   │   ├── 10_grid_search.ipynb
│   │   ├── 11_model_training.ipynb                     #Main notebook for model training 
│   │   ├── 11_model_training02.ipynb
│   │   ├── 11_model_training_test_20_12_2024.ipynb
│   │   ├── 11_model_training03_autoencoder.ipynb
│   │   ├── 11_model_training04_nn_classifier.ipynb
│   │   ├── 11_model_training05_autoencoder.ipynb
│   │   ├── 11_model_training06_cnns.ipynb
│   │   ├── 12_prediction_pipeline.ipynb                #Main notebook for inference 
│   │   ├── 13_backtesting_20241101.ipynb
│   │   ├── 13_backtesting_20250101.ipynb
│   │   ├── 13_backtesting_20250101_all_mdns.ipynb
│   │   ├── 14_training_pipeline.ipynb
│   ├── reports
│   ├── eda.ipynb
├── src/                                        #Projects main components and pipelines
│   ├── exception.py
│   ├── logger.py
│   ├── logger_class.py
│   ├── import_data.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_structuring.py
│   │   ├── inference_data_processing.py
│   │   ├── training_data_processing.py
│   ├── data_monitoring/
│   ├── eda/
│   │   ├── utils.py
│   ├── pipelines/
│   │   ├── predict_pipeline.py
│   │   ├── training_pipeline.py
│   │   └── __init.__.py
│   ├── ressources/

```

## Instructions 
This project contains the prediction pipeline to extract mdns churn scores for a given date, and the training pipeline to train a new model.
Follow these instructions to use this project:

1. Clone the project
```
git clone git@...
```
2. Create a virtual environnement 
3. Install necessary libraries 
```
pip install -r requirements.txt 
```

### Prediction pipeline (Inference)
We run prediction pipeline the first of each month
To extract mdn scores you can use one of the tow following methods:

1. **prediction_pipeline.py** module (Recommanded)

This module contains the code for the overall prediction pipeline from inference data processing, untill model prediction.
- First: you should ingest inference data from impala (Use jupyter on sandbox to do so, 12_prediction_pipeline.ipynb notebook will help you do that), 
- Second: download the final inference data on your local, path : "data/experiments_data"
- Third: On prediction_pipline.py code, on `"if __name__ == "__main__"`, change the `batch_date` varibale to the current cycle date
- Forth: run this command on your terminal :
```
python3 src/pipelines/prediction_pipline.py
```
`NOTE`: If you're running into this error : src can't be found. Run this command before runing the script : 
 ```export PYTHONPATH=/path/to/bcppmchurn/project```
- This will generate scores and will store them on "data/output_data"

2. **notebook 12_prediction_pipeline.ipynb** 

The notebook will allow you to run the prediction pipeline step by step.

It contains tow part data ingestion (will be done on sandbox dev 3) and data processing (will be done on your local machine).
Before using the notebook you should ingest inference data from impala (Use jupyter on sandbox to do so, use the same notebook), than, download the final inference data on "data/experiments_data".

When you've donwloaded the final inference data comeback to this notebook (on local) and start directly from the cell where you read the data (```df = pd.read_csv(f"data/experiments_data/inference_{batch_date}_final_df.csv")```), you don't need to re-run the ingestion part.
Don't forget to change the batch_date variables to the current batch cycle.

The reason why the first part is done on jupyter sandbox dev 3 and the other is done on local machine is that : many labraries aren't installed on jupyter dandbox dev 3.

This will generate scores and will store them on "data/output_data"

### Training pipeline (Train a new model)
To train a new model use one of the following methods :

1. **Using notebook model_training.ipynb** (Recommanded)

- First: Ingest data from impala using jupyter on sandbox machine
- Second: donwload this data to your local, path : "data/experiments_data"
- Third: copy paste the notebook `11_model_training.ipynb` and give it a different name
- Forth: follow the notebook blocks. Don't forget to change the `data_date` to the appropriate date 

2. **training_pipeline.py**
- First: Ingest data from impala using jupyter on sandbox machine
- Second: donwload this data to your local, path : "data/experiments_data"
- Third: On training_pipeline.py code, on `"if __name__ == "__main__"`, change the `data_date` variable. Replace it with the current data date.
- Forth: run the training pipeline using this command : 
```
python3 src/pipelines/training_pipeline.py
```
- The output of this pipeline is the following :
    - x_train, y_train, x_dev, y_dev, x_test, y_test will be stored in "data/x_y_sets" after training data processing. 
    - The final model will be stored in "models/ml_models"
    - Visualizations of model evaluation such as : training curve, confusion matrix, roc curve ...

### Backtesting
Use backtesting to compare model scores (extracted the first of every month), with real data.
We do backtesting on observed data of the 18th or 30th of the month (You can do other dates).

Simply copy past the `12_backtesting_20250201.ipynb` notebook, modify its name to 12_backtestin_{date}. Change the `data_date` variable, than run the notebook blocks.


