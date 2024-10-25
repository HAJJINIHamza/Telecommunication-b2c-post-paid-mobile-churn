import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve

from src.exception import CustomException
import sys
import random

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

        list_of_colors = [   "blue", "green", "red", "cyan", "magenta", "yellow", "black", 
                            "orange", "purple", "brown", "pink", "gray", "olive", "lime", 
                            "teal", "navy", "gold", "indigo", "lightblue", "lightgreen"
                         ]
        color = random.choice(list_of_colors)
        df = missing_values_per_column
        plt.figure(figsize=figsize)
        fig = sns.barplot(x= df.column, y=df["prc_null_values"], color = color)
        fig.set_title("percentage of missing values per column")
        plt.ylim(0, 100)
        plt.xticks(rotation= 315, ha="left")
        plt.show()
    except Exception as e:
        raise CustomException(e, sys)


def vis_target_distribution(target, figsize=(18, 6)):
    """
    This function plot a cercle representing the distribution of the target feature
    target:the target column, example: df["churn_segment"]
    """
    
    #Get churn segments and thier percentages 
    churn_segments = target.value_counts().index
    percentages = [ (value/len(target))*100 for value in target.value_counts().to_list() ]
    #Visualize percentages
    fig, ax1 = plt.subplots(1, figsize=figsize)
    ax1.pie(percentages, labels=churn_segments, autopct='%1.1f%%', startangle=140)
    plt.title("target distribution")
    plt.show()
    

def vis_box_plots(df):
    """
    Retruns box plots of df features for outliers exploration
    """
    columns = df.columns
    n_rows = (len(columns)//2) + (len(columns)%2)
    fig, axes= plt.subplots(nrows=n_rows, ncols=2, figsize=(20, 2*n_rows) )
    #Add some pading between figues
    plt.tight_layout(pad=4.0)
    #flatten axes                        
    axes = axes.flatten()                        
    #Create plots
    for i_column, column in enumerate(columns):
        sns.boxplot(x= df[column], ax=axes[i_column], color="green" )
        #axes[i_column].set_title(f"{column} boxplot")
    #Remove any empty subplots                        
    if len(columns)%2 != 0:
        for j in range(len(columns), len(axes)):
            fig.delaxes(axes[j])
    #Set title 
    fig.suptitle("Columns boxplot", fontsize=16, y = 1)
    plt.show()

    
def vis_correlations_with_target(df, target_name="churn"):
    """
    Returns a heatmap representing correlations with target variable
    """
    correlation_matrix = df.corr()[[target_name]]
    plt.figure(figsize=(23, 1))
    fig = sns.heatmap(correlation_matrix.T, annot = True, fmt=".2f", cmap="Reds", cbar = True)
    fig.set_title("correlations with target")
    plt.xticks(rotation=315, ha="left")
    plt.show()

    
def get_churn_target_from_churn_segment(df):
    """
    Returns df with new column "churn" 0 if not churner else 1,
    from churn_segment variable
    """
    target_list = []
    for churn_segment in df["churn_segment"].to_list():
        if churn_segment == "non_churners":
            target_list.append(0)
        else:
            target_list.append(1)
    df['churn'] = target_list
    return df

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


def vis_missing_values_heatmap(missing_values_matrix):
    """
    Plots a heatmap of missing values per columns and rows
    Parameters:
    -----------
    missing_values_matrix: matrix of missing values (df.isna())
    """
    print("Missing values in white, other values in black")
    plt.figure(figsize = (25, 13))
    sns.heatmap(missing_values_matrix, annot = False)
    plt.title ("Heatmap of missing values")
    plt.ylabel("Row index")
    plt.xticks(rotation = 315, ha="left")
    plt.show()

def vis_tsne_data_distribution(data, target, perplexity = 30, figsize=(15, 10), palette = "coolwarm"):
    """
    Applyes TSNE on data and plots data distribution labeled by target
    Parameters:
    -----------
    target: is a column of dataframe not a name of a column
    """
    print ("Applying tsne on data")
    data_tsne = TSNE(n_components = 2, perplexity = perplexity).fit_transform(data)
    data_tsne = pd.DataFrame(data_tsne, columns = ["x1", "x2"])
    data_tsne["churn"] = target
    #Plot
    print ("Ploting data distribution")
    plt.figure(figsize=figsize)
    sns.scatterplot(data = data_tsne, x="x1", y="x2", hue = "churn", palette=palette)
    plt.title(f"Perplexity {perplexity} TSNE data distribution")
    plt.show()
    
    
def vis_feature_importance(columns_names, importance, columns_families):
    """
    Returns a plot of feature importance per feature, by feature families
    Parameters:
    -----------
    columns_names: list of columns names that were given to model to extract the importances, same shape as importance
    importance : list of feature importances extracted from the mode
    columns_families : dictionnary of columns families, where key is family name and value is list of columns in this family
    
    """
    #Transform feature importance to dataframe
    feature_importance = {"column": columns_names, "importance" : importance }
    feature_importance = pd.DataFrame(feature_importance)
    feature_importance = feature_importance.T
    feature_importance.columns = feature_importance.iloc[0]
    feature_importance.drop("column", inplace = True)
    #Plot
    for key, value in columns_families.items():
        plt.figure(figsize = (22, 8))
        sns.barplot(x = value, y = feature_importance[value].values.tolist()[0])
        plt.ylabel("feature importance")
        plt.title(f"Family : {key}, Feature importance")
        plt.xticks(rotation= 315, ha="left")
        plt.ylim(0, importance.max()+0.002)
        plt.show()


def vis_feature_densities(df, target):
    """
    Returns a plot of df feature densities labeled with target
    Parameters:
    -----------
    df: DataFrame 
    target : DF Column, not column name. Example (df["churn"])
    """
    columns = df.columns
    nrows = len(columns)//3 + len(columns)%3

    fig, axes = plt.subplots(nrows= nrows, ncols=3, figsize=(5*3, 3*nrows))
    plt.tight_layout()
    axes = axes.flatten()
    for i_column, column in enumerate(columns):
        #sns.displot(df_train, x = column, hue="churn", kind="kde", multiple="stack", height= 4, aspect=1)
        sns.kdeplot(x = df[column], hue=target,ax = axes[i_column], multiple="stack")
    if len(columns)%3 != 0:
        for j in range(len(columns), len(axes)):
            fig.delaxes(axes[j])

    fig.suptitle("Feature density")
    plt.show()

def plot_feature_importance(importance,names,model_type, max_n_features = None):
    """
    Plot feature importance in descending order
    """
    
    if max_n_features == None:
        max_n_features = len(importance)

    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(20,80))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'][0:max_n_features], y=fi_df['feature_names'][0:max_n_features])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


def report_model_performances(y_train, y_train_predicted,y_test, y_test_predicted, model_name=""):
    """
    Reports a model performance : accuracy, precision, recall and f1 score
    """
    print ("                  train set      ||     test set")
    print ("------------------------------------------------------------")
    print (f"{model_name} accuracy    :", accuracy_score(y_train, y_train_predicted), " || ", accuracy_score(y_test, y_test_predicted) )
    print (f"{model_name} precision   :", precision_score(y_train, y_train_predicted)," || ", precision_score(y_test, y_test_predicted))
    print (f"{model_name} recall      :", recall_score(y_train, y_train_predicted),   " || ", recall_score(y_test, y_test_predicted))
    print (f"{model_name} f1 score    :", f1_score(y_train, y_train_predicted),       " || ", f1_score(y_test, y_test_predicted))
    print (f"-------------------------------------------------------------")

    matrice_confusion = confusion_matrix(y_test, y_test_predicted)
    plt.figure(figsize=(4, 3))
    sns.heatmap(matrice_confusion/np.sum(matrice_confusion), annot=True, cmap="Blues", fmt=".2%")
    plt.title(f"Confusion matrix of model {model_name} on test data")
    plt.show()


def vis_calibration_curve (n_bins, y_test, y_test_predicted_prob):
    """
    Plots calibration of the model, meaning fraction of positives per mean predicted porbabilities
    Parameters:
    -----------
    n_bins : should be a list of int values
    """
    nrows = len(n_bins)//2 + len(n_bins)%2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 5*nrows))
    axes = axes.flatten()
    for i, bins in enumerate(n_bins):
        # Calculate the calibration curve for each bin size
        fraction_of_positives, mean_predicted_probabilities = calibration_curve(y_test, y_test_predicted_prob, n_bins=bins)
        
        # Plot the calibration curve on the respective axis
        axes[i].plot(mean_predicted_probabilities, fraction_of_positives, marker='o', label=f'XGBoost (n_bins={bins})')
        axes[i].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        
        # Set labels and title for each subplot
        axes[i].set_xlabel('Mean Predicted Probability')
        axes[i].set_ylabel('Fraction of Positives')
        axes[i].set_title(f'Calibration Curve (n_bins={bins})')
        axes[i].legend()

    plt.tight_layout()
    plt.show()

def vis_roc_curve (y_test, y_test_predicted_prob):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_predicted_prob)

    # Calculate AUC (Area Under Curve) for reference
    auc_score = roc_auc_score(y_test, y_test_predicted_prob)

    # Plotting
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='Red', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random classifier")  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    

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
    
    #END OF CLASS
        
        


        