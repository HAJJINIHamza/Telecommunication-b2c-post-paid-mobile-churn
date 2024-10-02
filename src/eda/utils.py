import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_missing_values_per_column(df: DataFrame, figsize=(15,4)):
    """
    Visualize the number of missing values per columns.
    Takes as input a DataFrame or a sample of DataFrame
    """
    missing_values_per_column = pd.DataFrame (
                                            { "column": df.isna().sum().index, 
                                               "nbr_null_values": df.isna().sum().to_list() }
                                               )
    
    df_sample = missing_values_per_column
    plt.figure(figsize=figsize)
    fig = sns.barplot(x= df_sample.column, y=df_sample.nbr_null_values)
    fig.set_title("nbr of missing values per column")
    plt.xticks(rotation= 315, ha="left")
    plt.show()
        
