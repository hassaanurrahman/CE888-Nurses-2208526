import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,timedelta
from mpl_toolkits.mplot3d import Axes3D
import os
import lightgbm as lgb
from sklearn.utils import resample

#function for preprocessing data
def preProcessingData(file, filename):
    #accelerometer data file
    if filename == "ACC":
        #Rename columns for x,y,z
        file.columns = [filename+"_X", filename+"_Y", filename+"_Z"]
        #get the start timestamp and frequency
        start_timestamp = file.loc[0, filename+"_X"]
        freq = file[filename+"_X"].iloc[1]
        #remove the first two rows
        file = file[2:]
        file = file.reset_index(drop=True)
        #calculate the datetime column
        file["Datetime"] = datetime.fromtimestamp(start_timestamp)
        start_date = file["Datetime"].iloc[0] - timedelta(seconds=1/freq)
        end_date = file["Datetime"].iloc[0] + timedelta(seconds=(len(file)-2)*(1/freq))
        file["Datetime"] = pd.date_range(start=start_date, end=end_date, freq=f'{1/freq}S') + pd.Timedelta(seconds=1/freq)
        #if the frequency is greater than 4Hz then resample the data
        if freq > 4:
            file.index = file["Datetime"]
            file = file.drop("Datetime", axis=1)
            file[filename+"_X"] = file[filename+"_X"].resample('0.25S').mean()
            file[filename+"_Y"] = file[filename+"_Y"].resample('0.25S').mean()
            file[filename+"_Z"] = file[filename+"_Z"].resample('0.25S').mean()
            file = file.reset_index().rename(columns={"index": "Datetime"})
            file = file.dropna()
    #not an accelerometer data file
    else:
        file.columns = [filename]
        start_timestamp = file.loc[0, filename]
        freq = file[filename].iloc[1]
        #remove the first two rows
        file = file[2:]
        file = file.reset_index(drop=True)
        #calculate the datetime column
        file["Datetime"] = datetime.fromtimestamp(start_timestamp)
        start_date = file["Datetime"].iloc[0] - timedelta(seconds=1/freq)
        end_date = file["Datetime"].iloc[0] + timedelta(seconds=(len(file)-2)*(1/freq))
        file["Datetime"] = pd.date_range(start=start_date, end=end_date, freq=f'{1/freq}S') + pd.Timedelta(seconds=1/freq)
        #if the frequency is greater than 4Hz then resample the data
        if freq > 4:
            file.index = file["Datetime"]
            file = file.drop("Datetime", axis=1)
            file[filename] = file[filename].resample('0.25S').mean()
            file = file.reset_index().rename(columns={"index": "Datetime"})
            file = file.dropna()
    return file

#function for resampling of data
def resampling(df):
    stress = df[df["Stress level"] == 1.0]
    no_stress  = df[df["Stress level"] == 0.0]
    df_downsample = resample(stress,replace=True,n_samples=80,random_state=42)
    df_upsample = resample(no_stress,replace=True,n_samples=80,random_state=42)
    data = pd.concat([df_upsample,df_downsample])
    return data.reset_index(drop=True)