# Import required libraries

import pandas as pd
import numpy as np
import math as mt
from datetime import date, datetime, timedelta
import scipy.stats as ss
import os
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Normalizer, StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, RandomForestRegressor
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import *
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from dtreeviz.trees import dtreeviz


def load_and_fit_model():
    start_time = time.time()
    loading_phase = 'Initiating Loading'
    print(loading_phase)

    # Get cwd as this changes depending on which laptop is being used.
    # Could probably use a relative reference but this may restrict development.
    # Import all required data from CSV files.

    directory = os.getcwd()

    growth_data_path = f"{directory}//growth_db.csv"
    weather_data_path = f"{directory}//weather_db.csv"
    zone_data_path = f"{directory}//zone_db.csv"

    loading_phase = 'Ingesting Data'
    print(loading_phase)
    growth_data = pd.read_csv(growth_data_path)
    weather_data = pd.read_csv(weather_data_path)
    zone_data = pd.read_csv(zone_data_path)


    # Variables used within script.

    linearisation_coef = 0.625 # This is used to transform mean_diameter so it has a linear relationship with solar/heat.
    stripping_coef = 0.92 # This is to allow for a slight reduction in diameter once leeks are stripped at harvest.
    min_grow_temp = 3 # The minimum temperature that leeks will grow at. Needed to calculate heat units.
    max_grow_temp = 27 # The temperature at which maximum growth rate is achieved. Needed to calculate heat units.
    season = datetime(2021, 1, 1) # Output data only required for fields with a planting date in this season.


    loading_phase = 'Standardising Data'
    print(loading_phase)
    # Standardise weather data.

    weather_data['date'] = pd.to_datetime(weather_data['date'], format='%d/%m/%Y')
    weather_data['time'] = pd.to_datetime(weather_data['time'], format='%H:%M:%S')
    weather_data['day'] = weather_data.date.dt.day
    weather_data['month'] = weather_data.date.dt.month
    weather_data['day_month'] = weather_data['day'].astype(str) + " - " + weather_data['month'].astype(str) # This is used to calculate weather averages.

    # Calculate heat_units using min and max temperature variables.

    weather_data['heat_units'] = weather_data['avg_temp'] - min_grow_temp
    weather_data['heat_units'] = np.where((weather_data['heat_units'] < 0), 0, weather_data['heat_units'])
    weather_data['heat_units'] = np.where((weather_data['heat_units'] > max_grow_temp - min_grow_temp), 1, weather_data['heat_units']/24)


    # Standardise growth data.

    growth_data['sample_date'] = pd.to_datetime(growth_data['sample_date'], format='%d/%m/%Y')
    growth_data['fieldzone'] = growth_data["field"] + " - " + growth_data["zone"].astype(str)
    growth_data['stripped_diameter'] = growth_data['diameter'] * stripping_coef


    # Standardise zone data.

    zone_data['planting_date'] = pd.to_datetime(zone_data['planting_date'], format='%d/%m/%Y')
    zone_data['harvest_date'] = pd.to_datetime(zone_data['harvest_date'], format='%d/%m/%Y')
    zone_data['zone'] = zone_data['zone'].astype(int)
    zone_data["fieldzone"] = zone_data["field"] + " - " + zone_data["zone"].astype(str)
    zone_data["fieldvariety"] = zone_data["field"] + " - " + zone_data["variety"]


    # Start to build summary dataframe
    # This dataframe will be aggregated to fit models. Need to aggregate explained later.

    summary_data = growth_data.copy()

    summary_data["zone"] = summary_data["zone"].astype(str)
    summary_data["fieldzone"] = summary_data["field"] + " - " + summary_data["zone"]
    summary_data['fieldzonedate'] = summary_data['fieldzone'] + " - " + summary_data['sample_date'].astype(str)

    summary_data = summary_data.set_index('fieldzone')
    summary_data = summary_data.join(zone_data.set_index('fieldzone'), rsuffix = '_join')

    summary_data['fieldvarietydate'] = summary_data['fieldvariety'] + " - " + summary_data['sample_date'].astype(str)
    summary_data['heat_units'] = 0
    summary_data['solar_radiation'] = 0


    # Reset index and remove join columns from summary dataframe.

    summary_data = summary_data.reset_index(inplace=False)
    summary_data = summary_data.drop(columns=['field_join', 'zone_join'], inplace=False)


    def skewness(series):
        """Aggregate function to return skew of distribution"""
        return ss.skew(series, bias = False)

    def kurt(series):
        """Aggregate function to return kurtosis of distribution"""
        return ss.kurtosis(series, bias = False)

    loading_phase = 'Aggregating Data'
    print(loading_phase)
    # Aggregation of summary data to which can be used to fit models.
    # Non aggregated data can also be used to fit models but it will be more difficult to create a predicted distribution...
    # cont... due to not having a standard deviation variable which is generated during aggregation.

    summary_data_avg = summary_data.copy()

    summary_data_avg = summary_data_avg.groupby(['fieldzonedate']).agg({'stripped_diameter' : ['mean', 'std', 'count', skewness, kurt],
                                                                    'method' : ['first'],
                                                                    'inputs' : ['first'],
                                                                    'variety' : ['first'],
                                                                    'protection' : ['first'],
                                                                    'sand' : ['mean'],
                                                                    'silt' : ['mean'],
                                                                    'clay' : ['mean'],
                                                                    'organic_matter' : ['mean'],
                                                                    'planting_rate' : ['first'],
                                                                    'planting_date' : ['first'],
                                                                    'sample_date' : ['first'],
                                                                    'fieldzone' : ['first']}).reset_index()

    summary_data_avg.columns = ['fieldzonedate',
                                'mean_diameter',
                                'std_dev_diameter',
                                'pp2m2',
                                'skewness',
                                'kurtosis',
                                'method',
                                'inputs',
                                'variety',
                                'protection',
                                'sand',
                                'silt',
                                'clay',
                                'organic_matter',
                                'planting_rate',
                                'planting_date',
                                'sample_date',
                                'fieldzone']

    summary_data_avg['field'] = summary_data_avg['fieldzone'].str.split(' - ').str[0]
    summary_data_avg['d_lin'] = (summary_data_avg['mean_diameter'])**linearisation_coef
    summary_data_avg['s_lin'] = (summary_data_avg['std_dev_diameter'])**linearisation_coef
    summary_data_avg['heat_units'] = 0
    summary_data_avg['solar_radiation'] = 0


    # Aggregation of weather data to create a more manageable dataframe as we only need day by day accuracy not hour by hour.

    weather_data_avg = weather_data.copy()

    weather_data_avg = weather_data_avg.groupby(['date']).agg({'rain' : ['sum'],
                                                        'heat_units' : ['sum'],
                                                        'solar_radiation' : ['sum'],
                                                        'wind_speed_avg' : ['mean'],
                                                        'rh' : ['mean'],
                                                        'avg_temp' : ['mean']}).reset_index()

    weather_data_avg.columns = ['date',
                                'rain',
                                'heat_units',
                                'solar_radiation',
                                'wind_speed_avg',
                                'rh',
                                'avg_temp']

    weather_data_avg['day'] = weather_data_avg.date.dt.day
    weather_data_avg['month'] = weather_data_avg.date.dt.month
    weather_data_avg['day_month'] = weather_data_avg['day'].astype(str) + " - " + weather_data_avg['month'].astype(str)

    # Further aggregation of weather to get the average weather for a given day and a given month, regardless of the year.
    # This average weather will be used to predict heat and solar that a plant will receive.
    # Could possibly introduce some sort of short term weather forecast here???

    weather_data_avg_group = weather_data_avg.copy()

    weather_data_avg_group = weather_data_avg_group.groupby(['day_month']).agg({'rain' : ['mean'],
                                                                                'heat_units' : ['mean'],
                                                                                'solar_radiation' : ['mean'],
                                                                                'wind_speed_avg' : ['mean'],
                                                                                'rh' : ['mean'],
                                                                                'avg_temp' : ['mean']}).reset_index()

    weather_data_avg_group.columns = ['day_month',
                                    'rain',
                                    'heat_units',
                                    'solar_radiation',
                                    'wind_speed_avg',
                                    'rh',
                                    'avg_temp']

    loading_phase = 'Predicting Weather'
    print(loading_phase)

    # Extend aggregated weather data by adding 300 days of predicted weather.

    max_date = max(weather_data_avg.date)

    for i in range(1, 300):
        
        date = max_date + timedelta(days=i)
        weather_data_avg = weather_data_avg.append({'date': date,
                                                    'rain': np.nan,
                                                    'heat_units':np.nan,
                                                    'solar_radiation':np.nan,
                                                    'wind_speed_avg':np.nan,
                                                    'rh':np.nan,
                                                    'avg_temp':np.nan }, ignore_index=True)
        
        
    weather_data_avg['day'] = weather_data_avg.date.dt.day
    weather_data_avg['month'] = weather_data_avg.date.dt.month
    weather_data_avg['day_month'] = weather_data_avg['day'].astype(str) + " - " + weather_data_avg['month'].astype(str)


    ###THIS IS A VERY SLOW PROCESS (COULD PARALLEL PROCESSING BE INTRODUCED???)

    def mean_weather(day_month, variable):
        """function that calculates the average value for a weather variable for a given day within a given month"""
        df = weather_data_avg_group[weather_data_avg_group['day_month']==day_month]
        weather_value = df[variable].sum()
        return weather_value

    for variable in ['rain', 'heat_units', 'solar_radiation', 'wind_speed_avg', 'rh', 'avg_temp']:
        for i in weather_data_avg.index:
            
            # If statement so only future dates have weather predicted.
            # Tried == np.nan, but that didn't work so workaround implemented. Make sure no variable has a value larger than 'MEGA_NUM'.
            
            MEGA_NUM = 100000
            
            if weather_data_avg[variable][i]//MEGA_NUM != 0:
                day_month = weather_data_avg['day_month'][i]
                weather_data_avg[variable][i] = mean_weather(day_month, variable)


    def cumulative_weather(start, finish, weather_variable, weather_data):
        """Function used to find the cumulative weather input between 2 given dates"""
        df = weather_data.loc[(weather_data['date'] > start) & (weather_data['date'] < finish), [weather_variable]]
        total_units = df[weather_variable].sum()
        return total_units

    loading_phase = 'Creating Features'
    print(loading_phase)

    # Overwrite solar radiation with the cumulative total that has been received between planting and sampling.

    for i in summary_data_avg.index:
        
        planting_date = summary_data_avg['planting_date'][i]
        sample_date = summary_data_avg['sample_date'][i]
        summary_data_avg['solar_radiation'][i] = cumulative_weather(planting_date, sample_date, 'solar_radiation', weather_data_avg)


    # Overwrite heat units with the cumulative total that has been received between planting and sampling.

    for i in summary_data_avg.index:
        
        planting_date = summary_data_avg['planting_date'][i]
        sample_date = summary_data_avg['sample_date'][i]
        summary_data_avg['heat_units'][i] = cumulative_weather(planting_date, sample_date, 'heat_units', weather_data_avg)


    def average_count(fieldzone, df_1 = summary_data_avg):
        """Function used to find the average plants per two meters squared from every sample from a given fieldzone over the entire season"""
        
        df_1 = df_1[df_1['fieldzone']==fieldzone]
        average_count = df_1['pp2m2'].mean()
        
        if mt.isnan(average_count):
            average_count = 40
        
        return average_count


    def max_sample_date(fieldzone, df_1 = summary_data_avg, df_2 = zone_data):
        """Function used to find the most recent sample date for a given fieldzone"""
        
        df_1 = df_1[df_1['fieldzone'] == fieldzone]
        max_sample_date = max(df_1['sample_date'], default = 0)
        if max_sample_date == 0:
            df_2 = df_2[df_2['fieldzone'] == fieldzone]
            max_sample_date = df_2['planting_date'].max()
        
        return max_sample_date


    def max_mean_diameter_lin(fieldzone, df_1 = summary_data_avg):
        """Function used to find the mean diameter of the sample at the most recent sample date"""
        
        df_1 = df_1[df_1['fieldzone']==fieldzone]
        max_mean_diameter = df_1['mean_diameter'].max()
        max_mean_diameter_lin = max_mean_diameter ** linearisation_coef
        
        if mt.isnan(max_mean_diameter_lin):
            max_mean_diameter_lin = 0
        
        return max_mean_diameter_lin


    def max_std_dev_diameter_lin(fieldzone, df_1 = summary_data_avg):
        """Function used to find the standard deviation of the sample at the most recent sample date"""
        
        df_1 = df_1[df_1['fieldzone']==fieldzone]
        max_std_dev_diameter = df_1['std_dev_diameter'].max()
        max_std_dev_diameter_lin = max_std_dev_diameter ** linearisation_coef
        
        if mt.isnan(max_std_dev_diameter_lin):
            max_std_dev_diameter_lin = 0
        
        return max_std_dev_diameter_lin


    def max_solar(fieldzone, df_1 = summary_data_avg):
        """Function used to find the solar radiation received at the most recent sample date"""
        
        df_1 = df_1[df_1['fieldzone']==fieldzone]
        max_solar = df_1['solar_radiation'].max()
        
        if mt.isnan(max_solar):
            max_solar = 0
        
        return max_solar


    def max_heat(fieldzone, df_1 = summary_data_avg):
        """Function used to find the heat units received at the most recent sample date"""
        
        df_1 = df_1[df_1['fieldzone']==fieldzone]
        max_heat = df_1['heat_units'].max()
        
        if mt.isnan(max_heat):
            max_heat = 0
        
        return max_heat


    def filter_data(data, method, inputs, variety):
        """Function used to filter df so it only contains a single variety, input & method"""
        
        filtered = data[data['variety'].str.contains(variety)]
        filtered = filtered[filtered['inputs'].str.contains(inputs)]
        filtered = filtered[filtered['method'].str.contains(method)]
        

    def predict_weather(start, finish, variable, df_1 = weather_data):
        """Function used to calculated a predicted weather variable for a given timeframe"""
        
        df_1 = df_1.loc[(df_1['date'] > start) & (df_1['date'] < finish), [variable]]
        predicted_weather= df_1[variable].sum()
        
        return predicted_weather


    # This df will be used for Linear Regression model and Visualisation. Prediction will be taken from the most recent (maximum) sample date...
    # The 'max' variables just indicate the result of the most recent sample at that zone.

    zone_data['mean_pp2m2'] = 0.0
    zone_data['max_sample_date'] = 0
    zone_data['max_mean_diameter_lin'] = 0.0
    zone_data['max_std_dev_diameter_lin'] = 0.0
    zone_data['max_heat'] = 0.0
    zone_data['max_solar'] = 0.0
    zone_data['remaining_heat'] = 0.0
    zone_data['remaining_solar'] = 0.0
    zone_data['rain_after_planting'] = 0

    for i in zone_data.index:
        
        fieldzone = zone_data.loc[i, 'fieldzone']
        zone_data.loc[i, 'mean_pp2m2'] = average_count(fieldzone)
        zone_data.loc[i, 'max_sample_date'] = max_sample_date(fieldzone)
        zone_data.loc[i, 'max_mean_diameter_lin'] = max_mean_diameter_lin(fieldzone)
        zone_data.loc[i, 'max_std_dev_diameter_lin'] = max_std_dev_diameter_lin(fieldzone)
        zone_data.loc[i, 'max_heat'] = max_heat(fieldzone)
        zone_data.loc[i, 'max_solar'] = max_solar(fieldzone)

        start = zone_data.loc[i, 'max_sample_date']
        finish = zone_data.loc[i, 'harvest_date']
        rain_start = zone_data.loc[i, 'planting_date']
        rain_finish = rain_start + timedelta(days = 14)

        zone_data.loc[i, 'remaining_heat'] = cumulative_weather(start, finish, 'heat_units', weather_data_avg)
        zone_data.loc[i, 'remaining_solar'] = cumulative_weather(start, finish, 'solar_radiation', weather_data_avg)
        zone_data.loc[i, 'rain_after_planting'] = cumulative_weather(rain_start, rain_finish, 'rain', weather_data_avg)
        
    zone_data['establishment'] = (zone_data['mean_pp2m2']/2*10000)/zone_data['planting_rate']

    loading_phase = 'Completed Loading'
    print(loading_phase)

    writer = pd.ExcelWriter('estimator_test.xlsx', engine='xlsxwriter')


    summary_data_avg.to_excel(writer, sheet_name="summary_data_avg", index=False)
    weather_data_avg.to_excel(writer, sheet_name="weather_data_avg", index=False)
    zone_data.to_excel(writer, sheet_name="zone_data", index=False)
    writer.save()


    print("Process finished --- %s seconds ---" % (time.time() - start_time))




























